from __future__ import annotations

import itertools
import math
import random
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple, Optional

from .base import ObjectiveFn, Trial


def _log_softmax(xs: List[float]) -> List[float]:
    """
    Numerically stable log-softmax for python lists.
    (We keep this file torch-light; only GRPO update uses torch.)
    """
    m = max(xs)
    exps = [math.exp(x - m) for x in xs]
    s = sum(exps)
    return [math.log(e / s) for e in exps]


@dataclass
class GRPO:
    """
    GRPO (Group Relative Policy Optimization) for discrete hyper-parameter selection.

    This is a closer-to-standard GRPO/PPO-style implementation:
    - Sample a *group/batch* of configs from current policy (same "prompt" idea -> group).
    - Compute group-relative advantages: normalize by (mean, std) within the group.
    - Do clipped-ratio policy optimization (PPO surrogate) on the sampled batch.
    - Add KL penalty to a frozen reference policy (initial policy) to prevent collapse.

    Notes for this project:
    - One "trajectory" == one config (hyperparam combination).
    - Reward comes from your weighted metric score (rouge/bertscore/meteor).
    - Budget is number of objective evaluations (train trials), so group size consumes budget.
    """

    name: str = "grpo"

    def __init__(
        self,
        space: Dict[str, Sequence],
        seed: int = 42,
        lr: float = 0.2,
        group_size: int = 4,
        clip_eps: float = 0.2,
        kl_beta: float = 0.02,
        update_epochs: int = 1,
        # --- Stabilization / exploration knobs (useful under noisy rewards) ---
        # If True, use rank-based advantages within group (robust to outliers / scale).
        use_rank_adv: bool = False,
        # Entropy bonus coefficient (encourage exploration; 0 disables).
        entropy_beta: float = 0.0,
        # Keep top-k configs seen so far and add an imitation-style loss to increase their logprob.
        # This is lightweight and often helps "lock in" good interacting combinations.
        elite_buffer_size: int = 0,
        elite_beta: float = 0.0,
        # Scheme A: merge strongly interacting knobs into a single categorical.
        # Example: joint_pairs=[("chunker","retriever")] creates a 4-way categorical knob
        # and removes the original two independent knobs from sampling.
        joint_pairs: Sequence[Tuple[str, str]] = (),
    ):
        base_space: Dict[str, List] = {k: list(v) for k, v in space.items()}
        self.rng = random.Random(seed)
        self.lr = float(lr)
        self.group_size = int(group_size)
        self.clip_eps = float(clip_eps)
        self.kl_beta = float(kl_beta)
        self.update_epochs = int(update_epochs)
        self.use_rank_adv = bool(use_rank_adv)
        self.entropy_beta = float(entropy_beta)
        self.elite_buffer_size = int(elite_buffer_size)
        self.elite_beta = float(elite_beta)

        # --- Build joint knobs (Scheme A) ---
        # We keep sampling space in self.space, but decode joint knobs into the original cfg keys.
        self._joint_decode: Dict[str, Tuple[str, str, List[Tuple]]] = {}
        joint_keys_to_remove = set()
        for a, b in joint_pairs or ():
            if a == b:
                continue
            if a not in base_space or b not in base_space:
                continue
            joint_key = f"{a}__X__{b}"
            # 4-way (or generally |a|*|b|) categorical: list of (a_val, b_val)
            joint_choices = list(itertools.product(base_space[a], base_space[b]))
            self._joint_decode[joint_key] = (a, b, joint_choices)
            joint_keys_to_remove.add(a)
            joint_keys_to_remove.add(b)

        # Final sampling space: original keys excluding those merged, plus the joint keys.
        self.space: Dict[str, List] = {}
        for k, v in base_space.items():
            if k in joint_keys_to_remove:
                continue
            self.space[k] = v
        for joint_key, (_a, _b, joint_choices) in self._joint_decode.items():
            self.space[joint_key] = joint_choices

        # Python-side logits (used for sampling / logging); torch-side params are created lazily in search().
        self.logits: Dict[str, List[float]] = {k: [0.0 for _ in v] for k, v in self.space.items()}
        # Frozen reference policy logits (initial policy).
        self.ref_logits: Dict[str, List[float]] = {k: list(self.logits[k]) for k in self.space.keys()}

    def _sample_with_indices(self) -> Tuple[Dict, Dict[str, int], float]:
        """
        Returns: (config, chosen_index_per_key, logprob_under_current_policy)
        """
        cfg: Dict = {}
        idxs: Dict[str, int] = {}
        logp_total = 0.0
        for k, choices in self.space.items():
            logps = _log_softmax(self.logits[k])
            r = self.rng.random()
            cum = 0.0
            idx = 0
            for i, lp in enumerate(logps):
                p = math.exp(lp)
                cum += p
                if r <= cum:
                    idx = i
                    break
            # Decode joint knobs into original keys.
            if k in self._joint_decode:
                a, b, joint_choices = self._joint_decode[k]
                av, bv = joint_choices[idx]
                cfg[a] = av
                cfg[b] = bv
            else:
                cfg[k] = choices[idx]
            idxs[k] = idx
            logp_total += logps[idx]
        # validity guard
        if "chunk_overlap" in cfg and "chunk_size" in cfg:
            if int(cfg["chunk_overlap"]) >= int(cfg["chunk_size"]):
                cfg["chunk_overlap"] = min(int(cfg["chunk_overlap"]), int(cfg["chunk_size"]) - 1)
        return cfg, idxs, logp_total

    def search(self, *, objective: ObjectiveFn, budget: int) -> List[Trial]:
        if budget <= 0:
            return []

        if self.group_size <= 0:
            raise ValueError("group_size must be > 0")

        # Use torch for the PPO/GRPO update (autograd is simpler + less error-prone).
        import torch

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        params: Dict[str, torch.nn.Parameter] = {}
        ref_probs: Dict[str, torch.Tensor] = {}
        for k, choices in self.space.items():
            p = torch.nn.Parameter(torch.tensor(self.logits[k], dtype=torch.float32, device=device))
            params[k] = p
            rp = torch.softmax(torch.tensor(self.ref_logits[k], dtype=torch.float32, device=device), dim=-1)
            ref_probs[k] = rp

        opt = torch.optim.Adam(params.values(), lr=self.lr)

        trials: List[Trial] = []
        used = 0
        # Elite buffer: store (reward, idxs) for best configs seen so far.
        elite: List[Tuple[float, Dict[str, int]]] = []

        while used < budget:
            g = min(self.group_size, budget - used)
            batch_cfgs: List[Dict] = []
            batch_idxs: List[Dict[str, int]] = []
            batch_logp_old: List[float] = []
            batch_rewards: List[float] = []
            batch_trials: List[Trial] = []

            # 1) sample a group/batch
            for _ in range(g):
                cfg, idxs, logp = self._sample_with_indices()
                tr = objective(cfg)
                batch_cfgs.append(cfg)
                batch_idxs.append(idxs)
                batch_logp_old.append(logp)
                batch_rewards.append(float(tr.reward))
                batch_trials.append(tr)

            trials.extend(batch_trials)
            used += g

            # 2) group-relative advantages (normalize within group)
            r = torch.tensor(batch_rewards, dtype=torch.float32, device=device)
            if self.use_rank_adv:
                # Map rewards to normalized ranks in [-1, 1] (robust under noisy/outlier rewards).
                # rank 0 = worst, rank g-1 = best.
                ranks = torch.argsort(torch.argsort(r))
                denom = max(1.0, float(g - 1))
                adv = (ranks.to(torch.float32) / denom) * 2.0 - 1.0
            else:
                mean = torch.mean(r)
                std = torch.std(r, unbiased=False)
                adv = (r - mean) / (std + 1e-8)

            logp_old = torch.tensor(batch_logp_old, dtype=torch.float32, device=device).detach()

            # Update elite buffer with current batch bests.
            if self.elite_buffer_size > 0 and self.elite_beta > 0.0:
                for rew, idxs in zip(batch_rewards, batch_idxs):
                    elite.append((float(rew), dict(idxs)))
                elite.sort(key=lambda x: x[0], reverse=True)
                if len(elite) > self.elite_buffer_size:
                    elite = elite[: self.elite_buffer_size]

            # 3) PPO/GRPO clipped objective + KL(reference) penalty
            for _epoch in range(max(1, self.update_epochs)):
                opt.zero_grad(set_to_none=True)

                # compute logp_new for each sampled config under current params
                logp_new_list: List[torch.Tensor] = []
                for idxs in batch_idxs:
                    lp = torch.tensor(0.0, dtype=torch.float32, device=device)
                    for k, i in idxs.items():
                        lp_k = torch.log_softmax(params[k], dim=-1)[int(i)]
                        lp = lp + lp_k
                    logp_new_list.append(lp)
                logp_new = torch.stack(logp_new_list, dim=0)

                ratio = torch.exp(logp_new - logp_old)
                unclipped = ratio * adv
                clipped = torch.clamp(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * adv
                policy_loss = -torch.mean(torch.minimum(unclipped, clipped))

                # KL(current || reference) across each categorical
                kl_terms: List[torch.Tensor] = []
                for k in self.space.keys():
                    p_cur = torch.softmax(params[k], dim=-1)
                    p_ref = ref_probs[k]
                    kl = torch.sum(p_cur * (torch.log(p_cur + 1e-12) - torch.log(p_ref + 1e-12)))
                    kl_terms.append(kl)
                kl_pen = torch.sum(torch.stack(kl_terms, dim=0))

                # Entropy bonus (encourage exploration; useful when rewards are sparse/noisy).
                ent_bonus = torch.tensor(0.0, dtype=torch.float32, device=device)
                if self.entropy_beta > 0.0:
                    ent_terms: List[torch.Tensor] = []
                    for k in self.space.keys():
                        p_cur = torch.softmax(params[k], dim=-1)
                        ent = -torch.sum(p_cur * torch.log(p_cur + 1e-12))
                        ent_terms.append(ent)
                    ent_bonus = torch.sum(torch.stack(ent_terms, dim=0))

                # Elite imitation loss: increase logprob of best configs seen so far (simple, often effective).
                elite_loss = torch.tensor(0.0, dtype=torch.float32, device=device)
                if elite and self.elite_beta > 0.0:
                    elite_logps: List[torch.Tensor] = []
                    for _rew, e_idxs in elite:
                        lp = torch.tensor(0.0, dtype=torch.float32, device=device)
                        for k, i in e_idxs.items():
                            lp_k = torch.log_softmax(params[k], dim=-1)[int(i)]
                            lp = lp + lp_k
                        elite_logps.append(lp)
                    elite_lp = torch.stack(elite_logps, dim=0)
                    elite_loss = -torch.mean(elite_lp)

                loss = (
                    policy_loss
                    + self.kl_beta * kl_pen
                    - self.entropy_beta * ent_bonus
                    + self.elite_beta * elite_loss
                )
                loss.backward()
                opt.step()

            # sync python logits for next sampling
            for k in self.space.keys():
                self.logits[k] = params[k].detach().cpu().tolist()

        return trials


# Backward compatible alias (older name in the project code path)
GRPOLite = GRPO


