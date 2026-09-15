import math
from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence, Tuple

import torch


def _as_sorted_unique_tensor(indices: Optional[Sequence[int]],
                             device: torch.device) -> torch.Tensor:
    if not indices:
        return torch.empty(0, dtype=torch.long, device=device)
    return torch.tensor(sorted(set(int(i) for i in indices)), dtype=torch.long, device=device)


def _valid_mask(n: int, exclude_indices: Optional[Sequence[int]],
                device: torch.device) -> torch.Tensor:
    mask = torch.ones(n, dtype=torch.bool, device=device)
    exclude = _as_sorted_unique_tensor(exclude_indices, device)
    if exclude.numel() > 0:
        exclude = exclude[(exclude >= 0) & (exclude < n)]
        mask[exclude] = False
    return mask


def _row_l2_normalize(x: torch.Tensor) -> torch.Tensor:
    return x.float() / x.float().norm(dim=-1, keepdim=True).clamp(min=1e-8)


def _row_softmax_stable(x: torch.Tensor) -> torch.Tensor:
    x = x.float()
    x = x - x.max(dim=1, keepdim=True).values
    exp_x = torch.exp(x)
    return exp_x / exp_x.sum(dim=1, keepdim=True).clamp(min=1e-12)


def build_matrices(
    v_pre: torch.Tensor,
    v_post: torch.Tensor,
    t_raw: torch.Tensor,
    tau_t: float,
    tau_v: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if v_pre.dim() != 2 or v_post.dim() != 2:
        raise ValueError("StackTok expects single-crop tensors shaped [n, d].")
    if v_pre.shape[0] != v_post.shape[0]:
        raise ValueError(
            f"V_pre and V_post token counts differ: {v_pre.shape[0]} vs {v_post.shape[0]}")

    vp = _row_l2_normalize(v_pre)
    vq = _row_l2_normalize(v_post)
    if t_raw.numel() == 0:
        mtv = torch.empty((0, v_post.shape[0]), dtype=torch.float32, device=v_post.device)
    else:
        t = _row_l2_normalize(t_raw.to(device=v_post.device))
        mtv = _row_softmax_stable((t @ vq.T) / float(tau_t))
    mvv = _row_softmax_stable((vp @ vp.T) / float(tau_v))
    return mtv, mvv


@dataclass
class CovState:
    M: torch.Tensor

    def __post_init__(self) -> None:
        self.M = self.M.float()
        self.cmax = torch.zeros(self.M.shape[0], dtype=torch.float32, device=self.M.device)

    def f(self) -> torch.Tensor:
        if self.cmax.numel() == 0:
            return torch.zeros((), dtype=torch.float32, device=self.M.device)
        return self.cmax.mean()

    def gains_all(self, cand: torch.Tensor) -> torch.Tensor:
        if cand.numel() == 0:
            return torch.empty(0, dtype=torch.float32, device=self.M.device)
        if self.M.shape[0] == 0:
            return torch.zeros(cand.numel(), dtype=torch.float32, device=self.M.device)
        return (self.M.index_select(1, cand) - self.cmax[:, None]).clamp_min(0).mean(dim=0)

    def gain_one(self, s: int) -> torch.Tensor:
        if self.M.shape[0] == 0:
            return torch.zeros((), dtype=torch.float32, device=self.M.device)
        return (self.M[:, int(s)] - self.cmax).clamp_min(0).mean()

    def add(self, s: int) -> None:
        if self.M.shape[0] > 0:
            self.cmax = torch.maximum(self.cmax, self.M[:, int(s)])


class Cov2State:

    def __init__(self, M: torch.Tensor):
        self.M = M.float()
        device = self.M.device
        self.max1 = torch.zeros(self.M.shape[0], dtype=torch.float32, device=device)
        self.max2 = torch.zeros(self.M.shape[0], dtype=torch.float32, device=device)
        self.arg1 = torch.full((self.M.shape[0], ), -1, dtype=torch.long, device=device)

    def build(self, S: Sequence[int]) -> "Cov2State":
        device = self.M.device
        if self.M.shape[0] == 0:
            self.max1 = torch.empty(0, dtype=torch.float32, device=device)
            self.max2 = torch.empty(0, dtype=torch.float32, device=device)
            self.arg1 = torch.empty(0, dtype=torch.long, device=device)
            return self
        if not S:
            self.max1 = torch.zeros(self.M.shape[0], dtype=torch.float32, device=device)
            self.max2 = torch.zeros(self.M.shape[0], dtype=torch.float32, device=device)
            self.arg1 = torch.full((self.M.shape[0], ), -1, dtype=torch.long, device=device)
            return self

        S_t = torch.tensor(list(S), dtype=torch.long, device=device)
        sub = self.M.index_select(1, S_t)
        if len(S) == 1:
            self.max1 = sub[:, 0]
            self.max2 = torch.zeros_like(self.max1)
            self.arg1 = torch.full((self.M.shape[0], ), int(S[0]), dtype=torch.long, device=device)
            return self

        vals, local_idx = torch.topk(sub, k=2, dim=1, largest=True, sorted=True)
        self.max1 = vals[:, 0]
        self.max2 = vals[:, 1]
        self.arg1 = S_t.index_select(0, local_idx[:, 0])
        return self

    def f(self) -> torch.Tensor:
        if self.max1.numel() == 0:
            return torch.zeros((), dtype=torch.float32, device=self.M.device)
        return self.max1.mean()

    def cmax_without(self, s: int) -> torch.Tensor:
        if self.max1.numel() == 0:
            return torch.empty(0, dtype=torch.float32, device=self.M.device)
        return torch.where(self.arg1 == int(s), self.max2, self.max1)


def _candidate_indices(valid: torch.Tensor, selected: Iterable[int]) -> torch.Tensor:
    mask = valid.clone()
    selected_list = list(selected)
    if selected_list:
        selected_t = torch.tensor(selected_list, dtype=torch.long, device=valid.device)
        mask[selected_t] = False
    return torch.nonzero(mask, as_tuple=False).flatten()


def vv_reference(Mvv: torch.Tensor,
                 k_max: int,
                 valid: Optional[torch.Tensor] = None) -> List[float]:
    n = Mvv.shape[1]
    device = Mvv.device
    valid = torch.ones(n, dtype=torch.bool, device=device) if valid is None else valid.clone()
    k_max = min(int(k_max), int(valid.sum().item()))
    st = CovState(Mvv)
    selected: List[int] = []
    G = [0.0]
    for _ in range(k_max):
        cand = _candidate_indices(valid, selected)
        if cand.numel() == 0:
            break
        gains = st.gains_all(cand)
        best = int(cand[int(torch.argmax(gains).item())].item())
        selected.append(best)
        st.add(best)
        G.append(float(st.f().item()))
    while len(G) <= k_max:
        G.append(float(st.f().item()))
    return G


def compute_beta(Mtv: torch.Tensor, n: int, beta_min: float = 0.3, beta_max: float = 0.9) -> float:
    if Mtv.shape[0] == 0:
        return float(beta_max)
    if n <= 1:
        return float(beta_min)
    H = -(Mtv * torch.log(Mtv + 1e-12)).sum(dim=1)
    beta = H.mean() / math.log(float(n))
    return float(torch.clamp(beta, min=float(beta_min), max=float(beta_max)).item())


def _max_positive(gains: torch.Tensor) -> bool:
    return gains.numel() > 0 and bool((gains.max() > 0).item())


def _fallback_gains(
    primary: CovState,
    secondary: CovState,
    cand: torch.Tensor,
    use_both: bool = True,
) -> torch.Tensor:
    """Return a deterministic score vector even when all marginal gains are zero."""
    gains = primary.gains_all(cand)
    if _max_positive(gains):
        return gains

    secondary_gains = secondary.gains_all(cand)
    if _max_positive(secondary_gains):
        return secondary_gains

    if use_both:
        combined = gains + secondary_gains
        if combined.numel() > 0:
            return combined
    return gains


class StackTokSelector:

    def __init__(
        self,
        target_vision_tokens: int = 64,
        beta_min: float = 0.3,
        beta_max: float = 0.9,
        swap_mode: str = "auto",
        epsilon_swap: float = 1e-6,
        swap_passes: int = 1,
        swap_auto_max_k: int = 16,
        enable_global_multicrop: bool = True,
    ):
        if swap_mode not in {"auto", "true", "false"}:
            raise ValueError("swap_mode must be one of: auto, true, false")
        self.target_vision_tokens = int(target_vision_tokens)
        self.beta_min = float(beta_min)
        self.beta_max = float(beta_max)
        self.swap_mode = swap_mode
        self.epsilon_swap = float(epsilon_swap)
        self.swap_passes = int(swap_passes)
        self.swap_auto_max_k = int(swap_auto_max_k)
        self.enable_global_multicrop = bool(enable_global_multicrop)

    def _should_swap(self, k: int) -> bool:
        if self.swap_mode == "true":
            return True
        if self.swap_mode == "false":
            return False
        return k <= self.swap_auto_max_k

    def stacktok_single(
        self,
        text_token_embedding: torch.Tensor,
        vision_tokens: torch.Tensor,
        vision_tokens_clip: torch.Tensor,
        tv_temp: float = 0.02,
        vv_temp: float = 0.2,
        padding_patch_indices: Optional[Sequence[int]] = None,
    ) -> Tuple[List[int], torch.Tensor, dict]:
        Mtv, Mvv = build_matrices(vision_tokens_clip, vision_tokens, text_token_embedding, tv_temp,
                                  vv_temp)
        n = Mvv.shape[0]
        valid = _valid_mask(n, padding_patch_indices, Mvv.device)
        k_max = min(self.target_vision_tokens, int(valid.sum().item()))
        if k_max <= 0:
            return [], vision_tokens[:0], {"beta": self.beta_max, "selected_before_sort": []}

        G = vv_reference(Mvv, k_max, valid)
        beta = compute_beta(Mtv, n, self.beta_min, self.beta_max)
        stO = CovState(Mtv)
        stP = CovState(Mvv)
        S: List[int] = []
        has_text = Mtv.shape[0] > 0

        for _ in range(k_max):
            cand = _candidate_indices(valid, S)
            if cand.numel() == 0:
                break
            stable = bool((stP.f() >= beta * G[len(S)]).item())
            if not has_text:
                stable = False

            primary = stO if stable else stP
            secondary = stP if stable else stO
            gains = _fallback_gains(primary, secondary, cand, use_both=has_text)
            s_i = int(cand[int(torch.argmax(gains).item())].item())

            S.append(s_i)
            stO.add(s_i)
            stP.add(s_i)

        if self._should_swap(len(S)):
            S = self.swap_phase(S, Mtv, Mvv, beta, G, valid)
        selected_indices = sorted(S)
        selected_tokens = vision_tokens[torch.tensor(selected_indices,
                                                     dtype=torch.long,
                                                     device=vision_tokens.device)]
        return selected_indices, selected_tokens, {"beta": beta, "G": G, "selected_before_sort": S}

    def stacktok_multicrop(
        self,
        text_token_embedding: torch.Tensor,
        vision_tokens_by_crop: Sequence[torch.Tensor],
        vision_tokens_clip_by_crop: Sequence[torch.Tensor],
        tv_temp: float = 0.02,
        vv_temp: float = 0.2,
        padding_patch_indices_by_crop: Optional[Sequence[Optional[Sequence[int]]]] = None,
    ) -> Tuple[List[List[int]], List[torch.Tensor], dict]:
        C = len(vision_tokens_by_crop)
        if C == 0:
            return [], [], {}
        if C == 1 or not self.enable_global_multicrop:
            indices, tokens, info = self.stacktok_single(
                text_token_embedding,
                vision_tokens_by_crop[0],
                vision_tokens_clip_by_crop[0],
                tv_temp,
                vv_temp,
                None if not padding_patch_indices_by_crop else padding_patch_indices_by_crop[0],
            )
            return [indices], [tokens], {"crops": [info]}

        padding_patch_indices_by_crop = padding_patch_indices_by_crop or [None] * C
        crop = []
        for c, (v_post, v_pre) in enumerate(zip(vision_tokens_by_crop, vision_tokens_clip_by_crop)):
            Mtv, Mvv = build_matrices(v_pre, v_post, text_token_embedding, tv_temp, vv_temp)
            valid = _valid_mask(Mvv.shape[0], padding_patch_indices_by_crop[c], Mvv.device)
            crop.append({
                "Mtv":
                Mtv,
                "Mvv":
                Mvv,
                "valid":
                valid,
                "G":
                vv_reference(Mvv, min(self.target_vision_tokens, int(valid.sum().item())), valid),
                "beta":
                compute_beta(Mtv, Mvv.shape[0], self.beta_min, self.beta_max),
                "stO":
                CovState(Mtv),
                "stP":
                CovState(Mvv),
                "S": [],
                "cache":
                "STALE",
            })

        K_total = min(self.target_vision_tokens,
                      sum(int(item["valid"].sum().item()) for item in crop))
        has_text = text_token_embedding.numel() > 0

        def nominate(c: int):
            item = crop[c]
            cand = _candidate_indices(item["valid"], item["S"])
            if cand.numel() == 0:
                return None
            stable = bool((item["stP"].f() >= item["beta"] * item["G"][len(item["S"])]).item())
            if not has_text:
                stable = False
            primary = item["stO"] if stable else item["stP"]
            secondary = item["stP"] if stable else item["stO"]
            gains = _fallback_gains(primary, secondary, cand, use_both=has_text)
            s_star = int(cand[int(torch.argmax(gains).item())].item())
            g = item["stP"].gain_one(s_star) if not has_text else item["stO"].gain_one(
                s_star) + item["stP"].gain_one(s_star)
            return s_star, g

        for _ in range(K_total):
            for c, item in enumerate(crop):
                if item["cache"] == "STALE":
                    item["cache"] = nominate(c)
            valid_nominees = [(c, item["cache"]) for c, item in enumerate(crop)
                              if item["cache"] is not None]
            if not valid_nominees:
                break
            c_star, (s_star, g) = max(valid_nominees,
                                      key=lambda pair: (float(pair[1][1].item()), -pair[0]))
            item = crop[c_star]
            item["S"].append(int(s_star))
            item["stO"].add(int(s_star))
            item["stP"].add(int(s_star))
            item["cache"] = "STALE"

        indices_by_crop: List[List[int]] = []
        tokens_by_crop: List[torch.Tensor] = []
        infos = []
        for c, item in enumerate(crop):
            S = item["S"]
            if self._should_swap(len(S)):
                S = self.swap_phase(S, item["Mtv"], item["Mvv"], item["beta"], item["G"],
                                    item["valid"])
            selected_indices = sorted(S)
            idx_t = torch.tensor(selected_indices,
                                 dtype=torch.long,
                                 device=vision_tokens_by_crop[c].device)
            indices_by_crop.append(selected_indices)
            tokens_by_crop.append(vision_tokens_by_crop[c].index_select(0, idx_t) if idx_t.numel(
            ) else vision_tokens_by_crop[c][:0])
            infos.append({"beta": item["beta"], "G": item["G"], "selected_before_sort": S})
        return indices_by_crop, tokens_by_crop, {"crops": infos, "K_total": K_total}

    def swap_phase(
        self,
        S: Sequence[int],
        Mtv: torch.Tensor,
        Mvv: torch.Tensor,
        beta: float,
        G: Sequence[float],
        valid: Optional[torch.Tensor] = None,
    ) -> List[int]:
        S = list(int(s) for s in S)
        k = len(S)
        n = Mvv.shape[1]
        if valid is None:
            valid = torch.ones(n, dtype=torch.bool, device=Mvv.device)
        if k == 0 or k >= int(valid.sum().item()):
            return S

        T2O = Cov2State(Mtv).build(S)
        T2P = Cov2State(Mvv).build(S)
        F_cur = T2O.f() + T2P.f()

        for _ in range(self.swap_passes):
            improved = False
            for p in range(k):
                s_out = S[p]
                wO = T2O.cmax_without(s_out)
                wP = T2P.cmax_without(s_out)
                cand = _candidate_indices(valid, S)
                if cand.numel() == 0:
                    continue

                if Mtv.shape[0] == 0:
                    O_new = torch.zeros(cand.numel(), dtype=torch.float32, device=Mvv.device)
                else:
                    O_new = torch.maximum(wO[:, None], Mtv.index_select(1, cand)).mean(dim=0)
                P_new = torch.maximum(wP[:, None], Mvv.index_select(1, cand)).mean(dim=0)
                F_new = O_new + P_new

                floor = min(float(beta) * float(G[k]), float(T2P.f().item()))
                feas = P_new >= floor
                if not bool(feas.any().item()):
                    continue
                feasible_scores = F_new.masked_fill(~feas, float("-inf"))
                best_pos = int(torch.argmax(feasible_scores).item())
                if bool((F_new[best_pos] > F_cur * (1.0 + self.epsilon_swap)).item()):
                    S[p] = int(cand[best_pos].item())
                    T2O.build(S)
                    T2P.build(S)
                    F_cur = T2O.f() + T2P.f()
                    improved = True
            if not improved:
                break
        return S
