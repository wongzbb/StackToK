from stacktok.core import StackTokSelector, build_matrices, compute_beta
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _device():
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def test_single_with_swap_and_padding():
    device = _device()
    torch.manual_seed(7)
    n, d_pre, d_post, m = 32, 8, 16, 5
    v_pre = torch.randn(n, d_pre, device=device, dtype=torch.float16)
    v_post = torch.randn(n, d_post, device=device, dtype=torch.float16)
    t_raw = torch.randn(m, d_post, device=device, dtype=torch.float16)
    selector = StackTokSelector(target_vision_tokens=12, swap_mode="true", swap_passes=1)
    indices, tokens, info = selector.stacktok_single(
        t_raw,
        v_post,
        v_pre,
        tv_temp=0.02,
        vv_temp=0.2,
        padding_patch_indices=[0, 1, 31],
    )
    assert indices == sorted(indices)
    assert len(indices) == 12
    assert len(indices) == len(set(indices))
    assert not ({0, 1, 31} & set(indices))
    assert tokens.shape == (len(indices), d_post)
    assert 0.3 <= info["beta"] <= 0.9


def test_empty_query_degenerates_to_vv_path():
    device = _device()
    torch.manual_seed(11)
    v_pre = torch.randn(20, 6, device=device)
    v_post = torch.randn(20, 10, device=device)
    t_raw = torch.empty(0, 10, device=device)
    selector = StackTokSelector(target_vision_tokens=8, swap_mode="false")
    indices, tokens, info = selector.stacktok_single(t_raw, v_post, v_pre)
    assert len(indices) == 8
    assert tokens.shape[0] == len(indices)
    assert info["beta"] == 0.9


def test_force_full_k_when_all_later_gains_are_zero():
    device = _device()
    v_pre = torch.zeros(12, 6, device=device)
    v_post = torch.zeros(12, 10, device=device)
    t_raw = torch.empty(0, 10, device=device)
    selector = StackTokSelector(target_vision_tokens=5, swap_mode="false")
    indices, tokens, info = selector.stacktok_single(t_raw, v_post, v_pre)
    assert len(indices) == 5
    assert indices == sorted(indices)
    assert len(indices) == len(set(indices))
    assert tokens.shape == (5, 10)


def test_multicrop_global_budget_and_ragged_output():
    device = _device()
    torch.manual_seed(17)
    C, n, d_pre, d_post, m = 3, 24, 8, 16, 4
    crops_post = [torch.randn(n, d_post, device=device) for _ in range(C)]
    crops_pre = [torch.randn(n, d_pre, device=device) for _ in range(C)]
    t_raw = torch.randn(m, d_post, device=device)
    selector = StackTokSelector(target_vision_tokens=18, swap_mode="auto", swap_auto_max_k=16)
    indices_by_crop, tokens_by_crop, info = selector.stacktok_multicrop(
        t_raw, crops_post, crops_pre)
    total = sum(len(idx) for idx in indices_by_crop)
    assert total == 18
    assert len(indices_by_crop) == C
    assert len(tokens_by_crop) == C
    for indices, tokens in zip(indices_by_crop, tokens_by_crop):
        assert indices == sorted(indices)
        assert len(indices) == len(set(indices))
        assert tokens.shape[0] == len(indices)
        assert tokens.shape[-1] == d_post
    assert info["K_total"] == 18


def test_beta_and_matrices_are_float32():
    device = _device()
    torch.manual_seed(23)
    v_pre = torch.randn(12, 5, device=device, dtype=torch.float16)
    v_post = torch.randn(12, 9, device=device, dtype=torch.float16)
    t_raw = torch.randn(3, 9, device=device, dtype=torch.float16)
    mtv, mvv = build_matrices(v_pre, v_post, t_raw, 0.02, 0.2)
    assert mtv.dtype == torch.float32
    assert mvv.dtype == torch.float32
    assert torch.allclose(mtv.sum(dim=1), torch.ones(3, device=device), atol=1e-5)
    assert torch.allclose(mvv.sum(dim=1), torch.ones(12, device=device), atol=1e-5)
    beta = compute_beta(mtv, 12)
    assert 0.3 <= beta <= 0.9


if __name__ == "__main__":
    test_single_with_swap_and_padding()
    test_empty_query_degenerates_to_vv_path()
    test_force_full_k_when_all_later_gains_are_zero()
    test_multicrop_global_budget_and_ragged_output()
    test_beta_and_matrices_are_float32()
    print("StackTok selector smoke tests passed")
