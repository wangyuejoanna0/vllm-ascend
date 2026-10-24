# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# This file is a part of the vllm-ascend project.
"""Offline A/B comparison of VLLM_ASCEND_MOE_DEBUG_DUMP snapshots.

Compares a reference fused-MoE run (MegaMoe ``mega_*`` or DFC ``dfc_*``)
against a CAM fused_deep_moe run (``cam_*``) on the same prompt. Checks,
in order:

  1. the fused-MoE inputs are identical (x / expert_ids / expert_scales /
     mask / global_bs), so any difference must come from the operator;
  2. the two paths were fed the same INT4 weights, and if not, which N-axis
     transform maps one onto the other;
  3. output statistics and a fitted global scale factor.

Usage:
    python tools/moe_debug/compare_moe_dumps.py /tmp/dump_dfc /tmp/dump_cam
    python tools/moe_debug/compare_moe_dumps.py /tmp/dump_mega /tmp/dump_cam
    python tools/moe_debug/compare_moe_dumps.py /tmp/dump_dfc /tmp/dump_cam --rank 3 --layer 0
    python tools/moe_debug/compare_moe_dumps.py /tmp/dump_dfc /tmp/dump_cam --no-weights

Needs torch on CPU only; run it anywhere the dump files are visible.
"""

from __future__ import annotations

import argparse
import glob
import os
import re
import sys
from typing import Any, Callable

import torch

NAME_RE = re.compile(r"^(?P<tag>.+)_layer(?P<layer>\d+)_rank(?P<rank>-?\d+)\.pt$")
REF_IN_TAGS = ("dfc_in", "mega_in")
REF_OUT_TAGS = ("dfc_out", "mega_out")
CAM_IN_TAG = "cam_in"
CAM_OUT_TAG = "cam_out"


def first_tag(files: dict[tuple[str, int, int], str], tags: tuple[str, ...], layer: int, rank: int) -> tuple[str, str] | None:
    for tag in tags:
        path = files.get((tag, layer, rank))
        if path is not None:
            return tag, path
    return None


# --------------------------------------------------------------------------- #
# loading
# --------------------------------------------------------------------------- #
def discover(root: str) -> dict[tuple[str, int, int], str]:
    found: dict[tuple[str, int, int], str] = {}
    for path in sorted(glob.glob(os.path.join(root, "*.pt"))):
        match = NAME_RE.match(os.path.basename(path))
        if match is None:
            continue
        found[(match["tag"], int(match["layer"]), int(match["rank"]))] = path
    return found


def load(path: str) -> dict[str, Any]:
    return torch.load(path, map_location="cpu", weights_only=False)


def as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    return list(value) if isinstance(value, (list, tuple)) else [value]


def to_bool_mask(mask: Any) -> torch.Tensor | None:
    if mask is None:
        return None
    if not isinstance(mask, torch.Tensor):
        return None
    return mask.to(torch.bool).reshape(-1)


# --------------------------------------------------------------------------- #
# 1. input equality
# --------------------------------------------------------------------------- #
def compare_inputs(mega: dict[str, Any], cam: dict[str, Any]) -> bool:
    print("[1] fused-MoE inputs")
    ok = True

    for key in ("x", "expert_ids", "expert_scales"):
        lhs, rhs = mega.get(key), cam.get(key)
        if lhs is None or rhs is None:
            print(f"    {key:16s} missing (mega={lhs is not None}, cam={rhs is not None})")
            ok = False
            continue
        same = lhs.shape == rhs.shape and torch.equal(lhs.to(torch.float64), rhs.to(torch.float64))
        print(f"    {key:16s} shape {tuple(lhs.shape)} vs {tuple(rhs.shape)}  identical={same}")
        ok &= same

    mega_mask = to_bool_mask(mega.get("x_active_mask"))
    cam_mask = to_bool_mask(cam.get("x_active_mask"))
    if mega_mask is None and cam_mask is None:
        print("    x_active_mask    both None")
    elif mega_mask is not None and cam_mask is not None:
        same = mega_mask.shape == cam_mask.shape and torch.equal(mega_mask, cam_mask)
        print(
            f"    x_active_mask    dtype {mega.get('x_active_mask').dtype} vs "
            f"{cam.get('x_active_mask').dtype}, active {int(mega_mask.sum())}/{int(cam_mask.sum())}  identical={same}"
        )
        ok &= same
    else:
        print("    x_active_mask    MISMATCH: one side passes a mask, the other does not")
        ok = False

    for key in ("global_bs", "ep_world_size", "ep_rank_id"):
        lhs, rhs = mega.get(key), cam.get(key)
        same = lhs == rhs
        print(f"    {key:16s} {lhs} vs {rhs}  identical={same}")
        ok &= same

    print(f"    => inputs {'MATCH (difference is inside the operator)' if ok else 'DIFFER (look upstream of MoE)'}")
    return ok


# --------------------------------------------------------------------------- #
# 2. weight layout
# --------------------------------------------------------------------------- #
def unpack_int8_int4(t: torch.Tensor) -> torch.Tensor:
    """ModelSlim packing: low nibble = even N, high nibble = odd N."""
    lo = t.to(torch.int16) & 0xF
    hi = (t.to(torch.int16) >> 4) & 0xF

    def signed(x: torch.Tensor) -> torch.Tensor:
        return torch.where(x >= 8, x - 16, x).to(torch.int8)

    stacked = torch.stack([signed(lo), signed(hi)], dim=-1)
    return stacked.reshape(*t.shape[:-1], -1)


def unpack_int32_int4(t: torch.Tensor) -> torch.Tensor:
    """INT32 carrier: 8 nibbles, lowest nibble first."""
    parts = []
    for shift in range(0, 32, 4):
        nib = (t >> shift) & 0xF
        parts.append(torch.where(nib >= 8, nib - 16, nib).to(torch.int8).unsqueeze(-1))
    return torch.cat(parts, dim=-1).reshape(*t.shape[:-1], -1)


def nz_to_nd(weight: torch.Tensor) -> torch.Tensor | None:
    """Best-effort FRACTAL_NZ -> ND, matching ``convert_nd_to_nz`` in the sample.

    The sample builds NZ with ``reshape(..., K//16, 16, N//n0, n0)`` followed by
    ``permute(*lead, -2, -4, -3, -1)``, so a per-expert tensor is laid out as
    ``[K_blk, N_blk, 16, n0]`` — block row is K, block column is N, and inside a
    fractal the 16 axis is K and ``n0`` is N (n0 = 32 for INT8, 8 for INT32).
    """
    if weight.dim() == 3:
        return weight
    if weight.dim() != 5:
        return None
    k_blocks, n_blocks, k0, n0 = weight.shape[1:]
    # [E, K_blk, N_blk, k0, n0] -> [E, K_blk, k0, N_blk, n0]
    nd = weight.permute(0, 1, 3, 2, 4).contiguous()
    return nd.reshape(weight.shape[0], k_blocks * k0, n_blocks * n0)


def half_swap(x: torch.Tensor) -> torch.Tensor:
    half = x.shape[-1] // 2
    return torch.cat((x[..., half:], x[..., :half]), dim=-1)


def tile_interleave(x: torch.Tensor, tile: int = 128) -> torch.Tensor:
    """[gate | up] -> 64 gate + 64 up per 128-wide tile (the reverted permute)."""
    n = x.shape[-1]
    if tile < 2 or n % tile != 0:
        return x
    lead = x.shape[:-1]
    half = tile // 2
    return x.reshape(*lead, 2, n // tile, half).permute(*range(x.ndim - 1), -2, -3, -1).reshape(*lead, n)


def match_transform(lhs: torch.Tensor, rhs: torch.Tensor) -> str:
    probes: list[tuple[str, Callable[[torch.Tensor], torch.Tensor]]] = [
        ("identity", lambda t: t),
        ("global half swap (up|gate)", half_swap),
        ("per-128-tile interleave", lambda t: tile_interleave(t, 128)),
        ("half swap + tile interleave", lambda t: tile_interleave(half_swap(t), 128)),
    ]
    for name, fn in probes:
        if torch.equal(fn(lhs), rhs):
            return name
    return "none of the known transforms"


def compare_weights(
    mega: dict[str, Any], cam: dict[str, Any], experts: list[int], rows: int
) -> None:
    print("\n[2] GMM1 weight layout")
    mega_w1 = as_list(mega.get("w1"))
    cam_w1 = as_list(cam.get("w1"))
    if not mega_w1 or not cam_w1:
        print("    skipped: dump has no w1")
        return
    print(
        f"    carriers: mega {mega_w1[0].dtype} {tuple(mega_w1[0].shape)}"
        f"  |  cam {cam_w1[0].dtype} {tuple(cam_w1[0].shape)}"
    )

    for expert in experts:
        if expert >= len(mega_w1) or expert >= len(cam_w1):
            print(f"    expert {expert}: not present in both dumps")
            continue
        mega_nd = nz_to_nd(mega_w1[expert])
        cam_nd = nz_to_nd(cam_w1[expert])
        if mega_nd is None or cam_nd is None:
            print(f"    expert {expert}: unknown NZ layout, skipped")
            continue
        if mega_nd.dtype == torch.int8:
            mega_int4 = unpack_int8_int4(mega_nd)
        else:
            mega_int4 = unpack_int32_int4(mega_nd)
        cam_int4 = cam_nd if cam_nd.dtype == torch.int8 else unpack_int32_int4(cam_nd)

        if mega_int4.shape[-1] != cam_int4.shape[-1]:
            print(
                f"    expert {expert}: N mismatch after unpack, mega {mega_int4.shape[-1]} vs "
                f"cam {cam_int4.shape[-1]}"
            )
            continue

        take = min(rows, mega_int4.shape[0])
        lhs = mega_int4[:take]
        rhs = cam_int4[:take]
        equal_elems = int((lhs == rhs).sum())
        total = lhs.numel()
        relation = match_transform(lhs, rhs)
        print(
            f"    expert {expert}: K={mega_int4.shape[0]} N={mega_int4.shape[-1]}, "
            f"first {take} rows identical={equal_elems}/{total} ({100.0 * equal_elems / total:.1f}%), "
            f"cam == {relation}(mega)"
        )

    print("    interpretation: 'identity' means both paths were given the same INT4 weights;")
    print("    any other match means CAM is fed a reordered GMM1 N axis.")


# --------------------------------------------------------------------------- #
# 3. output comparison
# --------------------------------------------------------------------------- #
def describe(tag: str, out: torch.Tensor, mask: torch.Tensor | None) -> dict[str, float]:
    flat = out.to(torch.float32)
    if mask is not None and mask.numel() == flat.shape[0]:
        flat = flat[mask]
    row_norm = flat.norm(dim=-1)
    stats = {
        "abs_mean": float(flat.abs().mean()),
        "std": float(flat.std()),
        "abs_max": float(flat.abs().max()),
        "zero_frac": float((flat == 0).float().mean()),
        "row_norm_mean": float(row_norm.mean()),
        "row_norm_max": float(row_norm.max()),
    }
    print(
        f"    {tag:10s} abs_mean={stats['abs_mean']:.6g} std={stats['std']:.6g} "
        f"abs_max={stats['abs_max']:.6g} zero={stats['zero_frac']:.3f} "
        f"row_norm mean={stats['row_norm_mean']:.6g} max={stats['row_norm_max']:.6g}"
    )
    return stats


def compare_outputs(
    mega: dict[str, Any], cam: dict[str, Any], layer: int, rank: int
) -> None:
    print(f"\n[3] fused-MoE output (layer {layer}, rank {rank})")
    mega_out = mega.get("out")
    cam_out = cam.get("out")
    if mega_out is None or cam_out is None:
        print("    skipped: dump has no output tensor")
        return
    if mega_out.shape != cam_out.shape:
        print(f"    SHAPE MISMATCH: mega {tuple(mega_out.shape)} vs cam {tuple(cam_out.shape)}")
        return

    mask = to_bool_mask(mega.get("x_active_mask"))
    describe("mega_out", mega_out, mask)
    describe("cam_out", cam_out, mask)

    lhs = mega_out.to(torch.float32)
    rhs = cam_out.to(torch.float32)
    if mask is not None and mask.numel() == lhs.shape[0]:
        lhs, rhs = lhs[mask], rhs[mask]
    if lhs.numel() == 0:
        print("    no active rows")
        return

    lhs_flat, rhs_flat = lhs.reshape(-1), rhs.reshape(-1)
    denom = float((rhs_flat * rhs_flat).sum())
    alpha = float((lhs_flat * rhs_flat).sum()) / denom if denom > 0 else float("nan")
    residual = float((lhs_flat - alpha * rhs_flat).norm()) / float(lhs_flat.norm().clamp_min(1e-12))
    cosine = torch.nn.functional.cosine_similarity(lhs.flatten(), rhs.flatten(), dim=0)
    rel = float((lhs_flat - rhs_flat).norm()) / float(lhs_flat.norm().clamp_min(1e-12))

    print(f"    fitted cam ~= {alpha:.6g} * mega")
    print(f"    relative error after removing that factor: {residual:.4f}")
    print(f"    relative error as-is: {rel:.4f}   cosine similarity: {float(cosine):.6f}")

    if residual < 0.05 and abs(alpha - 1.0) > 0.05:
        print("    => CAM output is a scaled copy: suspect a dequant / compensation-matrix factor.")
    elif residual < 0.05:
        print("    => CAM matches MegaMoe within quantization noise.")
    elif float(cosine) < 0.5:
        print("    => CAM output is uncorrelated with the reference: suspect weight order or routing.")
    else:
        print("    => CAM output correlates but is distorted: suspect per-channel scales or mask handling.")


# --------------------------------------------------------------------------- #
def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("ref_dir", help="dump directory from the DFC or MegaMoe reference run")
    parser.add_argument("cam_dir", help="dump directory from the CAM run")
    parser.add_argument("--layer", type=int, default=0)
    parser.add_argument("--rank", type=int, default=None, help="EP rank to inspect (default: first shared rank)")
    parser.add_argument("--experts", type=int, nargs="*", default=[0], help="experts to compare weights for")
    parser.add_argument("--rows", type=int, default=16, help="K rows of GMM1 to compare per expert")
    parser.add_argument("--no-weights", action="store_true", help="skip the weight layout comparison")
    args = parser.parse_args()

    ref_files = discover(args.ref_dir)
    cam_files = discover(args.cam_dir)
    if not ref_files or not cam_files:
        print(f"no dumps found (ref={len(ref_files)}, cam={len(cam_files)} files)", file=sys.stderr)
        print("did you set VLLM_ASCEND_MOE_DEBUG_DUMP before starting the server?", file=sys.stderr)
        return 1

    ranks = sorted({rank for _, _, rank in ref_files} & {rank for _, _, rank in cam_files})
    if not ranks:
        print("the two dump directories have no rank in common", file=sys.stderr)
        return 1
    rank = args.rank if args.rank is not None else ranks[0]
    if rank not in ranks:
        print(f"rank {rank} not in both dumps (shared: {ranks})", file=sys.stderr)
        return 1

    print(f"ref: {args.ref_dir} ({len(ref_files)} files)")
    print(f"cam: {args.cam_dir} ({len(cam_files)} files)")
    print(f"shared ranks: {ranks}; inspecting rank {rank}, layer {args.layer}\n")

    ref_in = first_tag(ref_files, REF_IN_TAGS, args.layer, rank)
    cam_in = cam_files.get((CAM_IN_TAG, args.layer, rank))
    if ref_in is None or cam_in is None:
        print(
            "missing input dumps for that layer/rank "
            f"(ref tags tried: {REF_IN_TAGS}; cam tag: {CAM_IN_TAG})",
            file=sys.stderr,
        )
        return 1

    ref_tag, ref_in_path = ref_in
    print(f"reference tag: {ref_tag}")
    ref = load(ref_in_path)
    cam = load(cam_in)

    compare_inputs(ref, cam)
    if not args.no_weights:
        compare_weights(ref, cam, args.experts, args.rows)

    ref_out = first_tag(ref_files, REF_OUT_TAGS, args.layer, rank)
    cam_out_path = cam_files.get((CAM_OUT_TAG, args.layer, rank))
    if ref_out and cam_out_path:
        compare_outputs(load(ref_out[1]), load(cam_out_path), args.layer, rank)
    else:
        print("\n[3] output dumps missing; only inputs were compared")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
