# CAM Fused DeepMoE

Optional fused-MC2 backend for routed-expert W4A8 on Ascend A3. vLLM Ascend
loads the prebuilt UMDK CAM wheel and vendor `.run` package; it does not copy
the kernel sources into `csrc`.

## When it is used

Illegal combinations (non-A3, `enable_fused_mc2!=1`, LoRA, EPLB, MiniMax M3)
raise at config init. A missing `umdk_cam_op_lib` wheel disables the switch and
fused MC2 continues with MegaMoe or `dispatch_ffn_combine`. A missing operator
or argument mismatch at load keeps the CAM weight payload and falls back to
`dispatch_ffn_combine`.

CAM is selected when all of the following hold:

- Ascend A3 hardware
- `enable_fused_mc2=1`
- `enable_cam_fused_deep_moe=true`
- W4A8 INT (not MXFP), silu/swiglu routed experts, float16/bfloat16 activations
- Expert parallel world size greater than 1
- No LoRA, no EPLB, and not MiniMax M3
- `umdk_cam_op_lib` is installed and registers `torch.ops.umdk_cam_op_lib.fused_deep_moe`

Shared experts stay on the existing vLLM path. Shared CAM inputs are passed as
`None` and the unused `share_output` tensor is ignored.

## Install the external backend

1. Install the matching CANN / `torch_npu` stack for A3.
2. Install the CAM operator `.run` package from UMDK (registers `aclnnFusedDeepMoe`
   and the A3 W4A8 kernel).
3. Install the `umdk_cam_op_lib` wheel from the same UMDK revision so Python
   registers `torch.ops.umdk_cam_op_lib.fused_deep_moe`.
4. Set `HCCL_BUFFSIZE` large enough for
   `local_experts * global_bs * hidden * 2 * 2` bytes. When `global_bs` is `0`,
   host-side checks use `ep_world_size * 256` as `global_bs`.

Record the CAM `.run` / wheel / CANN / `torch_npu` versions used for a
deployment. Graph capture requires the wheel's Meta registration
(`TORCH_LIBRARY_IMPL(..., Meta)`) and a fixed `global_bs` / padding shape.

## Server flags

```bash
vllm serve <w4a8-moe-model> \
  --additional-config '{"enable_fused_mc2": 1, "enable_cam_fused_deep_moe": true}'
```

## Operator limits

Host-side checks raise a concrete error when any of these fail:

| Constraint | Limit |
| --- | --- |
| Tokens per rank | `<= 256` |
| Hidden size | `512–7168` |
| GMM1 hidden (`2 * intermediate_size`) | `1024–6144` |
| topK | `<= 12` |
| Experts | `<= 512` and divisible by EP |
| Local experts | `<= AIV/2` |
| `global_bs` | `0` or divisible by EP |
| Activations | float16/bfloat16 |
| Weights | INT32-packed INT4, per-expert TensorList, NZ layout |
| Scales / bias | float32 scales and float32 bias; not MegaMoe INT8-packed W4A8 |

## A3 multi-card validation

On A3, smoke-test the Python adapter against the installed UMDK operator:

```bash
torchrun --nproc_per_node=8 \
  tests/e2e/nightly/single_node/ops/test_cam_fused_deep_moe.py \
  --tokens 4 32 256 --graph
```

Cover active mask, more than one EP/topK pair, and routed-only (no shared)
inputs. Use a fresh process for each graph shape. The script does not compare
against `dispatch_ffn_combine` and does not replace accuracy tests against the
CAM sample.
