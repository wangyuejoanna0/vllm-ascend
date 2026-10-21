"""A3 CAM fused_deep_moe integration and graph-mode smoke test.

Requires idle A3 NPUs, the UMDK CAM `.run` package, and an `umdk_cam_op_lib`
wheel that registers `torch.ops.umdk_cam_op_lib.fused_deep_moe`.
Source the CANN environment and set the HCCL/GLOO interface before running:

    torchrun --nproc_per_node=8 test_cam_fused_deep_moe.py --tokens 4 32 256
    torchrun --nproc_per_node=8 test_cam_fused_deep_moe.py --tokens 256 --graph
    torchrun --nproc_per_node=8 test_cam_fused_deep_moe.py --tokens 32 --with-mask

Use a fresh process for each graph shape to isolate CANN/HCCL graph lifetime.
The script smoke-tests the Python adapter against the installed CAM operator
on INT32-packed W4A8 inputs. Shared experts are omitted so vLLM's external
shared-expert path remains the reference for that compute. It does not compare
against `dispatch_ffn_combine`.
"""

import argparse
import json
import os
from types import SimpleNamespace
from unittest.mock import patch

import torch
import torch.distributed as dist
import torch_npu  # noqa: F401  # registers torch.npu and compile backends

from vllm_ascend.ops.fused_moe.dataclass.fused_experts import build_fused_experts_input
from vllm_ascend.ops.fused_moe.moe_comm_method import FusedMC2CommImpl
from vllm_ascend.ops.fused_moe.moe_utils import load_cam_fused_deep_moe_op
from vllm_ascend.ops.fused_moe.token_dispatcher import TokenDispatcherWithMC2
from vllm_ascend.quantization.quant_type import QuantType


def _npu_graph_backend() -> str:
    """Resolve the torch.compile backend registered by the current torch_npu stack.

    ``npugraph`` is not a valid name. torchair (torch_npu 2.8) registers
    ``npugraphs``; newer stacks register ``npugraph_ex``.
    """
    try:
        import torchair  # noqa: F401
    except ImportError:
        pass
    try:
        import npugraph_ex  # noqa: F401
    except ImportError:
        pass
    available = torch._dynamo.list_backends()
    for name in ("npugraphs", "npugraph_ex"):
        if name in available:
            return name
    raise RuntimeError(
        "No NPU graph compile backend found. Tried npugraphs/npugraph_ex; "
        f"available={available}"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experts", type=int, default=64)
    parser.add_argument("--hidden", type=int, default=2048)
    parser.add_argument("--intermediate", type=int, default=1024)
    parser.add_argument("--topk", type=int, default=8)
    parser.add_argument("--tokens", type=int, nargs="+", default=[4, 32, 256])
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--graph", action="store_true")
    parser.add_argument("--with-mask", action="store_true")
    args = parser.parse_args()

    rank = int(os.environ["RANK"])
    world = int(os.environ["WORLD_SIZE"])
    torch.npu.set_device(int(os.environ["LOCAL_RANK"]))
    dist.init_process_group("hccl")
    group = dist.new_group(backend="hccl")
    group_name = group._get_backend(torch.device("npu")).get_hccl_comm_name(rank)
    torch.manual_seed(20261019 + rank)

    cam_op = load_cam_fused_deep_moe_op()
    local_experts = args.experts // world
    w1 = [
        torch.randint(-8, 8, (args.hidden, (2 * args.intermediate) // 8), device="npu", dtype=torch.int32)
        for _ in range(local_experts)
    ]
    w2 = [
        torch.randint(-8, 8, (args.intermediate, args.hidden // 8), device="npu", dtype=torch.int32)
        for _ in range(local_experts)
    ]
    s1 = [torch.rand(2 * args.intermediate, device="npu", dtype=torch.float32) * 0.003 for _ in range(local_experts)]
    s2 = [torch.rand(args.hidden, device="npu", dtype=torch.float32) * 0.003 for _ in range(local_experts)]
    b1 = [torch.rand(2 * args.intermediate, device="npu", dtype=torch.float32) for _ in range(local_experts)]
    b2 = [torch.rand(args.hidden, device="npu", dtype=torch.float32) for _ in range(local_experts)]
    weights = SimpleNamespace(
        w1=w1,
        w2=w2,
        w1_scale=s1,
        w2_scale=s2,
        w1_scale_bias=b1,
        w2_scale_bias=b2,
    )

    comm = object.__new__(FusedMC2CommImpl)
    comm.cam_fused_deep_moe = cam_op
    comm._cam_static_inputs_validated = False
    comm.moe_config = SimpleNamespace(
        num_experts=args.experts,
        intermediate_size_per_partition=args.intermediate,
    )
    comm.token_dispatcher = object.__new__(TokenDispatcherWithMC2)
    comm.token_dispatcher.ep_world_size = world
    comm.token_dispatcher.ep_rank_id = rank
    comm.token_dispatcher.global_bs = 0
    comm.token_dispatcher.moe_all_to_all_group_name = group_name
    print(json.dumps({"rank": rank, "stage": "initialized", "config": vars(args)}), flush=True)

    def run_once(num_tokens):
        x = torch.randn(num_tokens, args.hidden, device="npu", dtype=torch.bfloat16)
        ids = torch.randint(0, args.experts, (num_tokens, args.topk), device="npu", dtype=torch.int32)
        probs = torch.rand(num_tokens, args.topk, device="npu", dtype=torch.float32)
        mask = None
        if args.with_mask:
            # Uniform-mode padding (global_bs=0): trailing rows are pad, not
            # routed tokens. CAM AIV still indexes GM from expert_ids on those
            # rows, so leave placeholder ids / zeros instead of random experts.
            num_pad = num_tokens // 2
            mask = torch.ones(num_tokens, device="npu", dtype=torch.bool)
            if num_pad:
                mask[num_tokens - num_pad :] = False
                x[num_tokens - num_pad :] = 0
                ids[num_tokens - num_pad :] = torch.arange(args.topk, device="npu", dtype=torch.int32)
                probs[num_tokens - num_pad :] = 0
            mask = mask.contiguous()
        inp = build_fused_experts_input(
            hidden_states=x,
            topk_ids=ids,
            topk_weights=probs,
            quant_type=QuantType.W4A8,
            dynamic_eplb=False,
            mc2_mask=mask,
        )
        return comm._apply_cam_fused_deep_moe(inp, weights)

    for num_tokens in args.tokens:
        runner = run_once
        out, expert_tokens = runner(num_tokens)
        torch.npu.synchronize()
        dist.barrier()
        if args.graph:
            runner = torch.compile(run_once, backend=_npu_graph_backend())
            out, expert_tokens = runner(num_tokens)
            torch.npu.synchronize()
        print(
            json.dumps(
                {
                    "rank": rank,
                    "tokens": num_tokens,
                    "out_shape": list(out.shape),
                    "expert_tokens": int(expert_tokens.numel()),
                    "graph": args.graph,
                    "mask": args.with_mask,
                }
            ),
            flush=True,
        )
        for _ in range(args.repeats):
            runner(num_tokens)
            torch.npu.synchronize()

    dist.destroy_process_group()


if __name__ == "__main__":
    with patch.dict(os.environ, {"HCCL_BUFFSIZE": os.environ.get("HCCL_BUFFSIZE", "1024")}):
        main()
