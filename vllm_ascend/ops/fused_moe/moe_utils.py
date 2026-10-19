# Copyright (c) 2024; NVIDIA CORPORATION. All rights reserved.
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# Copyright 2023 The vLLM team.
# This file is a part of the vllm-ascend project.
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
#
import inspect
import os
from collections.abc import Callable, Sequence
from functools import lru_cache
from importlib import import_module
from typing import Any

import regex as re
import torch
import torch.distributed
import torch.distributed as dist
import torch.nn as nn
import torch_npu
from torch.nn.functional import pad
from vllm.config import get_current_vllm_config_or_none
from vllm.logger import logger

from vllm_ascend.quantization.quant_type import QuantType
from vllm_ascend.utils import enable_custom_op

COMM_STREAM = None

_CANN_ACL_INT8 = 258
_CANN_ACL_INT4 = 285
_CANN_MEGA_MOE_QUANT_MODE_NONE = 0
_CANN_MEGA_MOE_QUANT_MODE_INT8 = 2
_CANN_MEGA_MOE_QUANT_MODE_MXFP = 4
# Constants adapt to A5 cann_ops_transformer mega moe operator.
# which is documented in file mc2/mega_moe/README.md on
# repository https://gitcode.com/cann/ops-transformer.git.
# (23 = FLOAT8_E5M2, 24 = FLOAT8_E4M3FN, 296 = FLOAT4_E2M1).
_CANN_ACL_FLOAT8_E5M2 = 23
_CANN_ACL_FLOAT8_E4M3FN = 24
_CANN_ACL_FLOAT4_E2M1 = 296

CAM_FUSED_DEEP_MOE_MAX_BATCH_SIZE = 256
CAM_FUSED_DEEP_MOE_MIN_HIDDEN = 512
CAM_FUSED_DEEP_MOE_MAX_HIDDEN = 7168
CAM_FUSED_DEEP_MOE_MAX_TOPK = 12
CAM_FUSED_DEEP_MOE_MAX_EXPERTS = 512
CAM_FUSED_DEEP_MOE_MIN_GMM1_HIDDEN = 1024
CAM_FUSED_DEEP_MOE_MAX_GMM1_HIDDEN = 6144
CAM_FUSED_DEEP_MOE_QUANT_MODE = 0
CAM_FUSED_DEEP_MOE_DEFAULT_AIV_NUM = 48
CAM_FUSED_DEEP_MOE_TOKEN_DTYPE_BYTES = 2
CAM_FUSED_DEEP_MOE_DOUBLE_BUFFER = 2
CAM_FUSED_DEEP_MOE_MB_SIZE = 1024 * 1024
CAM_FUSED_DEEP_MOE_DEFAULT_HCCL_BUFFSIZE_MB = 200
CAM_FUSED_DEEP_MOE_PAYLOAD_VALIDATED_ATTR = "_cam_fused_deep_moe_payload_validated"
CAM_FUSED_DEEP_MOE_REQUIRED_ARGS = (
    "x",
    "expert_ids",
    "gmm1_weight",
    "gmm1_weight_scale",
    "gmm2_weight",
    "gmm2_weight_scale",
    "expert_scales",
    "share_gmm1_weight",
    "share_gmm1_weight_scale",
    "share_gmm2_weight",
    "share_gmm2_weight_scale",
    "expert_smooth_scales",
    "share_smooth_scales",
    "x_active_mask",
    "gmm1_bias",
    "gmm2_bias",
    "share_gmm1_bias",
    "share_gmm2_bias",
    "group_ep",
    "ep_rank_size",
    "ep_rank_id",
    "moe_expert_num",
    "quant_mode",
    "global_bs",
)

# dict: QuantType -> (dispatch_quant_mode, dispatch_quant_out_dtype, weight_type)
_QUANT_SETTING_MAP: dict[QuantType, tuple[int, int | None, int | None]] = {
    QuantType.W8A8: (_CANN_MEGA_MOE_QUANT_MODE_INT8, _CANN_ACL_INT8, _CANN_ACL_INT8),
    QuantType.W4A8: (_CANN_MEGA_MOE_QUANT_MODE_INT8, _CANN_ACL_INT8, _CANN_ACL_INT4),
    QuantType.NONE: (_CANN_MEGA_MOE_QUANT_MODE_NONE, None, None),
    QuantType.W8A8MXFP: (_CANN_MEGA_MOE_QUANT_MODE_MXFP, _CANN_ACL_FLOAT8_E4M3FN, _CANN_ACL_FLOAT8_E4M3FN),
    QuantType.W4A8MXFP: (_CANN_MEGA_MOE_QUANT_MODE_MXFP, _CANN_ACL_FLOAT8_E4M3FN, _CANN_ACL_FLOAT4_E2M1),
    QuantType.W4A4MXFP: (_CANN_MEGA_MOE_QUANT_MODE_MXFP, _CANN_ACL_FLOAT4_E2M1, _CANN_ACL_FLOAT4_E2M1),
}


def async_all_to_all(input_, output_split_sizes, input_split_sizes, group, event=None):
    if output_split_sizes is None:
        # Equal split (all2all)
        a2a_out = torch.empty_like(input_)
    else:
        # Unequal split (all2all-v)
        a2a_out = input_.new_empty(
            size=[sum(output_split_sizes)] + list(input_.size()[1:]),
            dtype=input_.dtype,
            device=torch.npu.current_device(),
        )

    if event:
        # multi stream wait event
        global COMM_STREAM
        if COMM_STREAM is None:
            COMM_STREAM = torch_npu.npu.Stream(device=torch.npu.current_device())
        with torch_npu.npu.stream(COMM_STREAM):
            event.wait()
            handle = dist.all_to_all_single(
                a2a_out,
                input_.contiguous(),
                output_split_sizes=output_split_sizes,
                input_split_sizes=input_split_sizes,
                group=group,
                async_op=True,
            )
    else:
        handle = dist.all_to_all_single(
            a2a_out,
            input_.contiguous(),
            output_split_sizes=output_split_sizes,
            input_split_sizes=input_split_sizes,
            group=group,
            async_op=True,
        )
    return input_, a2a_out, handle


def _gather_along_first_dim(input_, group, output_split_sizes=None):
    """Gather tensors and concatenate along the first dimension.

    Args:
        input_tensor (torch.Tensor):
            A tensor to be gathered.
        output_split_sizes (List[int], optional):
            A list specifying the sizes of the output splits along the first dimension.
            If None, equal splitting is assumed. Default: None.

    Returns:
        torch.Tensor: Gathered tensor.
    """
    world_size = torch.distributed.get_world_size(group)
    # Bypass the function if we are using only 1 GPU.
    if world_size == 1:
        return input_

    dim_size = list(input_.size())
    if output_split_sizes is None:
        dim_size[0] = dim_size[0] * world_size

        output = torch.empty(dim_size, dtype=input_.dtype, device=torch.npu.current_device())
        torch.distributed.all_gather_into_tensor(output, input_.contiguous(), group=group)
    else:
        dim_size[0] = sum(output_split_sizes)
        output = torch.empty(dim_size, dtype=input_.dtype, device=torch.npu.current_device())
        output_tensor_list = list(torch.split(output, output_split_sizes, dim=0))
        torch.distributed.all_gather(output_tensor_list, input_, group=group)

    return output


def gather_from_sequence_parallel_region(
    input_,
    group,
    output_split_sizes=None,
):
    """Wrapper for autograd function: forward: AG, backward: RS <first dim>"""
    return _gather_along_first_dim(input_, group, output_split_sizes)


def load_cann_mega_moe_ops():
    ops_module = import_module("cann_ops_transformer.ops")
    get_symm_buffer_for_mega_moe = ops_module.get_symm_buffer_for_mega_moe
    mega_moe = ops_module.mega_moe
    # A5's communication extension is lazy-loaded by SymmBuffer. Compile it
    # during initialization, before TorchDynamo / ACL graph capture starts.
    try:
        comm_context = import_module("cann_ops_transformer.ops.mc2.common.comm_context")
    except ModuleNotFoundError as exc:
        if exc.name not in {
            "cann_ops_transformer.ops.mc2",
            "cann_ops_transformer.ops.mc2.common",
            "cann_ops_transformer.ops.mc2.common.comm_context",
        }:
            raise
        # Older CANN packages keep the communication extension at this path.
        comm_context = import_module("cann_ops_transformer.ops.comm_context")
        comm_context.CommContextManager._ensure_loaded()
    else:
        comm_context.comm_context_op_builder.load()
    return get_symm_buffer_for_mega_moe, mega_moe


@lru_cache(maxsize=1)
def _cam_aiv_num() -> int:
    try:
        props = torch.npu.get_device_properties(torch.npu.current_device())
        vector_core_num = getattr(props, "vector_core_num", None)
        # MagicMock from CPU UTs int()-converts to 1; AIV/2 must be at least 1.
        aiv_num = int(vector_core_num) if vector_core_num else 0
        if aiv_num >= 2:
            return aiv_num
    except (RuntimeError, AttributeError, TypeError, ValueError):
        pass
    return CAM_FUSED_DEEP_MOE_DEFAULT_AIV_NUM


def _cam_hccl_buffer_size_bytes() -> int:
    raw = os.environ.get("HCCL_BUFFSIZE")
    if raw is None or raw == "":
        return CAM_FUSED_DEEP_MOE_DEFAULT_HCCL_BUFFSIZE_MB * CAM_FUSED_DEEP_MOE_MB_SIZE
    try:
        return int(raw) * CAM_FUSED_DEEP_MOE_MB_SIZE
    except ValueError as exc:
        raise ValueError(
            f"HCCL_BUFFSIZE must be an integer number of megabytes, got {raw!r}."
        ) from exc


def _cam_op_argument_names(cam_op: Any) -> set[str]:
    schema = getattr(cam_op, "_schema", None)
    # torch.ops packets keep the schema on the default overload, not the packet.
    if schema is None:
        default_overload = getattr(cam_op, "default", None)
        schema = getattr(default_overload, "_schema", None)
    if schema is not None:
        arguments = getattr(schema, "arguments", None)
        if arguments:
            return {getattr(argument, "name", str(argument)) for argument in arguments}
    try:
        names = set(inspect.signature(cam_op).parameters)
    except (TypeError, ValueError):
        return set()
    # OpOverloadPacket.signature() is (*args, **kwargs); that is not the ABI.
    if names <= {"args", "kwargs"}:
        return set()
    return names


def load_cam_fused_deep_moe_op() -> Callable[..., Any]:
    """Import the UMDK CAM wheel and return ``fused_deep_moe``.

    Importing the wheel only registers ``torch.ops.umdk_cam_op_lib.fused_deep_moe``.
    The operator itself is stateless: group/rank/world/global_bs are passed per
    call, and this loader must not compile kernels or allocate persistent
    communication resources.
    """
    import_module("umdk_cam_op_lib")
    try:
        cam_op = torch.ops.umdk_cam_op_lib.fused_deep_moe
    except AttributeError as exc:
        raise RuntimeError(
            "umdk_cam_op_lib is installed but torch.ops.umdk_cam_op_lib.fused_deep_moe "
            "is missing. Install a wheel that registers the CAM fused_deep_moe operator."
        ) from exc
    argument_names = _cam_op_argument_names(cam_op)
    missing = [name for name in CAM_FUSED_DEEP_MOE_REQUIRED_ARGS if argument_names and name not in argument_names]
    if missing:
        raise RuntimeError(
            "umdk_cam_op_lib.fused_deep_moe is missing required routed-only arguments "
            f"{missing}."
        )
    return cam_op


def validate_cam_fused_deep_moe_payload(
    *,
    w1: Sequence[torch.Tensor],
    w2: Sequence[torch.Tensor],
    w1_scale: Sequence[torch.Tensor],
    w2_scale: Sequence[torch.Tensor],
    w1_scale_bias: Sequence[torch.Tensor],
    w2_scale_bias: Sequence[torch.Tensor],
) -> None:
    """Reject MegaMoe INT8-packed W4A8 weights and other incompatible payloads.

    Dtypes are fixed after weight loading, so call this once when the CAM
    payload is installed. ``_apply_cam_fused_deep_moe`` only re-runs it for
    layers that skipped install (for example unit tests that pass lists
    directly).
    """
    if not w1 or not w2 or not w1_scale or not w2_scale or not w1_scale_bias or not w2_scale_bias:
        raise ValueError(
            "CAM fused_deep_moe requires per-expert INT32-packed W4A8 weights plus "
            "float scales and float32 bias lists. The MegaMoe INT8-packed payload "
            "cannot be reused."
        )
    if len({len(w1), len(w2), len(w1_scale), len(w2_scale), len(w1_scale_bias), len(w2_scale_bias)}) != 1:
        raise ValueError(
            "CAM fused_deep_moe weight/scale/bias TensorLists must have the same "
            f"length, got weights=({len(w1)}, {len(w2)}), scales=({len(w1_scale)}, "
            f"{len(w2_scale)}), bias=({len(w1_scale_bias)}, {len(w2_scale_bias)})."
        )
    for name, tensors, expected_dtype in (
        ("gmm1_weight", w1, torch.int32),
        ("gmm2_weight", w2, torch.int32),
        ("gmm1_bias", w1_scale_bias, torch.float32),
        ("gmm2_bias", w2_scale_bias, torch.float32),
    ):
        for tensor in tensors:
            if tensor.dtype != expected_dtype:
                raise ValueError(
                    f"CAM fused_deep_moe {name} must be {expected_dtype}, got {tensor.dtype}."
                )
    for name, tensors in (("gmm1_weight_scale", w1_scale), ("gmm2_weight_scale", w2_scale)):
        for tensor in tensors:
            if tensor.dtype not in (torch.float32, torch.bfloat16, torch.float16):
                raise ValueError(
                    f"CAM fused_deep_moe {name} must be float32/bfloat16/float16, got {tensor.dtype}. "
                    "Do not pass the MegaMoe or dispatch_ffn_combine INT64 packed scales."
                )


def validate_cam_fused_deep_moe_static_inputs(
    *,
    hidden_size: int,
    num_experts: int,
    ep_world_size: int,
    global_bs: int,
) -> None:
    """Check expert / EP / AIV / HCCL limits that do not change every forward.

    When ``global_bs == 0`` the HCCL demand uses the CAM max batch so a small
    first step cannot skip a buffer that a later 256-token step would need.
    """
    if num_experts > CAM_FUSED_DEEP_MOE_MAX_EXPERTS:
        raise ValueError(
            f"CAM fused_deep_moe supports at most {CAM_FUSED_DEEP_MOE_MAX_EXPERTS} experts, "
            f"got {num_experts}."
        )
    if ep_world_size <= 1:
        raise ValueError("CAM fused_deep_moe requires expert parallel world size > 1.")
    if num_experts % ep_world_size != 0:
        raise ValueError(
            f"CAM fused_deep_moe requires num_experts ({num_experts}) to be divisible by "
            f"EP world size ({ep_world_size})."
        )
    local_experts = num_experts // ep_world_size
    max_local_experts = _cam_aiv_num() // 2
    if local_experts > max_local_experts:
        raise ValueError(
            "CAM fused_deep_moe requires local experts <= AIV/2 "
            f"({max_local_experts}), got {local_experts}."
        )
    if global_bs < 0:
        raise ValueError(f"CAM fused_deep_moe global_bs must be >= 0, got {global_bs}.")
    if global_bs > 0 and global_bs % ep_world_size != 0:
        raise ValueError(
            f"CAM fused_deep_moe global_bs ({global_bs}) must be divisible by EP world size "
            f"({ep_world_size})."
        )
    effective_global_bs = (
        global_bs if global_bs > 0 else ep_world_size * CAM_FUSED_DEEP_MOE_MAX_BATCH_SIZE
    )
    buffer_demand = (
        local_experts
        * effective_global_bs
        * hidden_size
        * CAM_FUSED_DEEP_MOE_TOKEN_DTYPE_BYTES
        * CAM_FUSED_DEEP_MOE_DOUBLE_BUFFER
    )
    hccl_buffer_bytes = _cam_hccl_buffer_size_bytes()
    if buffer_demand > hccl_buffer_bytes:
        needed_mb = (buffer_demand + CAM_FUSED_DEEP_MOE_MB_SIZE - 1) // CAM_FUSED_DEEP_MOE_MB_SIZE
        have_mb = hccl_buffer_bytes // CAM_FUSED_DEEP_MOE_MB_SIZE
        raise ValueError(
            "CAM fused_deep_moe HCCL buffer is too small: "
            f"need HCCL_BUFFSIZE>={needed_mb} MB, current HCCL_BUFFSIZE={have_mb} MB "
            f"(local_experts={local_experts}, global_bs={effective_global_bs}, hidden={hidden_size})."
        )


def validate_cam_fused_deep_moe_shape_inputs(
    *,
    batch_size: int,
    hidden_size: int,
    gmm1_hidden: int,
    top_k: int,
) -> None:
    """Check per-step CAM shapes that can change with padding or graph size."""
    if batch_size > CAM_FUSED_DEEP_MOE_MAX_BATCH_SIZE:
        raise ValueError(
            f"CAM fused_deep_moe supports at most {CAM_FUSED_DEEP_MOE_MAX_BATCH_SIZE} tokens "
            f"per rank, got {batch_size}. Reduce max_num_batched_tokens / graph capture size "
            "or disable enable_cam_fused_deep_moe to fall back to dispatch_ffn_combine."
        )
    if hidden_size < CAM_FUSED_DEEP_MOE_MIN_HIDDEN or hidden_size > CAM_FUSED_DEEP_MOE_MAX_HIDDEN:
        raise ValueError(
            "CAM fused_deep_moe hidden size must be in "
            f"[{CAM_FUSED_DEEP_MOE_MIN_HIDDEN}, {CAM_FUSED_DEEP_MOE_MAX_HIDDEN}], got {hidden_size}."
        )
    if gmm1_hidden < CAM_FUSED_DEEP_MOE_MIN_GMM1_HIDDEN or gmm1_hidden > CAM_FUSED_DEEP_MOE_MAX_GMM1_HIDDEN:
        raise ValueError(
            "CAM fused_deep_moe GMM1 hidden (2 * intermediate_size) must be in "
            f"[{CAM_FUSED_DEEP_MOE_MIN_GMM1_HIDDEN}, {CAM_FUSED_DEEP_MOE_MAX_GMM1_HIDDEN}], "
            f"got {gmm1_hidden}."
        )
    if top_k > CAM_FUSED_DEEP_MOE_MAX_TOPK:
        raise ValueError(
            f"CAM fused_deep_moe supports topK <= {CAM_FUSED_DEEP_MOE_MAX_TOPK}, got {top_k}."
        )


def validate_cam_fused_deep_moe_inputs(
    *,
    batch_size: int,
    hidden_size: int,
    gmm1_hidden: int,
    top_k: int,
    num_experts: int,
    ep_world_size: int,
    global_bs: int,
) -> None:
    """Raise an actionable error when a CAM host-side constraint is violated."""
    validate_cam_fused_deep_moe_static_inputs(
        hidden_size=hidden_size,
        num_experts=num_experts,
        ep_world_size=ep_world_size,
        global_bs=global_bs,
    )
    validate_cam_fused_deep_moe_shape_inputs(
        batch_size=batch_size,
        hidden_size=hidden_size,
        gmm1_hidden=gmm1_hidden,
        top_k=top_k,
    )


def select_mega_moe_activation_kwargs(
    mega_moe_op: Callable[..., Any],
    *,
    activation: object,
    activation_clamp: float | None,
    swiglu_alpha: float,
    swiglu_beta: float,
    situ_beta: float | None = None,
    situ_linear_beta: float | None = None,
) -> dict[str, object]:
    """Build activation kwargs compatible with the installed MegaMoe wrapper.

    MiniMax-M3 uses SwiGLU-OAI (``alpha=1.702``, ``beta=1.0``). Older MegaMoe
    builds only expose ``activation_clamp`` for standard SwiGLU, while newer
    wrappers expose either ``activation``/``activation_params`` or direct
    alpha/beta keyword arguments. Standard SwiGLU keeps the legacy call.
    """
    kwargs: dict[str, object] = {"activation_clamp": activation_clamp}
    activation_name = getattr(activation, "value", activation)
    is_situ = activation_name == "situ"
    if not is_situ and activation_name not in ("swigluoai", "swigluoai_uninterleave"):
        return kwargs

    activation_parameter_pairs = (
        {"activation", "activation_params"},
        {"glu_alpha", "glu_bias"},
        {"swiglu_alpha", "swiglu_beta"},
    )
    try:
        parameters = inspect.signature(mega_moe_op).parameters
        parameter_names = {
            name
            for name, parameter in parameters.items()
            if parameter.kind in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)
        }
        accepts_kwargs = any(parameter.kind == inspect.Parameter.VAR_KEYWORD for parameter in parameters.values())
    except (TypeError, ValueError):
        parameter_names = set()
        accepts_kwargs = True
    if accepts_kwargs and not any(pair.issubset(parameter_names) for pair in activation_parameter_pairs):
        # Compiled operators and generic **kwargs wrappers may describe their
        # arguments only in metadata. Prefer an explicit Python activation API;
        # documentation must not introduce keywords a closed wrapper rejects.
        signature_metadata = " ".join(
            str(value)
            for value in (getattr(mega_moe_op, "__doc__", None), getattr(mega_moe_op, "_schema", None))
            if value is not None
        )
        known_names = {
            "activation",
            "activation_params",
            "glu_alpha",
            "glu_bias",
            "swiglu_alpha",
            "swiglu_beta",
        }
        parameter_names.update(name for name in known_names if re.search(rf"\b{re.escape(name)}\b", signature_metadata))

    if is_situ:
        if not {"activation", "activation_params"}.issubset(parameter_names):
            raise RuntimeError(
                "The installed CANN MegaMoe wrapper does not expose SiTU activation parameters. "
                "Install a compatible cann_ops_transformer build or disable fused MC2."
            )
        # CANN calls SituAndMul 'situglu'. Omitting linear_beta preserves the
        # unbounded up projection; passing zero would change the activation.
        params = {"beta": 1.0 if situ_beta is None else situ_beta}
        if situ_linear_beta is not None:
            params["linear_beta"] = situ_linear_beta
        return {"activation": "situglu", "activation_params": params}
    if {"activation", "activation_params"}.issubset(parameter_names):
        kwargs.update(
            activation="swigluoai",
            activation_params={"alpha": swiglu_alpha, "beta": swiglu_beta},
        )
        return kwargs
    if {"glu_alpha", "glu_bias"}.issubset(parameter_names):
        kwargs.update(glu_alpha=swiglu_alpha, glu_bias=swiglu_beta)
        return kwargs
    if {"swiglu_alpha", "swiglu_beta"}.issubset(parameter_names):
        kwargs.update(swiglu_alpha=swiglu_alpha, swiglu_beta=swiglu_beta)
        return kwargs

    raise RuntimeError(
        "The installed CANN MegaMoe wrapper does not expose SwiGLU-OAI "
        "activation parameters required by MiniMax-M3. Install a compatible "
        "cann_ops_transformer build or disable fused MC2."
    )


def _get_cann_mega_moe_quant_settings(quant_type: QuantType) -> tuple[int, int | None, int | None]:
    """
    Returns (dispatch_quant_mode, dispatch_quant_out_dtype, weight_type).
    The current custom op package still requires explicit INT4 for W4A8
    packed weights; otherwise it derives W4A8's packed N as an INT8 N and
    rejects weight2.

    dispatch_quant_out_dtype: the doc types this as torch.dtype (torch.int8 /
    torch.float8_e4m3fn). We pass the ACL enum ints (258 / 24) because W8A8
    was validated end‑to‑end this way in PD; switching W4A8 to torch.int8 did
    NOT fix the W4A8 accuracy issue and slowed graph capture (see bug_a3.md),
    so keep the working values until the W4A8 accuracy root cause is found on
    the operator side.

    Ascend 950 MegaMoe is MXFP‑only (dispatch_quant_mode == 4). The
    dispatch_quant_out_dtype must bind to the activation dtype and the
    weight_type to the weight dtype, per the A5 scenario table:
      W8A8MXFP -> A8W8‑FP (act FP8, weight FP8)
      W4A8MXFP -> A8W4‑FP (act FP8, weight FP4)
      W4A4MXFP -> A4W4‑FP (act FP4, weight FP4)
    """
    if quant_type not in _QUANT_SETTING_MAP:
        raise RuntimeError(
            "MegaMoe integration supports W8A8/W4A8 INT on A2/A3 and W8A8MXFP/W4A8MXFP/W4A4MXFP on A5 "
            "MegaMoe platforms for now. "
            f"Unsupported quant type: {quant_type}."
        )

    mega_moe_quant_settings = _QUANT_SETTING_MAP[quant_type]
    dispatch_quant_mode, dispatch_quant_out_dtype, weight_type = mega_moe_quant_settings
    logger.debug(
        "Get cann mega_moe quant_settings success: dispatch_quant_mode=%s dispatch_quant_out_dtype=%s weight_type=%s",
        dispatch_quant_mode,
        dispatch_quant_out_dtype,
        weight_type,
    )
    return mega_moe_quant_settings


def get_moe_num_logical_experts(
    layer: torch.nn.Module,
    num_experts: int,
    global_redundant_expert_num: int = 0,
    num_shared_experts: int = 0,
) -> int:
    moe_config = getattr(layer, "moe_config", None)
    num_logical_experts = getattr(moe_config, "num_logical_experts", None)
    if num_logical_experts is not None:
        return int(num_logical_experts)

    return int(num_experts - global_redundant_expert_num - num_shared_experts)


def _custom_gmm_swiglu_enabled(fusion, dynamic_eplb, activation=None):
    activation_name = getattr(activation, "value", activation)
    return fusion and dynamic_eplb and activation_name not in ("situ", "swigluoai_uninterleave") and enable_custom_op()


def cumsum_group_list(
    group_list: torch.Tensor, src_list_type: int, dst_list_type: int, active_num: int = 0, expert_num: int = 0
) -> torch.Tensor:
    if src_list_type not in [0, 1, 2]:
        raise ValueError(f"group_list_type should be in [0, 1, 2], but received {src_list_type}")

    if src_list_type == dst_list_type:
        return group_list
    if src_list_type == 1 and dst_list_type == 0:
        return group_list.cumsum(dim=0)
    if src_list_type == 0 and dst_list_type == 1:
        group_diff = torch.diff(group_list)
        new_group = torch.cat([group_list[0].unsqueeze(0), group_diff], dim=0)
        return new_group
    if src_list_type == 2 and dst_list_type == 0:
        experts = pad(group_list[:, 0], (1, 0))
        tokens = pad(group_list[:, 1].cumsum(dim=0), (1, 0))
        cumsum_group_list = torch.full(
            size=(expert_num,), fill_value=active_num, dtype=group_list.dtype, device=group_list.device
        )

        for i, (start, end) in enumerate(zip(experts[:-1], experts[1:])):
            if end > start:
                cumsum_group_list[start:end] = tokens[i]

        return cumsum_group_list
    raise NotImplementedError(
        f"Conversion from src_list_type={src_list_type} to dst_list_type={dst_list_type} is not implemented yet. "
        "This feature is under development."
    )


def _prepare_dequant_swiglu_weight_scale(
    w1_scale: list[torch.Tensor] | torch.Tensor,
    is_swigluoai_uninterleave: bool,
) -> torch.Tensor:
    """Prepare w1_scale for the swigluoai_uninterleave dequant fused op."""
    if not is_swigluoai_uninterleave:
        weight_scale = w1_scale[0]
    elif isinstance(w1_scale, list):
        if len(w1_scale) == 1:
            weight_scale = w1_scale[0]
        else:
            weight_scale = torch.stack([scale.reshape(-1) for scale in w1_scale], dim=0)
    else:
        weight_scale = w1_scale
    if weight_scale.dtype != torch.float32:
        weight_scale = weight_scale.to(torch.float32)
    if is_swigluoai_uninterleave and weight_scale.dim() == 1:
        weight_scale = weight_scale.reshape(1, -1)
    return weight_scale


def maybe_normalize_mxfp_scale_layout(scale: torch.Tensor | None) -> torch.Tensor | None:
    """Normalize an MXFP per-token scale to the layout expected by A5 kernels.

    MXFP quant types are only supported on Ascend A5, where a 2D scale is
    reshaped into the (rows, cols // 2, 2) layout used by the fused gmm ops.
    """
    if scale is None or scale.ndim != 2:
        return scale
    if scale.shape[-1] % 2 != 0:
        raise ValueError(f"Invalid MXFP scale shape: {tuple(scale.shape)}")
    return scale.reshape(scale.shape[0], scale.shape[1] // 2, 2)


@lru_cache(maxsize=1)
def enable_fusion_gmmswigluquant():
    from vllm_ascend.ascend_config import get_ascend_config

    ascend_config = get_ascend_config()
    return ascend_config.ascend_fusion_config.fusion_ops_gmmswigluquant


# Token-dim padding via `torch.cat` with a persistent zero block, replacing
# `nn.functional.pad` in the MoE prepare paths. On NPU each `F.pad` lowers to
# two device kernels (a full-output MemSet plus a PadV3 copy) while the padded
# data is only a few KB in decode, so the pair is pure launch overhead
# (~44us per MoE layer in a Kimi K3 TP16 decode profile). The cat variant is
# one ConcatD kernel per tensor and is safe under cudagraph replay: the cat
# output is a fresh tensor per call (each captured graph owns its output
# buffer), and the zero block is never written after allocation, so the
# zero-tail property is structural — no cross-call bookkeeping.

# (width..., dtype, device) -> zero block of [tp_size, width...]. Both MoE
# prepare paths pad to a multiple of tp_size by fewer than tp_size rows (MC2
# pads the local token count up to the DP-uniform padded_num_tokens; All2All
# pads up to exactly tp_size), so one tp_size-row block per tensor width
# serves every pad and is allocated exactly once — never replaced or freed,
# which is what makes it safe for captured graphs to reference.
#
# Outside a worker context there is no current vllm config (e.g. unit tests
# calling prepare() directly): tp_size then reads as 0 and every pad takes
# the F.pad fallback below.
#
# A pad wider than tp_size can only happen outside that invariant (e.g. an
# eager call with zero tokens); it falls back to plain `nn.functional.pad`
# instead of growing the entry, keeping the cache static.
#
# Memory footprint: the key drops the token dim, so the cache holds one entry
# per distinct (trailing shape, dtype, device) — a set fixed by the model
# config, independent of batch size and cudagraph capture sizes (two entries
# for Kimi K3 TP16: hidden 3584, experts 896), each tp_size * width *
# dtype.itemsize bytes, so the whole cache peaks at O(0.1) MB.
_PAD_ZERO_BLOCKS: dict[tuple, torch.Tensor] = {}


def _pad_tokens_with_cat(x: torch.Tensor, padded_len: int) -> torch.Tensor:
    """Token-dim padding of `x` ([n, ...] -> [padded_len, ...]) by concatenating
    a slice of a cached zero block: value-equivalent to
    `F.pad(x, (0, 0, 0, padded_len - n))` at one kernel instead of two."""
    pad_rows = padded_len - x.shape[0]
    assert pad_rows >= 0, f"padded_len ({padded_len}) is smaller than the input's token dim ({x.shape[0]})"
    vllm_config = get_current_vllm_config_or_none()
    tp_size = vllm_config.parallel_config.tensor_parallel_size if vllm_config is not None else 0
    if pad_rows > tp_size:
        return nn.functional.pad(x, (0, 0, 0, pad_rows))
    key = (*x.shape[1:], x.dtype, str(x.device))
    zero_block = _PAD_ZERO_BLOCKS.get(key)
    if zero_block is None:
        zero_block = torch.zeros((tp_size, *x.shape[1:]), dtype=x.dtype, device=x.device)
        _PAD_ZERO_BLOCKS[key] = zero_block
    return torch.cat([x, zero_block[:pad_rows]], dim=0)
