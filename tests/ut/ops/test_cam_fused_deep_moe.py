from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch
from vllm.model_executor.layers.fused_moe.activation import MoEActivation

from tests.ut.base import TestBase
from vllm_ascend.ascend_forward_context import _is_cam_w4a8_int_quant, use_cam_fused_deep_moe, use_cann_megamoe
from vllm_ascend.device.hardware import AscendDeviceType
from vllm_ascend.device.hardware_profile import HardwareCapability, get_hardware_profile
from vllm_ascend.ops.fused_moe.dataclass.fused_experts import MoEWeights, build_fused_experts_input
from vllm_ascend.ops.fused_moe.moe_comm_method import FusedMC2CommImpl
from vllm_ascend.ops.fused_moe.moe_utils import (
    CAM_FUSED_DEEP_MOE_PAYLOAD_VALIDATED_ATTR,
    CAM_FUSED_DEEP_MOE_REQUIRED_ARGS,
    load_cam_fused_deep_moe_op,
    validate_cam_fused_deep_moe_inputs,
    validate_cam_fused_deep_moe_payload,
)
from vllm_ascend.ops.fused_moe.token_dispatcher import TokenDispatcherWithMC2
from vllm_ascend.quantization.quant_type import QuantType


class TestCamFusedDeepMoeAbi(TestBase):
    def test_load_accepts_matching_schema(self):
        class _Arg:
            def __init__(self, name):
                self.name = name

        class _Op:
            _schema = SimpleNamespace(arguments=[_Arg(name) for name in CAM_FUSED_DEEP_MOE_REQUIRED_ARGS])

        op = _Op()
        fake_ops = SimpleNamespace(umdk_cam_op_lib=SimpleNamespace(fused_deep_moe=op))
        with (
            patch("vllm_ascend.ops.fused_moe.moe_utils.import_module"),
            patch("vllm_ascend.ops.fused_moe.moe_utils.torch.ops", fake_ops),
        ):
            self.assertIs(load_cam_fused_deep_moe_op(), op)

    def test_load_rejects_missing_required_argument(self):
        class _Arg:
            def __init__(self, name):
                self.name = name

        class _Op:
            _schema = SimpleNamespace(
                arguments=[_Arg(name) for name in CAM_FUSED_DEEP_MOE_REQUIRED_ARGS if name != "global_bs"]
            )

        fake_ops = SimpleNamespace(umdk_cam_op_lib=SimpleNamespace(fused_deep_moe=_Op()))
        with (
            patch("vllm_ascend.ops.fused_moe.moe_utils.import_module"),
            patch("vllm_ascend.ops.fused_moe.moe_utils.torch.ops", fake_ops),
            self.assertRaisesRegex(RuntimeError, "missing required routed-only arguments"),
        ):
            load_cam_fused_deep_moe_op()

    def test_load_accepts_torch_op_overload_packet_schema(self):
        class _Arg:
            def __init__(self, name):
                self.name = name

        class _Default:
            _schema = SimpleNamespace(arguments=[_Arg(name) for name in CAM_FUSED_DEEP_MOE_REQUIRED_ARGS])

        class _Packet:
            _schema = None
            default = _Default()

            @staticmethod
            def __call__(*args, **kwargs):
                return None

        op = _Packet()
        fake_ops = SimpleNamespace(umdk_cam_op_lib=SimpleNamespace(fused_deep_moe=op))
        with (
            patch("vllm_ascend.ops.fused_moe.moe_utils.import_module"),
            patch("vllm_ascend.ops.fused_moe.moe_utils.torch.ops", fake_ops),
        ):
            self.assertIs(load_cam_fused_deep_moe_op(), op)


class TestCamFusedDeepMoeConstraints(TestBase):
    def test_rejects_batch_hidden_topk_and_expert_shape(self):
        common = dict(
            batch_size=16,
            hidden_size=2048,
            gmm1_hidden=2048,
            top_k=8,
            num_experts=64,
            ep_world_size=8,
            global_bs=0,
        )
        with self.assertRaisesRegex(ValueError, "at most 256 tokens"):
            validate_cam_fused_deep_moe_inputs(**{**common, "batch_size": 257})
        with self.assertRaisesRegex(ValueError, "hidden size"):
            validate_cam_fused_deep_moe_inputs(**{**common, "hidden_size": 256})
        with self.assertRaisesRegex(ValueError, "topK"):
            validate_cam_fused_deep_moe_inputs(**{**common, "top_k": 13})
        with self.assertRaisesRegex(ValueError, "divisible by EP"):
            validate_cam_fused_deep_moe_inputs(**{**common, "num_experts": 60})
        with self.assertRaisesRegex(ValueError, "global_bs"):
            validate_cam_fused_deep_moe_inputs(**{**common, "global_bs": 12})

    def test_rejects_int8_or_int64_payload(self):
        int8_w = [torch.zeros(8, 4, dtype=torch.int8)]
        int32_w = [torch.zeros(8, 4, dtype=torch.int32)]
        float_s = [torch.ones(8, dtype=torch.float32)]
        int64_s = [torch.ones(8, dtype=torch.int64)]
        bias = [torch.ones(8, dtype=torch.float32)]
        with self.assertRaisesRegex(ValueError, "must be torch.int32"):
            validate_cam_fused_deep_moe_payload(
                w1=int8_w,
                w2=int32_w,
                w1_scale=float_s,
                w2_scale=float_s,
                w1_scale_bias=bias,
                w2_scale_bias=bias,
            )
        with self.assertRaisesRegex(ValueError, "INT64 packed scales"):
            validate_cam_fused_deep_moe_payload(
                w1=int32_w,
                w2=int32_w,
                w1_scale=int64_s,
                w2_scale=float_s,
                w1_scale_bias=bias,
                w2_scale_bias=bias,
            )


class TestCamFusedDeepMoeSelection(TestBase):
    def test_w4a8_int_quant_detection(self):
        w4a8 = SimpleNamespace(quant_config=SimpleNamespace(quant_description={"w": "W4A8_DYNAMIC"}))
        modelslim = SimpleNamespace(
            quant_config=SimpleNamespace(quant_description={"model_quant_type": "W4A8_DYNAMIC", "group_size": 0})
        )
        mxfp = SimpleNamespace(quant_config=SimpleNamespace(quant_description={"w": "W4A8MXFP"}))
        empty = SimpleNamespace(quant_config=SimpleNamespace(quant_description={}))
        self.assertTrue(_is_cam_w4a8_int_quant(w4a8))
        self.assertTrue(_is_cam_w4a8_int_quant(modelslim))
        self.assertFalse(_is_cam_w4a8_int_quant(empty))
        self.assertFalse(_is_cam_w4a8_int_quant(mxfp))
        self.assertFalse(_is_cam_w4a8_int_quant(SimpleNamespace(quant_config=None)))

    def test_use_cam_requires_explicit_switch_and_capabilities(self):
        vllm_config = SimpleNamespace(
            parallel_config=SimpleNamespace(enable_expert_parallel=True),
            quant_config=SimpleNamespace(quant_description={"experts": "W4A8_DYNAMIC"}),
        )
        ascend_config = SimpleNamespace(enable_cam_fused_deep_moe=True)
        with (
            patch("vllm_ascend.ascend_forward_context.get_ascend_config", return_value=ascend_config),
            patch("vllm_ascend.ascend_forward_context.is_moe_model", return_value=True),
            patch("vllm_ascend.ascend_forward_context.get_ep_group", return_value=SimpleNamespace(world_size=8)),
        ):
            self.assertTrue(use_cam_fused_deep_moe(vllm_config))

        ascend_config.enable_cam_fused_deep_moe = False
        with patch("vllm_ascend.ascend_forward_context.get_ascend_config", return_value=ascend_config):
            self.assertFalse(use_cam_fused_deep_moe(vllm_config))

        ascend_config.enable_cam_fused_deep_moe = True
        with (
            patch("vllm_ascend.ascend_forward_context.get_ascend_config", return_value=ascend_config),
            patch("vllm_ascend.ascend_forward_context.is_moe_model", return_value=True),
            patch("vllm_ascend.ascend_forward_context.get_ep_group", return_value=SimpleNamespace(world_size=1)),
        ):
            self.assertFalse(use_cam_fused_deep_moe(vllm_config))

        w8a8 = SimpleNamespace(
            parallel_config=SimpleNamespace(enable_expert_parallel=True),
            quant_config=SimpleNamespace(quant_description={"w": "W8A8"}),
        )
        with (
            patch("vllm_ascend.ascend_forward_context.get_ascend_config", return_value=ascend_config),
            patch("vllm_ascend.ascend_forward_context.is_moe_model", return_value=True),
            patch("vllm_ascend.ascend_forward_context.get_ep_group", return_value=SimpleNamespace(world_size=8)),
        ):
            self.assertFalse(use_cam_fused_deep_moe(w8a8))

        self.assertTrue(get_hardware_profile(AscendDeviceType.A3).supports(HardwareCapability.CAM_FUSED_DEEP_MOE))
        self.assertFalse(get_hardware_profile(AscendDeviceType.A2).supports(HardwareCapability.CAM_FUSED_DEEP_MOE))

    def test_use_cann_megamoe_stays_off_when_cam_is_selected(self):
        vllm_config = SimpleNamespace(
            parallel_config=SimpleNamespace(enable_expert_parallel=True),
            quant_config=SimpleNamespace(quant_description={"experts": "W4A8_DYNAMIC"}),
            lora_config=None,
        )
        ascend_config = SimpleNamespace(enable_cam_fused_deep_moe=True, enable_fused_mc2=1)
        with (
            patch("vllm_ascend.ascend_forward_context.get_ascend_config", return_value=ascend_config),
            patch("vllm_ascend.ascend_forward_context.is_moe_model", return_value=True),
            patch("vllm_ascend.ascend_forward_context.get_ep_group", return_value=SimpleNamespace(world_size=8)),
            patch("vllm_ascend.ascend_forward_context.is_mega_moe_supported", return_value=True),
            patch(
                "vllm_ascend.ascend_forward_context.get_current_hardware_profile",
                return_value=get_hardware_profile(AscendDeviceType.A3),
            ),
        ):
            self.assertTrue(use_cam_fused_deep_moe(vllm_config))
            self.assertFalse(use_cann_megamoe(vllm_config))

            ascend_config.enable_cam_fused_deep_moe = False
            self.assertTrue(use_cann_megamoe(vllm_config))


class TestCamFusedDeepMoeAdapter(TestBase):
    def _make_comm(self, cam_op):
        comm = object.__new__(FusedMC2CommImpl)
        comm.cam_fused_deep_moe = cam_op
        comm._cam_static_inputs_validated = False
        comm.moe_config = SimpleNamespace(num_experts=16, intermediate_size_per_partition=1024)
        comm.token_dispatcher = object.__new__(TokenDispatcherWithMC2)
        comm.token_dispatcher.ep_world_size = 8
        comm.token_dispatcher.ep_rank_id = 1
        comm.token_dispatcher.global_bs = 0
        comm.token_dispatcher.moe_all_to_all_group_name = "ep_group"
        return comm

    def test_maps_routed_only_arguments_and_ignores_share_output(self):
        captured = {}

        def cam_op(**kwargs):
            captured.update(kwargs)
            return torch.ones(2, 2048), torch.zeros(2, 2048), torch.ones(2, dtype=torch.int64)

        comm = self._make_comm(cam_op)
        hidden = torch.zeros(2, 2048, dtype=torch.bfloat16)
        topk_ids = torch.zeros(2, 4, dtype=torch.int32)
        topk_weights = torch.ones(2, 4)
        weights = MoEWeights(
            w1=[torch.zeros(2048, 256, dtype=torch.int32)],
            w2=[torch.zeros(1024, 256, dtype=torch.int32)],
            w1_scale=[torch.ones(2048, dtype=torch.float32)],
            w2_scale=[torch.ones(2048, dtype=torch.float32)],
            w1_scale_bias=[torch.ones(2048, dtype=torch.float32)],
            w2_scale_bias=[torch.ones(2048, dtype=torch.float32)],
        )
        fused_input = build_fused_experts_input(
            hidden_states=hidden,
            topk_ids=topk_ids,
            topk_weights=topk_weights,
            quant_type=QuantType.W4A8,
            dynamic_eplb=False,
            mc2_mask=torch.tensor([True, False]),
        )
        with patch("vllm_ascend.ops.fused_moe.moe_comm_method.moe_utils._cam_aiv_num", return_value=48):
            out, expert_tokens = comm._apply_cam_fused_deep_moe(fused_input, weights)

        self.assertTrue(torch.equal(out, torch.ones(2, 2048)))
        self.assertEqual(captured["x"].dtype, torch.bfloat16)
        self.assertEqual(expert_tokens.dtype, torch.int64)
        self.assertIs(captured["x"], hidden)
        self.assertEqual(captured["expert_ids"].dtype, torch.int32)
        self.assertEqual(captured["expert_scales"].dtype, torch.float32)
        self.assertIs(captured["gmm1_weight"], weights.w1)
        self.assertIsNone(captured["share_gmm1_weight"])
        self.assertIsNone(captured["share_gmm2_weight"])
        self.assertIsNone(captured["share_gmm1_bias"])
        self.assertIsNone(captured["share_gmm2_bias"])
        self.assertIsNone(captured["expert_smooth_scales"])
        self.assertEqual(captured["group_ep"], "ep_group")
        self.assertEqual(captured["ep_rank_size"], 8)
        self.assertEqual(captured["ep_rank_id"], 1)
        self.assertEqual(captured["moe_expert_num"], 16)
        self.assertEqual(captured["quant_mode"], 0)
        self.assertEqual(captured["global_bs"], 0)
        self.assertEqual(captured["x_active_mask"].dtype, torch.bool)

    def test_skips_payload_validation_after_first_layer_check(self):
        comm = self._make_comm(
            lambda **kwargs: (kwargs["x"], kwargs["x"], torch.ones(2, dtype=torch.int64))
        )
        layer = SimpleNamespace()
        setattr(layer, CAM_FUSED_DEEP_MOE_PAYLOAD_VALIDATED_ATTR, True)
        weights = MoEWeights(
            w1=[torch.zeros(2048, 256, dtype=torch.int8)],
            w2=[torch.zeros(1024, 256, dtype=torch.int32)],
            w1_scale=[torch.ones(2048, dtype=torch.int64)],
            w2_scale=[torch.ones(2048, dtype=torch.float32)],
            w1_scale_bias=[torch.ones(2048, dtype=torch.float32)],
            w2_scale_bias=[torch.ones(2048, dtype=torch.float32)],
        )
        fused_input = build_fused_experts_input(
            hidden_states=torch.zeros(2, 2048, dtype=torch.bfloat16),
            topk_ids=torch.zeros(2, 4, dtype=torch.int32),
            topk_weights=torch.ones(2, 4),
            quant_type=QuantType.W4A8,
            dynamic_eplb=False,
            layer=layer,
        )
        with (
            patch("vllm_ascend.ops.fused_moe.moe_comm_method.moe_utils._cam_aiv_num", return_value=48),
            patch(
                "vllm_ascend.ops.fused_moe.moe_comm_method.moe_utils.validate_cam_fused_deep_moe_payload",
                side_effect=AssertionError("payload should not be re-checked"),
            ),
        ):
            comm._apply_cam_fused_deep_moe(fused_input, weights)

    def test_skips_static_input_validation_after_first_check(self):
        comm = self._make_comm(
            lambda **kwargs: (kwargs["x"], kwargs["x"], torch.ones(2, dtype=torch.int64))
        )
        comm._cam_static_inputs_validated = True
        weights = MoEWeights(
            w1=[torch.zeros(2048, 256, dtype=torch.int32)],
            w2=[torch.zeros(1024, 256, dtype=torch.int32)],
            w1_scale=[torch.ones(2048, dtype=torch.float32)],
            w2_scale=[torch.ones(2048, dtype=torch.float32)],
            w1_scale_bias=[torch.ones(2048, dtype=torch.float32)],
            w2_scale_bias=[torch.ones(2048, dtype=torch.float32)],
        )
        fused_input = build_fused_experts_input(
            hidden_states=torch.zeros(2, 2048, dtype=torch.bfloat16),
            topk_ids=torch.zeros(2, 4, dtype=torch.int32),
            topk_weights=torch.ones(2, 4),
            quant_type=QuantType.W4A8,
            dynamic_eplb=False,
        )
        with patch(
            "vllm_ascend.ops.fused_moe.moe_comm_method.moe_utils.validate_cam_fused_deep_moe_static_inputs",
            side_effect=AssertionError("static inputs should not be re-checked"),
        ):
            comm._apply_cam_fused_deep_moe(fused_input, weights)

    def test_rejects_situ_activation(self):
        comm = self._make_comm(lambda **kwargs: (kwargs["x"], kwargs["x"], torch.ones(2, dtype=torch.int64)))
        fused_input = build_fused_experts_input(
            hidden_states=torch.zeros(2, 2048, dtype=torch.bfloat16),
            topk_ids=torch.zeros(2, 4, dtype=torch.int32),
            topk_weights=torch.ones(2, 4),
            quant_type=QuantType.W4A8,
            dynamic_eplb=False,
            activation=MoEActivation.SITU,
        )
        weights = MoEWeights(
            w1=[torch.zeros(2048, 256, dtype=torch.int32)],
            w2=[torch.zeros(1024, 256, dtype=torch.int32)],
            w1_scale=[torch.ones(2048, dtype=torch.float32)],
            w2_scale=[torch.ones(2048, dtype=torch.float32)],
            w1_scale_bias=[torch.ones(2048, dtype=torch.float32)],
            w2_scale_bias=[torch.ones(2048, dtype=torch.float32)],
        )
        with self.assertRaisesRegex(ValueError, "silu/swiglu"):
            comm._apply_cam_fused_deep_moe(fused_input, weights)

    def test_fused_experts_uses_cam_for_w4a8_when_loaded(self):
        comm = object.__new__(FusedMC2CommImpl)
        comm.enable_fused_mc2 = 1
        comm.cam_fused_deep_moe = object()
        comm.token_dispatcher = object.__new__(TokenDispatcherWithMC2)
        routed = torch.ones(2, 8)
        expert_tokens = torch.ones(2, dtype=torch.int64)
        comm._apply_cam_fused_deep_moe = MagicMock(return_value=(routed, expert_tokens))
        fused_input = MagicMock()
        fused_input.quant.quant_type = QuantType.W4A8
        fused_input.layer = MagicMock()
        weights = MoEWeights(
            w1=[torch.zeros(8, 4, dtype=torch.int32)],
            w2=[torch.zeros(8, 4, dtype=torch.int32)],
            w1_scale=[torch.ones(8, dtype=torch.float32)],
            w2_scale=[torch.ones(8, dtype=torch.float32)],
            w1_scale_bias=[torch.ones(8, dtype=torch.float32)],
            w2_scale_bias=[torch.ones(8, dtype=torch.float32)],
        )
        with patch(
            "vllm_ascend.ops.fused_moe.moe_comm_method._EXTRA_CTX",
            SimpleNamespace(use_cam_fused_deep_moe=True, use_mega_moe=False),
        ):
            result = comm.fused_experts(
                fused_input, quant_method=SimpleNamespace(get_cam_fused_deep_moe_weights=lambda _: weights)
            )
        comm._apply_cam_fused_deep_moe.assert_called_once()
        self.assertIs(result.routed_out, routed)
        self.assertIs(result.expert_tokens, expert_tokens)

    def test_fused_experts_skips_cam_when_op_missing(self):
        comm = object.__new__(FusedMC2CommImpl)
        comm.enable_fused_mc2 = 1
        comm.cam_fused_deep_moe = None
        comm._apply_cam_fused_deep_moe = MagicMock()
        comm._apply_cann_mega_moe = MagicMock()
        comm.expert_token_nums = torch.zeros(2, dtype=torch.int32)
        comm.swiglu_limit = 0.0
        comm.token_dispatcher = object.__new__(TokenDispatcherWithMC2)
        comm.token_dispatcher.ep_world_size = 8
        comm.token_dispatcher.moe_all_to_all_group_name = "ep_group"
        fused_input = MagicMock()
        fused_input.quant.quant_type = QuantType.W4A8
        fused_input.hidden_states = torch.zeros(2, 8)
        fused_input.topk_ids = torch.zeros(2, 2, dtype=torch.int32)
        fused_input.topk_weights = torch.ones(2, 2)
        fused_input.routing.mc2_mask = None
        fused_input.layer = MagicMock()
        weights = MoEWeights(
            w1=[torch.zeros(8, 4, dtype=torch.int32)],
            w2=[torch.zeros(8, 4, dtype=torch.int32)],
            w1_scale=[torch.ones(8, dtype=torch.int64)],
            w2_scale=[torch.ones(8, dtype=torch.int64)],
            w1_scale_bias=[torch.ones(8, dtype=torch.float32)],
            w2_scale_bias=[torch.ones(8, dtype=torch.float32)],
        )
        with (
            patch(
                "vllm_ascend.ops.fused_moe.moe_comm_method._EXTRA_CTX",
                SimpleNamespace(use_cam_fused_deep_moe=True, use_mega_moe=True),
            ),
            patch("vllm_ascend.ops.fused_moe.moe_comm_method.get_ascend_config") as mock_config,
            patch("torch.ops._C_ascend.dispatch_ffn_combine", create=True) as mock_dispatch,
        ):
            mock_config.return_value.mega_moe_max_tokens = 65536
            comm.fused_experts(fused_input, quant_method=SimpleNamespace(get_fused_mc2_weights=lambda _: weights))
        comm._apply_cam_fused_deep_moe.assert_not_called()
        comm._apply_cann_mega_moe.assert_not_called()
        mock_dispatch.assert_called_once()
