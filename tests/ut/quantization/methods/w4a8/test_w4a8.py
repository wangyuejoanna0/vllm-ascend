import inspect
from types import SimpleNamespace
from unittest.mock import Mock, patch

import regex as re
import torch

from tests.ut.base import TestBase
from tests.ut.quantization.conftest_quantization import identity
from vllm_ascend.ops.fused_moe.moe_utils import CAM_FUSED_DEEP_MOE_PAYLOAD_VALIDATED_ATTR
from vllm_ascend.quantization.methods.w4a8.w4a8 import (
    _CAM_INT32_NZ_COL_FRACTAL,
    _CAM_INT32_NZ_ROW_FRACTAL,
    AscendW4A8DynamicFusedMoEMethod,
)
from vllm_ascend.utils import ASCEND_QUANTIZATION_METHOD, COMPRESSED_TENSORS_METHOD

# Official INT32 NZ fractal is (16, 8). After ModelSlim transpose + INT32 pack:
# GMM1 is [H, I/4], GMM2 is [I, H/8]. Need H%16==0, I%32==0, H%64==0.
# I>=64 and H>=128 so N/8>1 and the NZ shuffle is not a no-op.
_CAM_TEST_INTERMEDIATE = 64
_CAM_TEST_HIDDEN = 128


def _cam_int32_nz_to_nd(weight: torch.Tensor) -> torch.Tensor:
    """Inverse of official ``convert_nd_to_nz`` for a 2-D or 3-D INT32 carrier."""
    row_fractal, col_fractal = _CAM_INT32_NZ_ROW_FRACTAL, _CAM_INT32_NZ_COL_FRACTAL
    *leading_dims, rows, cols = weight.shape
    viewed = weight.reshape(
        *leading_dims,
        cols // col_fractal,
        rows // row_fractal,
        row_fractal,
        col_fractal,
    )
    lead = len(leading_dims)
    nd = viewed.permute(*range(lead), lead + 1, lead + 2, lead, lead + 3)
    return nd.reshape(*leading_dims, rows, cols).contiguous()


class TestAscendW4A8DynamicFusedMoEMethod(TestBase):
    experts = 8
    input_size = 16
    output_size = 56

    @patch("vllm_ascend.quantization.methods.w4a8.w4a8.get_ascend_config")
    @patch("vllm_ascend.quantization.methods.w4a8.w4a8.get_current_vllm_config")
    @patch("vllm_ascend.quantization.methods.w4a8.w4a8.get_mc2_group")
    @patch("torch.distributed.get_rank", return_value=0)
    def setUp(self, mock_get_rank, mock_get_mc2_group, get_current_vllm_config, mock_get_ascend_config):
        # Mock ascend config
        mock_ascend_config = Mock()
        mock_ascend_config.eplb_config.dynamic_eplb = False
        mock_get_ascend_config.return_value = mock_ascend_config

        mock_vllm_config = Mock()
        mock_vllm_config.quant_config = Mock(quant_description={"group_size": 0})
        mock_vllm_config.quant_config.get_name.return_value = ASCEND_QUANTIZATION_METHOD
        mock_vllm_config.parallel_config = Mock(enable_expert_parallel=True, enable_eplb=False)
        mock_vllm_config.use_v2_model_runner = False
        mock_vllm_config.scheduler_config = Mock(
            max_num_batched_tokens=2048, max_model_len=2048, enable_chunked_prefill=False
        )
        get_current_vllm_config.return_value = mock_vllm_config
        self.quant_method = AscendW4A8DynamicFusedMoEMethod()

    @patch("vllm_ascend.quantization.methods.w4a8.w4a8.get_tensor_model_parallel_world_size")
    @patch("vllm_ascend.quantization.methods.w4a8.w4a8.get_current_vllm_config")
    def test_init_raise_error(self, get_current_vllm_config, mock_tp_size):
        # test init rejects per-group quantization
        mock_vllm_config = Mock()
        mock_vllm_config.quant_config = Mock(quant_description={"group_size": 256})
        mock_vllm_config.parallel_config = Mock(enable_expert_parallel=True)
        mock_vllm_config.use_v2_model_runner = False
        get_current_vllm_config.return_value = mock_vllm_config
        with self.assertRaisesRegex(ValueError, "no longer supported"):
            AscendW4A8DynamicFusedMoEMethod()

        # test init rejects tp>16 when use modelslim weights
        mock_vllm_config.use_v2_model_runner = True
        mock_vllm_config.quant_config = Mock(quant_description={})
        mock_vllm_config.quant_config.get_name.return_value = ASCEND_QUANTIZATION_METHOD
        mock_vllm_config.parallel_config = Mock(enable_expert_parallel=False)
        mock_tp_size.return_value = 32
        with self.assertRaisesRegex(ValueError, "tp>16"):
            AscendW4A8DynamicFusedMoEMethod()

    def test_get_weight(self):
        param_dict = self.quant_method.get_weight(self.experts, self.input_size, self.output_size, torch.bfloat16)
        self.assertEqual(param_dict["w13_weight"].dtype, torch.int8)
        self.assertEqual(param_dict["w13_weight"].shape, (self.experts, self.input_size, self.output_size))
        self.assertEqual(param_dict["w2_weight"].dtype, torch.int8)
        self.assertEqual(param_dict["w2_weight"].shape, (self.experts, self.output_size // 2, self.input_size))

    def test_get_weight_compressed_tensors(self):
        self.quant_method.quant_method = COMPRESSED_TENSORS_METHOD
        result = self.quant_method.get_weight(self.experts, self.input_size, self.output_size, torch.bfloat16)
        self.assertEqual(result["w13_weight"].dtype, torch.int8)
        self.assertEqual(result["w13_weight"].shape, (self.experts, 2 * self.input_size, self.output_size))
        self.assertEqual(result["w2_weight"].shape, (self.experts, self.output_size, self.input_size))

    def test_get_dynamic_quant_param(self):
        param_dict = self.quant_method.get_dynamic_quant_param(
            self.experts, self.input_size, self.output_size, torch.bfloat16
        )
        self.assertEqual(param_dict["w13_weight_scale"].dtype, torch.float32)
        self.assertEqual(param_dict["w13_weight_scale"].shape, (self.experts, 2 * self.input_size, 1))
        self.assertEqual(param_dict["w13_weight_offset"].dtype, torch.float32)
        self.assertEqual(param_dict["w13_weight_offset"].shape, (self.experts, 2 * self.input_size, 1))
        self.assertEqual(param_dict["w2_weight_scale"].dtype, torch.float32)
        self.assertEqual(param_dict["w2_weight_scale"].shape, (self.experts, self.output_size, 1))
        self.assertEqual(param_dict["w2_weight_offset"].dtype, torch.float32)
        self.assertEqual(param_dict["w2_weight_offset"].shape, (self.experts, self.output_size, 1))
        self.assertEqual(param_dict["w13_scale_bias"].dtype, torch.float32)
        self.assertEqual(param_dict["w13_scale_bias"].shape, (self.experts, 2 * self.input_size, 1))
        self.assertEqual(param_dict["w2_scale_bias"].dtype, torch.float32)
        self.assertEqual(
            param_dict["w2_scale_bias"].shape, (self.experts, self.output_size, 16 // self.quant_method.tp_size)
        )
        pergroup_param = [
            "w13_weight_scale_second",
            "w13_weight_offset_second",
            "w2_weight_scale_second",
            "w2_weight_offset_second",
        ]
        is_contains = any(key in param_dict for key in pergroup_param)
        self.assertFalse(is_contains)

    def test_get_dynamic_quant_param_compressed_tensors(self):
        self.quant_method.quant_method = COMPRESSED_TENSORS_METHOD
        result = self.quant_method.get_dynamic_quant_param(
            self.experts, self.input_size, self.output_size, torch.bfloat16
        )
        self.assertIn("w13_weight_scale", result)
        self.assertIn("w2_weight_scale", result)
        self.assertEqual(result["w13_weight_scale"].dtype, torch.bfloat16)
        self.assertEqual(result["w13_weight_scale"].shape, (self.experts, 2 * self.input_size, 1))
        self.assertEqual(result["w2_weight_scale"].dtype, torch.bfloat16)
        self.assertEqual(result["w2_weight_scale"].shape, (self.experts, self.output_size, 1))

    def build_layer(self):
        layer = torch.nn.Module()
        layer.w13_weight = torch.nn.Parameter(
            torch.zeros((self.experts, self.input_size, self.output_size), dtype=torch.int8), requires_grad=False
        )
        layer.w2_weight = torch.nn.Parameter(
            torch.zeros((self.experts, self.output_size // 2, self.input_size), dtype=torch.int8),
            requires_grad=False,
        )
        w13_scale_bias = torch.zeros((self.experts, 2 * self.input_size, 1), dtype=torch.float32)
        layer.w13_scale_bias = torch.nn.Parameter(w13_scale_bias, requires_grad=False)
        w2_scale_bias = torch.zeros(
            (self.experts, self.output_size, 16 // self.quant_method.tp_size), dtype=torch.float32
        )
        layer.w2_scale_bias = torch.nn.Parameter(w2_scale_bias, requires_grad=False)
        layer.w13_weight_scale = torch.nn.Parameter(
            torch.ones((self.experts, 2 * self.input_size, 1), dtype=torch.float32), requires_grad=False
        )
        layer.w2_weight_scale = torch.nn.Parameter(
            torch.ones((self.experts, self.output_size, 1), dtype=torch.float32), requires_grad=False
        )
        return layer

    def build_cam_layer(self, intermediate: int = _CAM_TEST_INTERMEDIATE, hidden: int = _CAM_TEST_HIDDEN):
        """ModelSlim W4A8 shapes that satisfy CAM INT32 NZ alignment."""
        layer = torch.nn.Module()
        layer.w13_weight = torch.nn.Parameter(
            torch.zeros((self.experts, intermediate, hidden), dtype=torch.int8), requires_grad=False
        )
        layer.w2_weight = torch.nn.Parameter(
            torch.zeros((self.experts, hidden // 2, intermediate), dtype=torch.int8),
            requires_grad=False,
        )
        layer.w13_scale_bias = torch.nn.Parameter(
            torch.zeros((self.experts, 2 * intermediate, 1), dtype=torch.float32), requires_grad=False
        )
        layer.w2_scale_bias = torch.nn.Parameter(
            torch.zeros((self.experts, hidden, 16 // self.quant_method.tp_size), dtype=torch.float32),
            requires_grad=False,
        )
        layer.w13_weight_scale = torch.nn.Parameter(
            torch.ones((self.experts, 2 * intermediate, 1), dtype=torch.float32), requires_grad=False
        )
        layer.w2_weight_scale = torch.nn.Parameter(
            torch.ones((self.experts, hidden, 1), dtype=torch.float32), requires_grad=False
        )
        return layer

    def build_layer_compressed_tensors(self):
        layer = torch.nn.Module()
        layer.w13_weight = torch.nn.Parameter(
            torch.zeros((self.experts, 2 * self.input_size, self.output_size), dtype=torch.int8), requires_grad=False
        )
        layer.w2_weight = torch.nn.Parameter(
            torch.zeros((self.experts, self.output_size, self.input_size), dtype=torch.int8), requires_grad=False
        )
        layer.w13_weight_scale = torch.nn.Parameter(
            torch.ones((self.experts, 2 * self.input_size, 1), dtype=torch.bfloat16), requires_grad=False
        )
        layer.w2_weight_scale = torch.nn.Parameter(
            torch.ones((self.experts, self.output_size, 1), dtype=torch.bfloat16), requires_grad=False
        )
        return layer

    def test_get_eplb_weight_views_matches_routed_compute_inputs(self):
        layer = self.build_layer()

        weight_views = self.quant_method.get_eplb_weight_views(layer)

        expected = [
            layer.w13_weight,
            layer.w2_weight,
            layer.w13_weight_scale,
            layer.w2_weight_scale,
            layer.w13_scale_bias,
            layer.w2_scale_bias,
        ]
        self.assertEqual(len(weight_views), len(expected))
        for actual, expected_view in zip(weight_views, expected):
            self.assertIs(actual, expected_view)

    def test_get_eplb_weight_views_rejects_incomplete_scale_bias(self):
        layer = self.build_layer()
        del layer.w2_scale_bias

        with self.assertRaisesRegex(RuntimeError, "w13_scale_bias and w2_scale_bias to be present or absent together"):
            self.quant_method.get_eplb_weight_views(layer)

    def test_get_eplb_weight_views_rejects_incomplete_scale_bias_lists(self):
        layer = torch.nn.Module()
        layer.w13_weight_list = [torch.empty(2, 3) for _ in range(self.experts)]
        layer.w2_weight_list = [torch.empty(3, 2) for _ in range(self.experts)]
        layer.w13_weight_scale_list = [torch.empty(3) for _ in range(self.experts)]
        layer.w2_weight_scale_list = [torch.empty(2) for _ in range(self.experts)]
        layer.w13_scale_bias_list = [torch.empty(3) for _ in range(self.experts)]
        layer.w2_scale_bias_list = None

        with self.assertRaisesRegex(
            RuntimeError,
            "w13_scale_bias_list and w2_scale_bias_list to be present or absent together",
        ):
            self.quant_method.get_eplb_weight_views(layer)

    @patch("vllm_ascend.quantization.methods.w4a8.w4a8.get_current_vllm_config", new=lambda: None)
    @patch("vllm_ascend.quantization.methods.w4a8.w4a8.use_cam_fused_deep_moe", new=lambda _: False)
    @patch("vllm_ascend.quantization.methods.w4a8.w4a8.use_cann_megamoe", new=lambda _: False)
    @patch("vllm_ascend.quantization.methods.w4a8.w4a8.get_ascend_config")
    @patch("vllm_ascend.quantization.methods.w4a8.w4a8.maybe_trans_nz")
    @patch("torch.Tensor.npu", new=lambda self: self, create=True)
    def test_process_weights_after_loading(self, mock_maybe_trans_nz, mock_get_ascend_config):
        mock_maybe_trans_nz.side_effect = identity
        mock_get_ascend_config.return_value.enable_fused_mc2 = 0
        layer = self.build_layer()
        self.quant_method.process_weights_after_loading(layer)
        self.assertEqual(layer.w13_scale_bias.data.shape, (self.experts, 2 * self.input_size))
        self.assertEqual(layer.w13_scale_bias.data.dtype, torch.float32)
        self.assertEqual(layer.w2_scale_bias.data.shape, (self.experts, self.output_size))
        self.assertEqual(layer.w2_scale_bias.data.dtype, torch.float32)

        self.quant_method.use_expert_weight_list = True
        list_layer = self.build_layer()
        self.quant_method.process_weights_after_loading(list_layer)
        self.assertFalse(hasattr(list_layer, "w13_weight"))
        self.assertEqual(len(list_layer.w13_weight_list), self.experts)
        self.assertTrue(all(weight.storage_offset() == 0 for weight in list_layer.w13_weight_list))
        weight_views = self.quant_method.get_eplb_weight_views(list_layer)
        self.assertIs(weight_views[0], list_layer.w13_weight_list)
        self.assertIs(weight_views[-1], list_layer.w2_scale_bias_list)

    @patch("vllm_ascend.quantization.methods.w4a8.w4a8.get_current_vllm_config", new=lambda: None)
    @patch("vllm_ascend.quantization.methods.w4a8.w4a8.use_cam_fused_deep_moe", new=lambda _: False)
    @patch("vllm_ascend.quantization.methods.w4a8.w4a8.use_cann_megamoe", new=lambda _: False)
    @patch("vllm_ascend.quantization.methods.w4a8.w4a8.get_ascend_config")
    @patch("vllm_ascend.quantization.methods.w4a8.w4a8.maybe_trans_nz")
    @patch("torch.Tensor.npu", new=lambda self: self, create=True)
    def test_process_weights_after_loading_compressed_tensors(self, mock_maybe_trans_nz, mock_get_ascend_config):
        mock_maybe_trans_nz.side_effect = identity
        mock_get_ascend_config.return_value.enable_fused_mc2 = 0
        self.quant_method.quant_method = COMPRESSED_TENSORS_METHOD
        layer = self.build_layer_compressed_tensors()
        self.quant_method.process_weights_after_loading(layer)
        self.assertTrue(hasattr(layer, "w13_scale_bias"))
        self.assertEqual(layer.w13_scale_bias.data.shape, (self.experts, 2 * self.input_size))
        self.assertEqual(layer.w13_scale_bias.data.dtype, torch.float32)
        self.assertEqual(layer.w2_scale_bias.data.shape, (self.experts, self.output_size))

        self.quant_method.use_expert_weight_list = True
        list_layer = self.build_layer_compressed_tensors()
        self.quant_method.process_weights_after_loading(list_layer)
        self.assertFalse(hasattr(list_layer, "w13_weight"))
        self.assertEqual(len(list_layer.w13_weight_list), self.experts)
        self.assertEqual(list_layer.w13_weight_list[0].dtype, torch.int8)
        self.assertTrue(all(weight.storage_offset() == 0 for weight in list_layer.w13_weight_list))

    @patch("vllm_ascend.quantization.methods.w4a8.w4a8.get_current_vllm_config")
    @patch("vllm_ascend.quantization.methods.w4a8.w4a8.use_cam_fused_deep_moe", new=lambda _: True)
    @patch("vllm_ascend.quantization.methods.w4a8.w4a8.use_cann_megamoe", new=lambda _: False)
    @patch("vllm_ascend.quantization.methods.w4a8.w4a8.get_ascend_config")
    @patch("vllm_ascend.quantization.methods.w4a8.w4a8.maybe_trans_nz")
    @patch("torch.Tensor.npu", new=lambda self: self, create=True)
    def test_process_weights_builds_cam_int32_payload(
        self, mock_maybe_trans_nz, mock_get_ascend_config, mock_vllm_config
    ):
        mock_maybe_trans_nz.side_effect = identity
        mock_get_ascend_config.return_value.enable_fused_mc2 = 1
        mock_vllm_config.return_value = object()
        layer = self.build_cam_layer()
        self.quant_method.process_weights_after_loading(layer)
        self.assertEqual(layer.w13_weight.data.dtype, torch.int32)
        self.assertEqual(len(layer.cam_fused_deep_moe_w13_weight_list), self.experts)
        self.assertTrue(all(weight.dtype == torch.int32 for weight in layer.cam_fused_deep_moe_w13_weight_list))
        self.assertTrue(all(weight.storage_offset() == 0 for weight in layer.cam_fused_deep_moe_w13_weight_list))
        self.assertEqual(
            layer.cam_fused_deep_moe_w13_weight_list[0].shape,
            (_CAM_TEST_HIDDEN, _CAM_TEST_INTERMEDIATE // 4),
        )
        # CAM TensorList keeps the ND format tag (OpDef rejects FRACTAL_NZ).
        # Only stacked fallback weights are npu_format_cast for DFC.
        self.assertGreaterEqual(mock_maybe_trans_nz.call_count, 2)
        self.assertTrue(all(scale.dtype == torch.float32 for scale in layer.cam_fused_deep_moe_w13_weight_scale_list))
        self.assertTrue(all(bias.dtype == torch.float32 for bias in layer.cam_fused_deep_moe_w13_scale_bias_list))
        self.assertEqual(layer.w13_weight_scale.data.dtype, torch.int64)
        self.assertTrue(getattr(layer, CAM_FUSED_DEEP_MOE_PAYLOAD_VALIDATED_ATTR, False))

        cam_payload = self.quant_method.get_cam_fused_deep_moe_weights(layer)
        self.assertIs(cam_payload.w1, layer.cam_fused_deep_moe_w13_weight_list)
        self.assertIs(cam_payload.w1_scale, layer.cam_fused_deep_moe_w13_weight_scale_list)
        self.assertNotEqual(cam_payload.w1_scale[0].dtype, torch.int64)

        with patch(
            "vllm_ascend.quantization.methods.w4a8.w4a8._EXTRA_CTX",
            SimpleNamespace(use_cam_fused_deep_moe=True, use_mega_moe=False),
        ):
            fallback_payload = self.quant_method.get_fused_mc2_weights(layer)
        self.assertEqual(fallback_payload.w1_scale[0].dtype, torch.int64)

    @patch("vllm_ascend.quantization.methods.w4a8.w4a8.get_current_vllm_config")
    @patch("vllm_ascend.quantization.methods.w4a8.w4a8.use_cam_fused_deep_moe", new=lambda _: True)
    @patch("vllm_ascend.quantization.methods.w4a8.w4a8.use_cann_megamoe", new=lambda _: False)
    @patch("vllm_ascend.quantization.methods.w4a8.w4a8.get_ascend_config")
    @patch("vllm_ascend.quantization.methods.w4a8.w4a8.maybe_trans_nz")
    @patch("torch.Tensor.npu", new=lambda self: self, create=True)
    def test_cam_payload_keeps_gate_then_up_and_nz(
        self, mock_maybe_trans_nz, mock_get_ascend_config, mock_vllm_config
    ):
        """Official CAM and ModelSlim SwiGLU are both ``silu(left) * right``.

        Keep ModelSlim ``[gate | up]`` on the CAM copies; only INT32-pack and
        NZ-shuffle. Half-swapping would compute ``silu(up) * gate``.
        """
        mock_maybe_trans_nz.side_effect = identity
        mock_get_ascend_config.return_value.enable_fused_mc2 = 1
        mock_vllm_config.return_value = object()
        intermediate = _CAM_TEST_INTERMEDIATE
        hidden = _CAM_TEST_HIDDEN
        layer = self.build_cam_layer(intermediate, hidden)
        gate_packed = intermediate // 2
        layer.w13_weight.data[:, :gate_packed, :] = 1
        layer.w13_weight.data[:, gate_packed:, :] = 2
        layer.w13_weight_scale.data[:, :intermediate] = 3.0
        layer.w13_weight_scale.data[:, intermediate:] = 4.0
        layer.w13_scale_bias.data[:, :intermediate] = 5.0
        layer.w13_scale_bias.data[:, intermediate:] = 6.0
        layer.w2_scale_bias.data.fill_(7.0)
        self.quant_method.process_weights_after_loading(layer)

        cam_w13_nz = layer.cam_fused_deep_moe_w13_weight_list[0]
        self.assertEqual(cam_w13_nz.shape, (hidden, intermediate // 4))
        self.assertFalse(torch.equal(cam_w13_nz, _cam_int32_nz_to_nd(cam_w13_nz)))
        cam_w13 = _cam_int32_nz_to_nd(cam_w13_nz).view(torch.int8)
        self.assertEqual(cam_w13.shape, (hidden, intermediate))
        expected_w13 = torch.tensor([1] * gate_packed + [2] * gate_packed, dtype=torch.int8)
        self.assertTrue(torch.equal(cam_w13, expected_w13.expand_as(cam_w13)))
        cam_scale = layer.cam_fused_deep_moe_w13_weight_scale_list[0]
        self.assertTrue(torch.equal(cam_scale, torch.tensor([3.0] * intermediate + [4.0] * intermediate)))
        cam_bias = layer.cam_fused_deep_moe_w13_scale_bias_list[0]
        self.assertTrue(torch.equal(cam_bias, torch.tensor([5.0] * intermediate + [6.0] * intermediate)))
        dfc_w13 = layer.w13_weight.data.view(torch.int8)
        self.assertTrue(torch.equal(dfc_w13[0], expected_w13.expand_as(dfc_w13[0])))
        for w13_bias in layer.cam_fused_deep_moe_w13_scale_bias_list:
            self.assertEqual(w13_bias.numel(), 2 * intermediate)
            self.assertTrue(torch.count_nonzero(w13_bias) > 0)
        for w2_bias in layer.cam_fused_deep_moe_w2_scale_bias_list:
            self.assertTrue(torch.count_nonzero(w2_bias) > 0)

    def test_cam_gmm1_payload_uses_official_nz_without_swap(self):
        """CAM install must NZ-shuffle INT32 and must not half-swap GMM1."""
        payload_source = inspect.getsource(AscendW4A8DynamicFusedMoEMethod._install_cam_fused_deep_moe_payload)
        self.assertIn("_convert_nd_to_nz", payload_source)
        self.assertNotIn("_swap_last_dim_halves", payload_source)
        self.assertNotIn("maybe_trans_nz", payload_source)
        self.assertNotIn("npu_format_cast", payload_source)

    def test_pack_to_int32_asserts_packed_dim(self):
        weight = torch.zeros((self.experts, self.output_size, 10), dtype=torch.int8)
        expected_message = f"the last dim of weight needs to be divided by 4 but got shape {weight.shape}"

        with self.assertRaisesRegex(AssertionError, re.escape(expected_message)):
            self.quant_method._pack_to_int32(weight)

    def test_swap_last_dim_halves(self):
        swapped = AscendW4A8DynamicFusedMoEMethod._swap_last_dim_halves(torch.tensor([[1, 1, 2, 2]]))
        self.assertTrue(torch.equal(swapped, torch.tensor([[2, 2, 1, 1]])))
        with self.assertRaisesRegex(ValueError, "even last dim"):
            AscendW4A8DynamicFusedMoEMethod._swap_last_dim_halves(torch.zeros(3))

    def test_convert_nd_to_nz_roundtrip(self):
        rows, cols = 64, 16
        nd = torch.arange(rows * cols, dtype=torch.int32).reshape(rows, cols)
        nz = AscendW4A8DynamicFusedMoEMethod._convert_nd_to_nz(nd)
        self.assertEqual(tuple(nz.shape), tuple(nd.shape))
        self.assertFalse(torch.equal(nz, nd))
        self.assertTrue(torch.equal(_cam_int32_nz_to_nd(nz), nd))
        with self.assertRaisesRegex(ValueError, r"K%16==0"):
            AscendW4A8DynamicFusedMoEMethod._convert_nd_to_nz(torch.zeros(15, 8, dtype=torch.int32))
        with self.assertRaisesRegex(ValueError, "INT32"):
            AscendW4A8DynamicFusedMoEMethod._convert_nd_to_nz(torch.zeros(16, 8, dtype=torch.int8))

    @patch("vllm_ascend.quantization.methods.w4a8.w4a8.get_current_vllm_config")
    @patch("vllm_ascend.quantization.methods.w4a8.w4a8.use_cam_fused_deep_moe", new=lambda _: True)
    @patch("vllm_ascend.quantization.methods.w4a8.w4a8.use_cann_megamoe", new=lambda _: False)
    @patch("vllm_ascend.quantization.methods.w4a8.w4a8.get_ascend_config")
    @patch("vllm_ascend.quantization.methods.w4a8.w4a8.maybe_trans_nz")
    @patch("torch.Tensor.npu", new=lambda self: self, create=True)
    def test_cam_payload_rejects_unaligned_int32_shape(
        self, mock_maybe_trans_nz, mock_get_ascend_config, mock_vllm_config
    ):
        mock_maybe_trans_nz.side_effect = identity
        mock_get_ascend_config.return_value.enable_fused_mc2 = 1
        mock_vllm_config.return_value = object()
        layer = self.build_layer()
        with self.assertRaisesRegex(ValueError, "INT32 NZ shuffle"):
            self.quant_method.process_weights_after_loading(layer)

    @patch("vllm_ascend.quantization.methods.w4a8.w4a8._EXTRA_CTX")
    @patch("vllm_ascend.quantization.methods.w4a8.w4a8.build_fused_experts_input")
    def test_apply_comprehensive(self, mock_build_input, mock_ctx):
        tokens = 4
        num_experts = self.experts
        hidden_size = self.output_size
        top_k = 2

        layer = self.build_layer()
        layer.swiglu_limit = 1000000
        x = torch.randn(tokens, hidden_size, dtype=torch.bfloat16)
        topk_weights = torch.randn(tokens, top_k, dtype=torch.float32)
        topk_ids = torch.randint(0, num_experts, (tokens, top_k), dtype=torch.int64)
        expert_map = torch.randint(0, num_experts, (num_experts,), dtype=torch.int64)
        mc2_mask = torch.tensor([1, 0, 1, 0], dtype=torch.bool)
        pertoken_scale = torch.randn(tokens, dtype=torch.float32)
        log2phy = torch.randint(0, num_experts, (num_experts,), dtype=torch.int64)
        layer.ascend_expert_map = expert_map
        layer.global_redundant_expert_num = 0
        layer.log2phy = log2phy
        layer.ascend_mc2_mask = mc2_mask
        layer.ascend_pertoken_scale = pertoken_scale
        layer.activation = "silu"
        layer.apply_router_weight_on_input = False
        layer.swiglu_alpha = 1.0
        layer.swiglu_beta = 0.0

        mock_fused_input = Mock()
        mock_fused_input.hidden_states = x
        mock_fused_input.topk_weights = topk_weights
        mock_fused_input.topk_ids = topk_ids
        mock_fused_input.activation = "silu"
        mock_build_input.return_value = mock_fused_input

        mock_comm = Mock()
        expected_output = torch.randn(tokens, hidden_size, dtype=torch.bfloat16)
        mock_comm.fused_experts.return_value = expected_output
        mock_ctx.moe_comm_method = mock_comm
        mock_ctx.use_mega_moe = False

        output = self.quant_method.apply(
            layer=layer,
            x=x,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            shared_experts=None,
            shared_experts_input=None,
        )

        mock_build_input.assert_called_once()
        build_kwargs = mock_build_input.call_args.kwargs
        self.assertTrue(torch.equal(build_kwargs["hidden_states"], x))
        self.assertEqual(build_kwargs["quant_type"], self.quant_method.quant_type)
        self.assertTrue(build_kwargs["is_per_channel_weight"])
        self.assertEqual(build_kwargs["activation"], "silu")
        self.assertEqual(build_kwargs["apply_router_weight_on_input"], False)

        mock_comm.fused_experts.assert_called_once()
        self.assertEqual(mock_comm.fused_experts.call_args.kwargs["fused_experts_input"], mock_fused_input)
        self.assertTrue(torch.equal(output, expected_output))
