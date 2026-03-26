# Copyright 2025 Google LLC
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

import functools
from typing import Optional, Union

import jax
import jax.numpy as jnp
import torch
from jax.sharding import NamedSharding, PartitionSpec
from torch.nn.parameter import Parameter
from torchax.interop import jax_view, torch_view
from vllm.model_executor.layers.linear import LinearBase, LinearMethodBase
from vllm.model_executor.layers.quantization import \
    register_quantization_config
from vllm.model_executor.layers.quantization.base_config import \
    QuantizeMethodBase
from vllm.model_executor.layers.quantization.gptq import (GPTQConfig,
                                                          GPTQLinearMethod)

from tpu_inference.kernels.megablox.gmm_v2 import gmm_v2
from tpu_inference.layers.common.process_weights.linear_weights import (
    LinearWeights, process_linear_weights, to_parameter_list)
from tpu_inference.layers.common.quant_methods import GPTQ
from tpu_inference.layers.common.quantization import gptq_i32_unpack_u4
from tpu_inference.layers.common.sharding import ShardingAxisName
from tpu_inference.layers.common.utils import (
    general_device_put,
    slice_sharded_tensor_for_concatenation)
from tpu_inference.layers.vllm.quantization.configs import (
    VllmQuantConfig, VllmQuantLinearConfig)
from tpu_inference.logger import init_logger
from tpu_inference.utils import t2j

P = PartitionSpec

logger = init_logger(__name__)


@register_quantization_config(GPTQ)
class VllmGPTQConfig(GPTQConfig, VllmQuantConfig):

    @classmethod
    def get_name(cls):
        return GPTQ

    def get_supported_act_dtypes(self) -> list[torch.dtype]:
        # NOTE: GPTQ checkpoints typically use float16 scales/zeros. On TPUs,
        # bfloat16 is significantly preferred over float16. This may lead to
        # minor numeric differences but is acceptable given 4-bit weight
        # quantization.
        return [torch.bfloat16]

    def get_quant_method(
        self, layer: torch.nn.Module, prefix: str
    ) -> Optional[Union["LinearMethodBase", "QuantizeMethodBase"]]:
        if isinstance(layer, LinearBase):
            linear_config = self.get_linear_config(layer)
            return VllmGPTQLinearMethod(self, linear_config)
        return None


def _dense_gmm_local(x, weight, scale, *, group_size):
    """Run GMM kernel on a local shard for dense matmul.

    Treats the dense linear as a single-group GMM, allowing the kernel
    to handle int8 weights with subchannel scale application.

    Args:
        x: Input activations, shape (batch, K_local).
        weight: Int8 weight (zero-point subtracted), shape (K_local, N_local).
        scale: Per-group scales, shape (num_groups_local, N_local).
        group_size: Number of input channels per quantization group.
    """
    batch = x.shape[0]
    rhs = weight[jnp.newaxis, :, :]  # (1, K_local, N_local)
    rhs_scale = scale[jnp.newaxis, :, jnp.newaxis, :]  # (1, G_local, 1, N_local)
    group_sizes = jnp.array([batch], dtype=jnp.int32)

    return gmm_v2(
        lhs=x,
        rhs=rhs,
        rhs_scale=rhs_scale,
        group_sizes=group_sizes,
        zero_initialize=False,
        maybe_quantize_lhs=False,
    )


class VllmGPTQLinearMethod(GPTQLinearMethod):

    def __init__(self, quant_config: VllmGPTQConfig,
                 linear_config: VllmQuantLinearConfig):
        super().__init__(quant_config)
        self.linear_config = linear_config

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        # GPTQ qweight: packed_dim=0 (input dim packed)
        assert layer.qweight.packed_dim == 0
        weight = t2j(layer.qweight, use_dlpack=False)
        delattr(layer, "qweight")

        weight_scale = t2j(layer.scales, use_dlpack=False)
        delattr(layer, "scales")

        # GPTQ qzeros: packed_dim=1 (output dim packed)
        assert layer.qzeros.packed_dim == 1
        zero_point = t2j(layer.qzeros, use_dlpack=False)
        delattr(layer, "qzeros")

        g_idx = t2j(layer.g_idx, use_dlpack=False)
        delattr(layer, "g_idx")

        if layer.bias is not None and not layer.skip_bias_add:
            if layer.return_bias:
                logger.warning_once("Bias might return incorrect value.")
            bias = t2j(layer.bias, use_dlpack=False)
            delattr(layer, "bias")
        else:
            bias = None

        # Check if g_idx requires input permutation (desc_act=True).
        # For desc_act=False, g_idx is sequential and no permutation needed.
        needs_perm = not bool(jnp.all(g_idx[:-1] <= g_idx[1:]))

        @jax.jit
        def process_gptq_gmm_weights(
            weight: jax.Array,
            weight_scale: jax.Array,
            zero_point: jax.Array,
            g_idx: jax.Array,
            bias: jax.Array | None,
        ) -> tuple[LinearWeights, jax.Array]:
            # GPTQ qweight is packed along dim 0 (input dimension).
            # Shape: (input/8, output). Transpose so packed dim is last,
            # unpack, then transpose back to (input, output).
            weight = gptq_i32_unpack_u4(weight.T).T

            # GPTQ qzeros is packed along dim 1 (output dimension).
            # Shape: (num_groups, output/8) -> (num_groups, output)
            zero_point = gptq_i32_unpack_u4(zero_point)

            # GPTQ v1 format stores zero_point as (true_zero - 1).
            # AutoGPTQ does `zeros -= 1` before packing, so we add 1
            # to recover the true zero point.
            if not self.use_v2_format:
                zero_point = zero_point.astype(jnp.int8) + jnp.int8(1)

            # Sort weight rows by g_idx so quantization groups are contiguous.
            # For desc_act=False this is already the case (identity sort).
            # For desc_act=True this reorders rows so subchannel quant works.
            sort_indices = jnp.argsort(g_idx)
            weight = weight[sort_indices, :]
            sorted_g_idx = g_idx[sort_indices]

            # Subtract zero point per group to get signed int8 weights.
            # After this, weight values are centered around 0.
            zeros_per_row = zero_point[sorted_g_idx]
            weight = weight.astype(jnp.int8) - zeros_per_row.astype(jnp.int8)

            # weight is now (K, N) int8, zero-adjusted.
            # weight_scale is (num_groups, N) -- one scale per group per output.
            # These map directly to GMM subchannel quantization where
            # quant_block_size = group_size and num_quant_blocks = num_groups.

            weights = process_linear_weights(
                LinearWeights(
                    weight=weight,
                    weight_scale=weight_scale,
                    zero_point=None,
                    bias=bias,
                ),
                fused=False,  # Always split for GMM to avoid VMEM OOM on large N
                output_sizes=self.linear_config.output_sizes,
                reorder_size=self.linear_config.n_shards,
                transposed=False,
            )
            return weights, sort_indices

        weights, sort_indices = process_gptq_gmm_weights(
            weight, weight_scale, zero_point, g_idx, bias)

        # Manually shard weights. We cannot use shard_linear_weights because
        # it does not correctly handle 2D scale tensors with transposed=False.
        mesh = self.linear_config.mesh
        orig_p_spec = self.linear_config.weight_sharding
        # Reverse for non-transposed layout: (K, N) instead of (N, K)
        weight_p_spec = P(*orig_p_spec[::-1])
        weight_sharding = NamedSharding(mesh, weight_p_spec)
        # Scale (num_groups, N) follows same sharding as weight (K, N):
        # both have input-related dim first, output dim last.
        scale_sharding = NamedSharding(mesh, weight_p_spec)
        bias_p_spec = P(weight_p_spec[-1])
        bias_sharding = NamedSharding(mesh, bias_p_spec)

        def shard(arr, sharding):
            if arr is None:
                return None
            if isinstance(arr, list):
                return [general_device_put(a, sharding) for a in arr]
            return general_device_put(arr, sharding)

        sharded_weight = shard(weights.weight, weight_sharding)
        sharded_scale = shard(weights.weight_scale, scale_sharding)
        sharded_bias = shard(weights.bias, bias_sharding)

        if False:  # Always use split path for GMM
            layer.weight = Parameter(
                torch_view(sharded_weight), requires_grad=False)
            layer.weight_scale = Parameter(
                torch_view(sharded_scale), requires_grad=False)
            if sharded_bias is not None:
                layer.bias = Parameter(
                    torch_view(sharded_bias), requires_grad=False)
        else:
            layer.weight = to_parameter_list(
                [torch_view(w) for w in sharded_weight])
            layer.weight_scale = to_parameter_list(
                [torch_view(s) for s in sharded_scale])
            if sharded_bias is not None:
                layer.bias = to_parameter_list(
                    [torch_view(b) for b in sharded_bias])

        # Store input permutation for desc_act=True models.
        if needs_perm:
            layer.input_perm = Parameter(
                torch_view(sort_indices), requires_grad=False)

    def _get_tp_axis(self):
        """Determine which mesh axis is used for tensor parallelism."""
        p_spec = self.linear_config.weight_sharding
        for axis in p_spec:
            if axis is not None:
                return axis
        return None

    def _is_row_parallel(self):
        """Check if this is a row-parallel linear (K dimension sharded).

        For transposed weight_sharding P(out_axis, in_axis):
        - Column-parallel: P(tp, None) -> out dim sharded
        - Row-parallel: P(None, tp) -> in dim sharded
        """
        p_spec = self.linear_config.weight_sharding
        return p_spec[0] is None and p_spec[1] is not None

    def _gmm_matmul(self, x_jax: jax.Array, weight: jax.Array,
                     scale: jax.Array) -> jax.Array:
        """Perform quantized matmul using GMM V2 kernel via shard_map.

        Uses shard_map to handle multi-device tensor parallelism, since
        Pallas kernels cannot be auto-partitioned by XLA.

        Args:
            x_jax: Input activations, shape (batch, K).
            weight: Int8 weight with zero points subtracted, shape (K, N).
            scale: Per-group scales, shape (num_groups, N).

        Returns:
            Output of shape (batch, N).
        """
        mesh = self.linear_config.mesh
        tp_axis = self._get_tp_axis()
        is_row_parallel = self._is_row_parallel()

        # Non-transposed weight sharding: P(K_shard, N_shard)
        weight_p_spec = P(*self.linear_config.weight_sharding[::-1])

        if is_row_parallel:
            # Row-parallel: K is sharded, need psum after local matmul
            x_p_spec = P(None, tp_axis)
            scale_p_spec = P(tp_axis, None)
            out_p_spec = P()

            def _local_fn(x, w, s):
                out = _dense_gmm_local(
                    x, w, s, group_size=self.quant_config.group_size)
                return jax.lax.psum(out, axis_name=tp_axis)
        else:
            # Column-parallel: N is sharded, no reduction needed
            x_p_spec = P()
            scale_p_spec = P(None, tp_axis)
            out_p_spec = P(None, tp_axis)

            _local_fn = functools.partial(
                _dense_gmm_local,
                group_size=self.quant_config.group_size)

        return jax.shard_map(
            _local_fn,
            mesh=mesh,
            in_specs=(x_p_spec, weight_p_spec, scale_p_spec),
            out_specs=out_p_spec,
            check_vma=False,
        )(x_jax, weight, scale)

    def apply(self,
              layer: torch.nn.Module,
              x: torch.Tensor,
              bias: Optional[torch.Tensor] = None) -> torch.Tensor:

        with jax.named_scope(layer._get_name()):
            if False:  # Always use split path for GMM
                out = self._apply_fused(layer, x, bias)
            else:
                out = self._apply_split(layer, x, bias)

        return out

    def _apply_fused(self,
                     layer: torch.nn.Module,
                     x: torch.Tensor,
                     bias: Optional[torch.Tensor] = None) -> torch.Tensor:
        x_jax = jax_view(x)

        # Permute input channels for desc_act=True models.
        if hasattr(layer, 'input_perm'):
            x_jax = x_jax[:, jax_view(layer.input_perm)]

        weight = jax_view(layer.weight)
        scale = jax_view(layer.weight_scale)
        outs = self._gmm_matmul(x_jax, weight, scale)

        if bias is not None and not layer.skip_bias_add:
            outs += bias.jax()

        outs = slice_sharded_tensor_for_concatenation(
            outs, self.linear_config.output_sizes, self.linear_config.n_shards)
        out = jnp.concatenate(outs, axis=-1)
        return torch_view(out)

    def _apply_split(self,
                     layer: torch.nn.Module,
                     x: torch.Tensor,
                     bias: Optional[torch.Tensor] = None) -> torch.Tensor:
        assert isinstance(layer.weight, torch.nn.ParameterList)

        x_jax = jax_view(x)

        # Permute input channels for desc_act=True models.
        if hasattr(layer, 'input_perm'):
            x_jax = x_jax[:, jax_view(layer.input_perm)]

        outs = []
        for i, (w, s) in enumerate(zip(layer.weight, layer.weight_scale)):
            out = self._gmm_matmul(x_jax, jax_view(w), jax_view(s))

            if bias is not None and not layer.skip_bias_add:
                out += jax_view(bias[i])

            outs.append(out)
        out = jnp.concatenate(outs, axis=-1)
        return torch_view(out)
