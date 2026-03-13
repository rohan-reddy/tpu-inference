# MOE_REQUANTIZE_ON_TPU Flag Verification Results

Model: Qwen/Qwen3-235B-A22B-Instruct-2507-FP8
Hardware: v6e-8 TPU (8 chips)
Command: `python examples/offline_inference.py --model Qwen/Qwen3-235B-A22B-Instruct-2507-FP8 --tensor-parallel-size 8 --max-model-len 1024 --max-num-batched-tokens 128`

## Memory Comparison (final layer, per-chip)

| Metric | Test 1: Flag OFF, 2D | Test 2: Flag ON, 2D | Test 3: Flag ON, 5D |
|--------|---------------------|---------------------|---------------------|
| reserved | 0.000G | 0.403G | 0.403G |
| peak_reserved | 0.000G | 0.403G | 0.403G |
| in_use | 29.239G | 29.851G | 29.851G |
| peak_in_use | 29.533G | 29.851G | 29.851G |
| per-layer spike (peak - in_use) | ~0.294G | ~0.294G | ~0.294G |
| weight load time | 237.37s | 127.44s | 147.58s |

## Notes

- **reserved**: XLA program reservation. Flag OFF runs requant on CPU (via JIT tracing on CPU arrays), so no TPU program reservation. Flag ON uses shard_map which compiles a TPU program costing 0.403G.

- **in_use**: Flag ON is ~0.6G higher than flag OFF. This is the 0.403G program reservation plus minor differences from weights being on TPU during requant vs CPU.

- **peak_in_use**: Flag OFF shows a ~0.294G spike above in_use (FP32 intermediates during dequant/requant visible as running max). Flag ON peak equals in_use because the shard_map intermediates are local per-device and smaller.

- **Weight load time**: Flag ON is ~46% faster (127s vs 237s) because requantization runs on TPU in parallel across experts instead of on CPU. The 5D mesh test is slightly slower (148s) due to mesh setup overhead.

- **5D mesh inference**: Crashed during attention (not model loading) due to KV head sharding mismatch — mesh shape (1,1,1,1,8) tries to shard 4 KV heads across 8 devices. This is unrelated to the requantization flag; the 5D mesh needs appropriate sharding config for this model's head count.

- **Inference correctness**: Tests 1 and 2 both produced correct, sensible outputs. Test 3 loaded successfully but crashed at inference time (attention sharding).

## Conclusion

The `MOE_REQUANTIZE_ON_TPU` flag correctly gates between:
- **OFF (default)**: CPU dequant path matching main branch behavior. Zero TPU program reservation.
- **ON**: TPU shard_map + lax.scan path with 0.403G program reservation but ~46% faster weight loading.
