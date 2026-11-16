diff a/hyper_multicore/README.md b/hyper_multicore/README.md	(rejected hunks)
@@ -259,10 +261,11 @@ output = experts(
 
 `MegaMoeNumericsOptions` 是公开稳定的数值配置接口，可直接从 `hyper_multicore` 导入。
 类型定义位于 `modules/mega_moe/numerics.py`，由 MegaMoE 模块管理。
-通过 `MegaMoeExperts(..., numerics_options=...)` 按模型逐层配置以下三个字段：
+通过 `MegaMoeExperts(..., numerics_options=...)` 按模型逐层配置以下字段：
 
 | 字段 | 默认值 | 语义 |
 | --- | --- | --- |
+| `partial_weight_grad_dtype` | `None` | 保留后端既有默认；`"native"` 按参数 dtype 写 partial dW；`"float32"` 以 FP32 保存 partial dW。最终参数梯度和 main_grad 的 dtype 由调用方决定。 |
 | `swiglu_limit` | `None` | 不进行 clamp；提供时须为正、有限且可由 FP32 表示的值。gate 分支只截上界，up 分支截到 `[-limit, limit]`。 |
 | `clamp_after_silu` | `False` | `False` 在 SiLU 前截 gate；`True` 在 SiLU(gate) 后、乘 up 前截 gate 激活。仅在提供 `swiglu_limit` 时生效。 |
 | `apply_probs_before_fc2` | `False` | `False` 在 FC2 后对专家输出加权；`True` 对 BF16 SwiGLU 激活加权，再送入 FC2。 |
