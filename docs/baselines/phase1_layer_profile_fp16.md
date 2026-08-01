# Phase 1d — TensorRT vision engine layer profile (fp16)

Engine `vision_tower_fp16.plan` at medium_448px (784 patches), 20 runs, 24.30 ms/run total.

| op family | ms/run | % total | layers |
|---|---|---|---|
| gemm / conv | 20.881 | 85.9% | 131 |
| elementwise | 2.959 | 12.2% | 100 |
| other | 0.458 | 1.9% | 27 |

## Top layers

| layer | ms/run | % |
|---|---|---|
| `__myl_FcAddCastMeanSubMulMeanAddSqrtDivMulCastMulAdd_myl5_249` | 0.374 | 1.5% |
| `/mlp/linear_fc1_14/Gemm_myl5_155` | 0.343 | 1.4% |
| `/mlp/linear_fc1_1/Gemm_myl5_19` | 0.322 | 1.3% |
| `/mlp/linear_fc1_18/Gemm_myl5_198` | 0.313 | 1.3% |
| `/mlp/linear_fc1_5/Gemm_myl5_59` | 0.312 | 1.3% |
| `/mlp/linear_fc1_16/Gemm_myl5_175` | 0.311 | 1.3% |
| `/mlp/linear_fc1_9/Gemm_myl5_102` | 0.304 | 1.3% |
| `/mlp/linear_fc1_10/Gemm_myl5_112` | 0.304 | 1.3% |
| `/mlp/linear_fc1_13/Gemm_myl5_145` | 0.303 | 1.2% |
| `/mlp/linear_fc1_8/Gemm_myl5_92` | 0.303 | 1.2% |
| `/mlp/linear_fc1_22/Gemm_myl5_238` | 0.299 | 1.2% |
| `/mlp/linear_fc1_21/Gemm_myl5_228` | 0.298 | 1.2% |
| `/mlp/linear_fc1_11/Gemm_myl5_122` | 0.293 | 1.2% |
| `/mlp/linear_fc1_15/Gemm_myl5_165` | 0.291 | 1.2% |
| `/mlp/linear_fc1_2/Gemm_myl5_29` | 0.291 | 1.2% |
| `/mlp/linear_fc1_12/Gemm_myl5_135` | 0.291 | 1.2% |
| `/mlp/linear_fc1_7/Gemm_myl5_82` | 0.290 | 1.2% |
| `/mlp/linear_fc1_3/Gemm_myl5_39` | 0.290 | 1.2% |
| `/mlp/linear_fc1_4/Gemm_myl5_49` | 0.290 | 1.2% |
| `/mlp/linear_fc1_6/Gemm_myl5_72` | 0.290 | 1.2% |
