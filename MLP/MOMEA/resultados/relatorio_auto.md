# Relatório automático - mlp_ea.py

Gerado em 2026-11-10 17:32. Janela = 5 s, 5 blocos por corrida, harmonização de limiar: sim (A >= 65.12 dBAE, B >= 54.99 dBAE).

- Corridas: 56 (1 por condição); janelas: 3540; descritores: 36.
- Hits com ROVL=1: 100.0%; SAT=1: 69.8%; MAXDUR=1: 17.8%.

## Frequência do motor (regressão; nível = arredondado a 20..90 Hz)

| esquema   | modelo   |   MAE_Hz |   RMSE_Hz |    R2 |   acc_nivel |   acc_10Hz |   MAE_Hz_corrida |   acc_nivel_corrida |
|:----------|:---------|---------:|----------:|------:|------------:|-----------:|-----------------:|--------------------:|
| BLOCOS    | GB       |    1.161 |     2.977 | 0.983 |       0.952 |      0.988 |            0.41  |               0.982 |
| BLOCOS    | MLP      |    2.061 |     4.474 | 0.962 |       0.927 |      0.991 |            0.865 |               0.982 |
| BLOCOS    | RF       |    1.079 |     2.841 | 0.985 |       0.943 |      0.99  |            0.523 |               0.982 |
| LORO      | GB       |    4.965 |     8.026 | 0.879 |       0.608 |      0.949 |            4.838 |               0.625 |
| LORO      | MLP      |    5.608 |    10.562 | 0.79  |       0.691 |      0.918 |            4.975 |               0.714 |
| LORO      | RF       |    4.846 |     7.872 | 0.883 |       0.612 |      0.942 |            4.724 |               0.589 |
| LOSO      | GB       |    4.209 |     7.716 | 0.888 |       0.701 |      0.951 |            3.935 |               0.714 |
| LOSO      | MLP      |    5.094 |     8.925 | 0.85  |       0.698 |      0.939 |            4.394 |               0.768 |
| LOSO      | RF       |    4.232 |     7.801 | 0.885 |       0.691 |      0.951 |            4.013 |               0.679 |

## Configuração das válvulas (7 classes)

| esquema   | modelo   |   acc |   acc_bal |   F1_macro |   acc_rec |   acc_suc |   acc_corrida |
|:----------|:---------|------:|----------:|-----------:|----------:|----------:|--------------:|
| BLOCOS    | GB       | 0.934 |     0.933 |      0.933 |     0.948 |     0.964 |         1     |
| BLOCOS    | MLP      | 0.888 |     0.888 |      0.888 |     0.919 |     0.946 |         1     |
| BLOCOS    | RF       | 0.945 |     0.944 |      0.944 |     0.958 |     0.97  |         1     |
| LORO      | GB       | 0.224 |     0.221 |      0.209 |     0.546 |     0.532 |         0.232 |
| LORO      | MLP      | 0.28  |     0.279 |      0.276 |     0.573 |     0.594 |         0.304 |
| LORO      | RF       | 0.21  |     0.208 |      0.194 |     0.535 |     0.53  |         0.179 |

Referência de acaso: acc (7 classes equilibradas) = 0.143; acc_rec e acc_suc chutando sempre 100 % = 0.571 (4 das 7 configurações têm 100 %).


## Métricas por fold (BLOCOS)

| fold   |   ('MAE_Hz', 'freq', 'GB') |   ('MAE_Hz', 'freq', 'MLP') |   ('MAE_Hz', 'freq', 'RF') |   ('acc', 'config', 'GB') |   ('acc', 'config', 'MLP') |   ('acc', 'config', 'RF') |
|:-------|---------------------------:|----------------------------:|---------------------------:|--------------------------:|---------------------------:|--------------------------:|
| bloco1 |                      1.26  |                       2.006 |                      1.282 |                     0.92  |                      0.863 |                     0.938 |
| bloco2 |                      1.033 |                       1.906 |                      0.947 |                     0.943 |                      0.902 |                     0.956 |
| bloco3 |                      1.061 |                       1.692 |                      0.895 |                     0.957 |                      0.909 |                     0.96  |
| bloco4 |                      1.069 |                       1.91  |                      0.982 |                     0.926 |                      0.901 |                     0.935 |
| bloco5 |                      1.38  |                       2.804 |                      1.283 |                     0.922 |                      0.867 |                     0.937 |

## Não supervisionado (k-means sobre os descritores padronizados)

|   k |   config |   corrida |   dia |   freq |   limiar |
|----:|---------:|----------:|------:|-------:|---------:|
|   2 |    0.002 |     0.024 | 0.007 |  0.163 |    0.01  |
|   3 |    0.094 |     0.055 | 0.012 |  0.202 |   -0.007 |
|   7 |    0.095 |     0.11  | 0.035 |  0.309 |    0.011 |
|   8 |    0.096 |     0.11  | 0.035 |  0.31  |    0.011 |
|  14 |    0.145 |     0.205 | 0.038 |  0.309 |    0.063 |
|  56 |    0.124 |     0.465 | 0.023 |  0.2   |    0.034 |

(ARI; 0 = acaso, 1 = partição idêntica.)


## 10 descritores mais importantes (média dos 3 modelos)

- freq: A_log_energia_s (6.58), B_log_RA_p50 (4.22), A_log_energia_p90 (3.38), B_log_energia_p50 (2.79), B_amp_p10 (2.75), B_log_energia_p90 (2.02), A_log_energia_p50 (2.02), B_log_dur_p90 (1.8), B_amp_med (1.63), A_fm_kHz_med (1.61)
- config: A_fm_kHz_med (0.161), A_frac_maxdur (0.113), B_log_dur_p50 (0.0988), A_log_cont_p90 (0.0891), B_log_dur_p90 (0.075), A_log_cont_p50 (0.0749), A_taxa_hits (0.072), B_log_cont_p90 (0.0692), B_log_cont_p50 (0.0612), A_log_energia_s (0.0604)
