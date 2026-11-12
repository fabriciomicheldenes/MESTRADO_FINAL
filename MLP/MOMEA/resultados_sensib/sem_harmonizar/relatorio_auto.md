# Relatório automático - mlp_ea.py

Gerado em 2026-11-10 17:36. Janela = 5 s, 5 blocos por corrida, harmonização de limiar: não.

- Corridas: 56 (1 por condição); janelas: 3540; descritores: 36.
- Hits com ROVL=1: 100.0%; SAT=1: 69.8%; MAXDUR=1: 17.8%.

## Frequência do motor (regressão; nível = arredondado a 20..90 Hz)

| esquema   | modelo   |   MAE_Hz |   RMSE_Hz |    R2 |   acc_nivel |   acc_10Hz |   MAE_Hz_corrida |   acc_nivel_corrida |
|:----------|:---------|---------:|----------:|------:|------------:|-----------:|-----------------:|--------------------:|
| LORO      | GB       |    4.973 |     8.032 | 0.878 |       0.609 |      0.948 |            4.873 |               0.625 |
| LORO      | MLP      |    5.444 |     9.6   | 0.826 |       0.689 |      0.926 |            4.922 |               0.732 |
| LORO      | RF       |    4.846 |     7.872 | 0.883 |       0.612 |      0.942 |            4.725 |               0.589 |
| LOSO      | GB       |    4.216 |     7.724 | 0.888 |       0.698 |      0.951 |            3.933 |               0.714 |
| LOSO      | MLP      |    5.082 |     9.026 | 0.846 |       0.714 |      0.938 |            4.381 |               0.768 |
| LOSO      | RF       |    4.232 |     7.802 | 0.885 |       0.691 |      0.951 |            4.013 |               0.679 |

## Configuração das válvulas (7 classes)

| esquema   | modelo   |   acc |   acc_bal |   F1_macro |   acc_rec |   acc_suc |   acc_corrida |
|:----------|:---------|------:|----------:|-----------:|----------:|----------:|--------------:|
| LORO      | GB       | 0.219 |     0.216 |      0.207 |     0.545 |     0.529 |         0.214 |
| LORO      | MLP      | 0.283 |     0.283 |      0.28  |     0.58  |     0.598 |         0.286 |
| LORO      | RF       | 0.211 |     0.208 |      0.194 |     0.536 |     0.532 |         0.179 |

Referência de acaso: acc (7 classes equilibradas) = 0.143; acc_rec e acc_suc chutando sempre 100 % = 0.571 (4 das 7 configurações têm 100 %).


## Métricas por fold (BLOCOS)

| fold   |
|--------|

## Não supervisionado (k-means sobre os descritores padronizados)

|   k |   config |   corrida |   dia |   freq |   limiar |
|----:|---------:|----------:|------:|-------:|---------:|
|   2 |    0.002 |     0.024 | 0.007 |  0.163 |    0.01  |
|   3 |    0.094 |     0.055 | 0.012 |  0.202 |   -0.007 |
|   7 |    0.095 |     0.11  | 0.035 |  0.309 |    0.011 |
|   8 |    0.096 |     0.11  | 0.035 |  0.31  |    0.011 |
|  14 |    0.145 |     0.205 | 0.038 |  0.309 |    0.063 |
|  56 |    0.126 |     0.479 | 0.023 |  0.209 |    0.034 |

(ARI; 0 = acaso, 1 = partição idêntica.)


## 10 descritores mais importantes (média dos 3 modelos)

- freq: A_log_energia_s (6.76), B_log_RA_p50 (4.17), A_log_energia_p90 (3.38), B_log_energia_p50 (2.79), B_amp_p10 (2.73), B_log_energia_p90 (2.07), A_log_energia_p50 (1.96), B_log_dur_p90 (1.77), A_fm_kHz_med (1.64), B_amp_med (1.61)
- config: A_fm_kHz_med (0.161), A_frac_maxdur (0.115), B_log_dur_p50 (0.0979), A_log_cont_p90 (0.0886), B_log_dur_p90 (0.0772), A_log_cont_p50 (0.0749), A_taxa_hits (0.0737), B_log_cont_p90 (0.0696), B_log_cont_p50 (0.0622), A_log_energia_s (0.0616)
