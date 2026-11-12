# Relatório automático - mlp_ea.py

Gerado em 2026-11-10 17:46. Janela = 2 s, 5 blocos por corrida, harmonização de limiar: sim (A >= 65.12 dBAE, B >= 54.99 dBAE).

- Corridas: 56 (1 por condição); janelas: 8884; descritores: 36.
- Hits com ROVL=1: 100.0%; SAT=1: 69.8%; MAXDUR=1: 17.8%.

## Frequência do motor (regressão; nível = arredondado a 20..90 Hz)

| esquema   | modelo   |   MAE_Hz |   RMSE_Hz |    R2 |   acc_nivel |   acc_10Hz |   MAE_Hz_corrida |   acc_nivel_corrida |
|:----------|:---------|---------:|----------:|------:|------------:|-----------:|-----------------:|--------------------:|
| LORO      | GB       |    5.601 |     8.964 | 0.849 |       0.554 |      0.935 |            5.466 |               0.554 |
| LORO      | MLP      |    5.098 |     9.074 | 0.845 |       0.72  |      0.931 |            4.63  |               0.786 |
| LORO      | RF       |    5.736 |     9.031 | 0.846 |       0.545 |      0.932 |            5.774 |               0.518 |
| LOSO      | GB       |    5.021 |     8.819 | 0.854 |       0.627 |      0.937 |            4.84  |               0.643 |
| LOSO      | MLP      |    5.013 |     9.313 | 0.837 |       0.714 |      0.94  |            4.365 |               0.732 |
| LOSO      | RF       |    4.858 |     8.462 | 0.865 |       0.657 |      0.938 |            4.751 |               0.696 |

## Configuração das válvulas (7 classes)

| esquema   | modelo   |   acc |   acc_bal |   F1_macro |   acc_rec |   acc_suc |   acc_corrida |
|:----------|:---------|------:|----------:|-----------:|----------:|----------:|--------------:|
| LORO      | GB       | 0.192 |     0.188 |      0.181 |     0.494 |     0.539 |         0.214 |
| LORO      | MLP      | 0.277 |     0.274 |      0.271 |     0.574 |     0.575 |         0.286 |
| LORO      | RF       | 0.208 |     0.206 |      0.19  |     0.527 |     0.539 |         0.196 |

Referência de acaso: acc (7 classes equilibradas) = 0.143; acc_rec e acc_suc chutando sempre 100 % = 0.571 (4 das 7 configurações têm 100 %).


## Métricas por fold (BLOCOS)

| fold   |
|--------|

## Não supervisionado (k-means sobre os descritores padronizados)

|   k |   config |   corrida |   dia |   freq |   limiar |
|----:|---------:|----------:|------:|-------:|---------:|
|   2 |    0.002 |     0.024 | 0.007 |  0.163 |    0.01  |
|   3 |    0.089 |     0.055 | 0.013 |  0.202 |   -0.005 |
|   7 |    0.093 |     0.106 | 0.036 |  0.284 |    0.013 |
|   8 |    0.093 |     0.106 | 0.037 |  0.284 |    0.014 |
|  14 |    0.113 |     0.163 | 0.04  |  0.249 |    0.041 |
|  56 |    0.096 |     0.375 | 0.018 |  0.169 |    0.027 |

(ARI; 0 = acaso, 1 = partição idêntica.)


## 10 descritores mais importantes (média dos 3 modelos)

- freq: A_log_energia_s (8.29), A_log_energia_p90 (5.31), B_amp_p10 (4.43), A_fm_kHz_med (2.43), B_log_energia_p90 (2.22), A_log_energia_p50 (2.16), A_log_dur_p90 (1.84), B_log_dur_p90 (1.74), B_log_energia_p50 (1.71), B_log_energia_s (1.6)
- config: A_fm_kHz_med (0.222), A_log_energia_s (0.127), A_frac_maxdur (0.11), A_log_cont_p90 (0.102), A_log_dur_p50 (0.0802), A_log_cont_p50 (0.0779), A_log_dur_p90 (0.0757), B_log_cont_p90 (0.0709), B_log_dur_p90 (0.0698), B_log_dur_p50 (0.0681)
