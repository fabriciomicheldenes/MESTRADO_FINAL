/* Núcleo int8 das MLPs 36-64-32-k (FREQ: k=1, CFG: k=7).
 * O mesmo fonte compila no PC, na MicroBlaze (mb-gcc) e no Vitis HLS (função de topo). */
#ifndef MLP_NUCLEO_Q8_H
#define MLP_NUCLEO_Q8_H
#include <stdint.h>

#define MLP_NUCLEO_N_IN 36
#define MLP_NUCLEO_N_CFG 7

/* xq      : descritores padronizados e quantizados (mlp_quantiza_entrada)
 * acc_freq: acumulador int32 da saída do regressor (Hz = acc*FREQ_Q_S_OUT*FREQ_Y_ESCALA + FREQ_Y_MEDIA)
 * acc_cfg : acumuladores int32 das 7 classes (logits = acc*CFG_Q_S_OUT)
 * classe  : índice da classe vencedora (empate: menor índice) */
void mlp_nucleo_q8(const int8_t xq[MLP_NUCLEO_N_IN], int32_t acc_freq[1],
                   int32_t acc_cfg[MLP_NUCLEO_N_CFG], int32_t classe[1]);

#endif
