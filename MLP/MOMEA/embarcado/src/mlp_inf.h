/* Inferência das MLPs de EA para a MicroBlaze (ou PC).
 * Entrada: vetor bruto de 36 descritores de uma janela de 5 s, na ordem listada em
 * gerado/mlp_pesos_f32.h. NaN indica descritor indefinido (janela sem hits normais)
 * e é substituído pela mediana de treino. */
#ifndef MLP_INF_H
#define MLP_INF_H
#include <stdint.h>
#include "mlp_nucleo_q8.h"

#define MLP_N_ENTRADAS MLP_NUCLEO_N_IN
#define MLP_N_CLASSES  MLP_NUCLEO_N_CFG

/* ---- caminho float32 (requer FPU na MicroBlaze para ser rápido) ---- */
float mlp_freq_f32(const float x[MLP_N_ENTRADAS]);                          /* Hz */
int   mlp_cfg_f32(const float x[MLP_N_ENTRADAS], float logits[MLP_N_CLASSES]); /* classe */

/* ---- caminho int8 ---- */
void  mlp_quantiza_entrada(const float x[MLP_N_ENTRADAS], int8_t xq[MLP_N_ENTRADAS]);
float mlp_freq_de_acc(int32_t acc_freq);                                    /* Hz */
/* conveniência: pré-processa, roda o núcleo int8 e devolve frequência (Hz) e classe */
void  mlp_q8(const float x[MLP_N_ENTRADAS], float *freq_hz, int *classe,
             int32_t acc_cfg[MLP_N_CLASSES]);

const char *mlp_nome_classe(int k);   /* "recalque-sucção", ex. "100-50" */

#endif
