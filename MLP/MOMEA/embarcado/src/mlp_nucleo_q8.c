/* Núcleo int8 das MLPs - ver mlp_nucleo_q8.h.
 *
 * Aritmética (idêntica à emulação em exportar_modelo.py):
 *   camada oculta: acc = B[o] + sum_i x[i]*W[i][o]           (int32)
 *                  y   = min(127, (max(acc,0)*M + 2^30) >> 31) (int8, ReLU embutida)
 *   camada de saída: acumulador int32, sem requantização.
 *
 * Em HLS cada laço interno é pipelinado com II=1: 1 multiplicação-acumulação por ciclo,
 * ~4,6 mil ciclos por rede. Os pesos (const) viram ROMs em BRAM/LUTRAM. */
#include "mlp_nucleo_q8.h"
#include "../gerado/mlp_pesos_q8.h"

#if (MLP_N_IN != MLP_NUCLEO_N_IN) || (CFG_N_OUT != MLP_NUCLEO_N_CFG)
#error "dimensões do cabeçalho gerado não batem com mlp_nucleo_q8.h"
#endif

static int8_t requant(int32_t acc, int32_t m)
{
#pragma HLS INLINE
    if (acc < 0) acc = 0;
    int64_t t = ((int64_t)acc * (int64_t)m + ((int64_t)1 << (MLP_Q_SHIFT - 1))) >> MLP_Q_SHIFT;
    return (int8_t)(t > 127 ? 127 : t);
}

/* Camada densa com ReLU + requantização. Macro para que cada chamada tenha
 * dimensões e ROMs próprias (sem ponteiros para arrays distintos em HLS).
 * O bloco { } isola as variáveis de laço de cada camada. */
#define CAMADA_OCULTA(ROT, NIN, NOUT, X, W, B, M, Y)                          \
    {                                                                         \
    ROT##_o: for (int o = 0; o < (NOUT); o++) {                               \
        int32_t acc = (B)[o];                                                 \
        ROT##_i: for (int i = 0; i < (NIN); i++) {                            \
            _Pragma("HLS PIPELINE II=1")                                      \
            acc += (int32_t)(X)[i] * (int32_t)(W)[i * (NOUT) + o];            \
        }                                                                     \
        (Y)[o] = requant(acc, (M));                                           \
    }                                                                         \
    }

#define CAMADA_SAIDA(ROT, NIN, NOUT, X, W, B, Y)                              \
    {                                                                         \
    ROT##_o: for (int o = 0; o < (NOUT); o++) {                               \
        int32_t acc = (B)[o];                                                 \
        ROT##_i: for (int i = 0; i < (NIN); i++) {                            \
            _Pragma("HLS PIPELINE II=1")                                      \
            acc += (int32_t)(X)[i] * (int32_t)(W)[i * (NOUT) + o];            \
        }                                                                     \
        (Y)[o] = acc;                                                         \
    }                                                                         \
    }

void mlp_nucleo_q8(const int8_t xq[MLP_NUCLEO_N_IN], int32_t acc_freq[1],
                   int32_t acc_cfg[MLP_NUCLEO_N_CFG], int32_t classe[1])
{
#pragma HLS INTERFACE mode=s_axilite port=xq
#pragma HLS INTERFACE mode=s_axilite port=acc_freq
#pragma HLS INTERFACE mode=s_axilite port=acc_cfg
#pragma HLS INTERFACE mode=s_axilite port=classe
#pragma HLS INTERFACE mode=s_axilite port=return

    int8_t x[MLP_N_IN];
    int8_t h1[MLP_N_H1], h2[MLP_N_H2];
    int8_t g1[MLP_N_H1], g2[MLP_N_H2];
    int32_t cfg[CFG_N_OUT];

    copia: for (int i = 0; i < MLP_N_IN; i++) {
#pragma HLS PIPELINE II=1
        x[i] = xq[i];
    }

    /* regressor de frequência */
    CAMADA_OCULTA(f1, MLP_N_IN, MLP_N_H1, x,  FREQ_QW0, FREQ_QB0, FREQ_QM0, h1)
    CAMADA_OCULTA(f2, MLP_N_H1, MLP_N_H2, h1, FREQ_QW1, FREQ_QB1, FREQ_QM1, h2)
    CAMADA_SAIDA (f3, MLP_N_H2, 1,        h2, FREQ_QW2, FREQ_QB2, acc_freq)

    /* classificador de configuração de válvulas */
    CAMADA_OCULTA(c1, MLP_N_IN, MLP_N_H1, x,  CFG_QW0, CFG_QB0, CFG_QM0, g1)
    CAMADA_OCULTA(c2, MLP_N_H1, MLP_N_H2, g1, CFG_QW1, CFG_QB1, CFG_QM1, g2)
    CAMADA_SAIDA (c3, MLP_N_H2, CFG_N_OUT, g2, CFG_QW2, CFG_QB2, cfg)

    int32_t melhor = 0;
    int32_t vmax = cfg[0];
    argmax: for (int k = 0; k < CFG_N_OUT; k++) {
#pragma HLS PIPELINE II=1
        acc_cfg[k] = cfg[k];
        if (cfg[k] > vmax) { vmax = cfg[k]; melhor = k; }
    }
    classe[0] = melhor;
}
