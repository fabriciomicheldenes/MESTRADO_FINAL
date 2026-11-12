/* Inferência das MLPs de EA - ver mlp_inf.h.
 * Compilar com -ffp-contract=off para que o pré-processamento int8 seja bit-exato
 * em relação à emulação de exportar_modelo.py. */
#include <math.h>
#include "mlp_inf.h"
#include "../gerado/mlp_pesos_q8.h"
#include "../gerado/mlp_classes.h"

/* -DMLP_SEM_F32 remove o caminho float32 e seus pesos (~37 kB) quando só o int8 é usado */
#ifndef MLP_SEM_F32
#include "../gerado/mlp_pesos_f32.h"

/* y[o] = b[o] + sum_i x[i]*W[i][o], com ReLU opcional */
static void densa_f32(const float *x, int nin, const float *W, const float *b,
                      int nout, float *y, int relu)
{
    for (int o = 0; o < nout; o++) {
        float acc = b[o];
        for (int i = 0; i < nin; i++)
            acc += x[i] * W[i * nout + o];
        y[o] = (relu && acc < 0.0f) ? 0.0f : acc;
    }
}

static void imputa(const float *x, const float *med, float *v)
{
    for (int i = 0; i < MLP_N_IN; i++)
        v[i] = isnan(x[i]) ? med[i] : x[i];
}

float mlp_freq_f32(const float x[MLP_N_ENTRADAS])
{
    float v[MLP_N_IN], h1[MLP_N_H1], h2[MLP_N_H2], y;
    imputa(x, FREQ_MEDIANA, v);
    densa_f32(v,  MLP_N_IN, FREQ_W0, FREQ_B0, MLP_N_H1, h1, 1);
    densa_f32(h1, MLP_N_H1, FREQ_W1, FREQ_B1, MLP_N_H2, h2, 1);
    densa_f32(h2, MLP_N_H2, FREQ_W2, FREQ_B2, 1, &y, 0);
    return y;  /* padronização da entrada e desnormalização do alvo já dobradas nos pesos */
}

int mlp_cfg_f32(const float x[MLP_N_ENTRADAS], float logits[MLP_N_CLASSES])
{
    float v[MLP_N_IN], h1[MLP_N_H1], h2[MLP_N_H2];
    imputa(x, CFG_MEDIANA, v);
    densa_f32(v,  MLP_N_IN, CFG_W0, CFG_B0, MLP_N_H1, h1, 1);
    densa_f32(h1, MLP_N_H1, CFG_W1, CFG_B1, MLP_N_H2, h2, 1);
    densa_f32(h2, MLP_N_H2, CFG_W2, CFG_B2, CFG_N_OUT, logits, 0);
    int k = 0;
    for (int c = 1; c < CFG_N_OUT; c++)
        if (logits[c] > logits[k]) k = c;
    return k;
}
#endif /* MLP_SEM_F32 */

void mlp_quantiza_entrada(const float x[MLP_N_ENTRADAS], int8_t xq[MLP_N_ENTRADAS])
{
    for (int i = 0; i < MLP_N_IN; i++) {
        float v = isnan(x[i]) ? Q_MEDIANA[i] : x[i];
        float z = (v - Q_MEDIA[i]) * Q_INV_ESCALA[i];
        if (z > MLP_Q_CLIP) z = MLP_Q_CLIP;
        if (z < -MLP_Q_CLIP) z = -MLP_Q_CLIP;
        float q = rintf(z * Q_INV_S_IN);          /* metade para par, como np.rint */
        if (q > 127.0f) q = 127.0f;
        if (q < -127.0f) q = -127.0f;
        xq[i] = (int8_t)q;
    }
}

float mlp_freq_de_acc(int32_t acc_freq)
{
    return (float)acc_freq * FREQ_Q_S_OUT * FREQ_Y_ESCALA + FREQ_Y_MEDIA;
}

void mlp_q8(const float x[MLP_N_ENTRADAS], float *freq_hz, int *classe,
            int32_t acc_cfg[MLP_N_CLASSES])
{
    int8_t xq[MLP_N_IN];
    int32_t acc_f[1], cl[1];
    mlp_quantiza_entrada(x, xq);
    mlp_nucleo_q8(xq, acc_f, acc_cfg, cl);
    *freq_hz = mlp_freq_de_acc(acc_f[0]);
    *classe = (int)cl[0];
}

const char *mlp_nome_classe(int k)
{
    return (k >= 0 && k < CFG_N_CLASSES) ? CFG_CLASSES[k] : "?";
}
