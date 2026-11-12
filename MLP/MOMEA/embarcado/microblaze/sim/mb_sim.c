/* Programa para medição de ciclos na MicroBlaze por SIMULAÇÃO RTL (xsim), sem BSP.
 *
 * Cada etapa escreve uma marca num AXI GPIO de 32 bits (endereço GPIO_BASE); o testbench
 * (tb_mb.v) registra o ciclo de clock em que cada marca aparece. As diferenças entre marcas
 * dão os ciclos de cada etapa, já incluindo a escrita no GPIO (~poucos ciclos).
 *
 * Variantes (compilação):
 *   (padrão)       cadeia completa: 4 janelas reais (gerado/hits_mb.h) -> descritores -> MLP int8
 *   -DSIM_MLP      só a MLP: 16 janelas (gerado/vetores_mb.h), caminho float32 e int8
 *   -DHOST         compila no PC e imprime as marcas (para conferir os valores com a simulação)
 *
 * Marcas (byte alto = código):
 *   cadeia: 01|w início hits ; 02|w fim hits ; 03|w fim fechamento ; 04|w fim MLP
 *           05|cl<<16|div resultado (classe, descritores divergentes) ; 06|acc_freq (24 bits)
 *           0F|div_total fim
 *   MLP   : 11|j início ; 12|j fim float ; 13|j fim quantização ; 14|j fim núcleo int8
 *           15|cf<<8|cq classes ; 1F|erros fim */
#include <stdint.h>
#include <math.h>
#include "ea_descritores.h"
#include "mlp_inf.h"
#include "descritores_cfg.h"

#ifdef HOST
#include <stdio.h>
#define MARCA(v) printf("%08x\n", (unsigned)(v))
#define FIM_PROGRAMA() return 0
#else
#define GPIO_BASE 0x40000000u
#define MARCA(v) (*(volatile uint32_t *)GPIO_BASE = (uint32_t)(v))
#define FIM_PROGRAMA() for (;;) { }
#endif

/* Carga reduzida para a simulação RTL (lenta): os custos por hit e por inferência são
 * praticamente determinísticos, então poucas amostras bastam para medi-los. */
#ifndef SIM_N_JAN
#define SIM_N_JAN 2      /* janelas reais processadas (de 4 em hits_mb.h) */
#endif
#ifndef SIM_N_MLP
#define SIM_N_MLP 2      /* vetores da MLP (de 16 em vetores_mb.h) */
#endif

#ifndef SIM_MLP
#include "hits_mb.h"
static ea_janela_t jan;

int main(void)
{
    uint32_t div_total = 0;
    MARCA(0x00000001u);
    for (uint32_t w = 0; w < SIM_N_JAN && w < HMB_N_JAN; w++) {
        ea_calib_t cal[2];
        ea_calib_prepara(&cal[0], HMB_CAL[w][0], HMB_CAL[w][2]);
        ea_calib_prepara(&cal[1], HMB_CAL[w][1], HMB_CAL[w][3]);
        ea_janela_inicia(&jan, EA_LIMIAR_A_DBAE, EA_LIMIAR_B_DBAE);

        MARCA(0x01000000u | w);
        for (uint32_t i = HMB_INICIO[w]; i < HMB_INICIO[w + 1]; i++) {
            const hit_mb_t *r = &HMB_HITS[i];
            ea_hit_t h;
            ea_hit_de_registro(r->canal, r->pico, r->contagens, r->subida, r->duracao,
                               r->energia, r->sat, r->maxdur, &cal[r->canal & 1], &h);
            ea_janela_adiciona(&jan, &h);
        }
        MARCA(0x02000000u | w);
        float x[EA_N_DESCR];
        ea_janela_fecha(&jan, EA_JANELA_S, x);
        MARCA(0x03000000u | w);
        float hz; int cl; int32_t acc[MLP_N_CLASSES];
        int8_t xq[MLP_N_ENTRADAS];
        int32_t acc_f[1], clq[1];
        mlp_quantiza_entrada(x, xq);
        mlp_nucleo_q8(xq, acc_f, acc, clq);
        MARCA(0x04000000u | w);
        hz = mlp_freq_de_acc(acc_f[0]);
        cl = (int)clq[0];
        (void)hz;

        uint32_t div = 0;
        for (int d = 0; d < EA_N_DESCR; d++) {
            float e = HMB_X_ESPERADO[w][d];
            if (isnan(e) != isnan(x[d])) { div++; continue; }
            if (!isnan(e) && fabsf(x[d] - e) > 1e-3f * (1.0f + fabsf(e))) div++;
        }
        div_total += div;
        MARCA(0x05000000u | ((uint32_t)cl << 16) | (div & 0xFFFFu));
        MARCA(0x06000000u | ((uint32_t)acc_f[0] & 0x00FFFFFFu));
    }
    MARCA(0x0F000000u | div_total);
    FIM_PROGRAMA();
}

#else  /* ------------------------- só MLP ------------------------- */
#include "vetores_mb.h"

int main(void)
{
    uint32_t erros = 0;
    MARCA(0x00000001u);
    for (uint32_t j = 0; j < SIM_N_MLP && j < MB_N; j++) {
        const float *x = &MB_X[j * MLP_N_ENTRADAS];
        float logits[MLP_N_CLASSES];
        MARCA(0x11000000u | j);
        volatile float fr = mlp_freq_f32(x);
        int cf = mlp_cfg_f32(x, logits);
        (void)fr;
        MARCA(0x12000000u | j);
        int8_t xq[MLP_N_ENTRADAS];
        int32_t acc_f[1], acc_c[MLP_N_CLASSES], cl[1];
        mlp_quantiza_entrada(x, xq);
        MARCA(0x13000000u | j);
        mlp_nucleo_q8(xq, acc_f, acc_c, cl);
        MARCA(0x14000000u | j);
        if (cf != MB_CLASSE_F32[j] || cl[0] != MB_CLASSE_Q8[j] || acc_f[0] != MB_ACC_FREQ[j])
            erros++;
        MARCA(0x15000000u | ((uint32_t)cf << 8) | (uint32_t)cl[0]);
    }
    MARCA(0x1F000000u | erros);
    FIM_PROGRAMA();
}
#endif
