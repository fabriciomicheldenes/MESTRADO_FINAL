/* Benchmark das MLPs na MicroBlaze (Vitis, aplicação standalone).
 *
 * Adicionar ao projeto de aplicação: este arquivo, ../src/mlp_inf.c, ../src/mlp_nucleo_q8.c
 * e o include path ../src e ../gerado. Opções recomendadas do compilador:
 *   -O2 -ffp-contract=off   (+ -mhard-float/-mxl-float-* se a MicroBlaze tiver FPU)
 * Requer um AXI Timer (XPAR_TMRCTR_0_DEVICE_ID) no bloco do Vivado; sem ele, defina
 * MB_SEM_TIMER para só verificar a correção.
 *
 * Para cada uma das 16 janelas de gerado/vetores_mb.h: roda float e int8, confere com as
 * saídas esperadas do Python e mede ciclos do processador. */
#include <stdio.h>
#include "xil_printf.h"
#include "xparameters.h"
#include "mlp_inf.h"
#include "vetores_mb.h"

#ifndef MB_SEM_TIMER
#include "xtmrctr.h"
static XTmrCtr tmr;
#define T_INICIA()  XTmrCtr_Reset(&tmr, 0), XTmrCtr_Start(&tmr, 0)
#define T_LE()      (XTmrCtr_Stop(&tmr, 0), XTmrCtr_GetValue(&tmr, 0))
#else
#define T_INICIA()  ((void)0)
#define T_LE()      (0u)
#endif

int main(void)
{
#ifndef MB_SEM_TIMER
    XTmrCtr_Initialize(&tmr, XPAR_TMRCTR_0_DEVICE_ID);
#endif
    unsigned c_f32 = 0, c_q8 = 0, c_pre = 0;
    int erros = 0;

    for (int j = 0; j < MB_N; j++) {
        const float *x = &MB_X[j * MLP_N_ENTRADAS];
        float logits[MLP_N_CLASSES];

        T_INICIA();
        volatile float fr = mlp_freq_f32(x);
        int kf = mlp_cfg_f32(x, logits);
        c_f32 += T_LE();
        (void)fr;

        int8_t xq[MLP_N_ENTRADAS];
        int32_t acc_f[1], acc_c[MLP_N_CLASSES], cl[1];
        T_INICIA();
        mlp_quantiza_entrada(x, xq);
        c_pre += T_LE();
        T_INICIA();
        mlp_nucleo_q8(xq, acc_f, acc_c, cl);
        c_q8 += T_LE();

        if (kf != MB_CLASSE_F32[j] || cl[0] != MB_CLASSE_Q8[j] || acc_f[0] != MB_ACC_FREQ[j])
            erros++;
        if (j < 4) {
            int hz10 = (int)(mlp_freq_de_acc(acc_f[0]) * 10.0f + 0.5f);
            xil_printf("janela %d: freq %d.%d Hz, classe %s\r\n", j, hz10 / 10, hz10 % 10,
                       mlp_nome_classe(cl[0]));
        }
    }
    xil_printf("MLP: %d janelas, %d divergencias\r\n", MB_N, erros);
    xil_printf("ciclos medios por janela (2 redes): float %u | int8 pre %u + nucleo %u\r\n",
               c_f32 / MB_N, c_pre / MB_N, c_q8 / MB_N);
    return 0;
}
