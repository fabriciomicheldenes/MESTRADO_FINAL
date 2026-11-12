/* Cadeia embarcada completa na MicroBlaze: registros de hit -> descritores -> MLP int8.
 *
 * Projeto Vitis (standalone): este arquivo + ../src/ea_descritores.c + ../src/mlp_inf.c +
 * ../src/mlp_nucleo_q8.c ; include paths ../src e ../gerado ; -O2 -ffp-contract=off.
 * Opcional: -DEA_DESCR_HISTOGRAMA (percentis por histograma, ~10 kB em vez de ~64 kB).
 * Usa o AXI Timer 0 para medir ciclos; sem ele, compile com -DMB_SEM_TIMER.
 *
 * Usa 4 janelas reais (gerado/hits_mb.h): converte cada registro bruto com a calibração da
 * corrida, acumula, fecha a janela, confere os 36 descritores com o Python e roda a MLP.
 *
 * Integração no firmware real (esboço):
 *   - no tratamento de cada registro de hit (antes/depois do envio UDP):
 *         ea_hit_de_registro(...campos do registro...); ea_janela_adiciona(&jan, &h);
 *   - num temporizador de EA_JANELA_S segundos:
 *         ea_janela_fecha(&jan, EA_JANELA_S, x); mlp_q8(x, &hz, &classe, acc);
 *         enviar {hz, classe, acc} num datagrama de status. */
#include <math.h>
#include "xil_printf.h"
#include "xparameters.h"
#include "ea_descritores.h"
#include "mlp_inf.h"
#include "descritores_cfg.h"
#include "hits_mb.h"

#ifndef MB_SEM_TIMER
#include "xtmrctr.h"
static XTmrCtr tmr;
#define T_INICIA()  XTmrCtr_Reset(&tmr, 0), XTmrCtr_Start(&tmr, 0)
#define T_LE()      (XTmrCtr_Stop(&tmr, 0), XTmrCtr_GetValue(&tmr, 0))
#else
#define T_INICIA()  ((void)0)
#define T_LE()      (0u)
#endif

static ea_janela_t jan;   /* ~64 kB (exato) ou ~10 kB (histograma): estático */

int main(void)
{
#ifndef MB_SEM_TIMER
    XTmrCtr_Initialize(&tmr, XPAR_TMRCTR_0_DEVICE_ID);
#endif
    unsigned c_add = 0, c_fecha = 0, c_mlp = 0, n_hits = 0;
    int div_descr = 0;

    for (int w = 0; w < HMB_N_JAN; w++) {
        ea_calib_t cal[2];
        ea_calib_prepara(&cal[0], HMB_CAL[w][0], HMB_CAL[w][2]);
        ea_calib_prepara(&cal[1], HMB_CAL[w][1], HMB_CAL[w][3]);
        ea_janela_inicia(&jan, EA_LIMIAR_A_DBAE, EA_LIMIAR_B_DBAE);

        for (uint32_t i = HMB_INICIO[w]; i < HMB_INICIO[w + 1]; i++) {
            const hit_mb_t *r = &HMB_HITS[i];
            ea_hit_t h;
            T_INICIA();
            ea_hit_de_registro(r->canal, r->pico, r->contagens, r->subida, r->duracao,
                               r->energia, r->sat, r->maxdur, &cal[r->canal & 1], &h);
            ea_janela_adiciona(&jan, &h);
            c_add += T_LE();
            n_hits++;
        }

        float x[EA_N_DESCR];
        T_INICIA();
        ea_janela_fecha(&jan, EA_JANELA_S, x);
        c_fecha += T_LE();

        for (int d = 0; d < EA_N_DESCR; d++) {
            float e = HMB_X_ESPERADO[w][d];
            if (isnan(e) != isnan(x[d])) { div_descr++; continue; }
            if (!isnan(e) && fabsf(x[d] - e) > 1e-3f * (1.0f + fabsf(e))) div_descr++;
        }

        float hz; int cl; int32_t acc[MLP_N_CLASSES];
        T_INICIA();
        mlp_q8(x, &hz, &cl, acc);
        c_mlp += T_LE();

        int hz10 = (int)(hz * 10.0f + 0.5f);
        xil_printf("janela %d: %d hits, freq %d.%d Hz, classe %s\r\n", w,
                   (int)(HMB_INICIO[w + 1] - HMB_INICIO[w]), hz10 / 10, hz10 % 10,
                   mlp_nome_classe(cl));
    }
    xil_printf("descritores divergentes do Python (tol. 1e-3): %d (esperado 0 no modo exato)\r\n",
               div_descr);
    xil_printf("ciclos: por hit %u | fechar janela %u | MLP int8 %u\r\n",
               c_add / (n_hits ? n_hits : 1), c_fecha / HMB_N_JAN, c_mlp / HMB_N_JAN);
    return 0;
}
