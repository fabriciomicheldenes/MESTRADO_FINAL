/* Testbench HLS do núcleo int8: 256 janelas com entradas já quantizadas e acumuladores
 * esperados gerados pelo Python (gerado/vetores_tb.h). Usado em csim e em cosim (RTL).
 * Retorna 0 somente se todas as saídas forem idênticas. */
#include <stdio.h>
#include "../src/mlp_nucleo_q8.h"
#include "../gerado/vetores_tb.h"

int main(void)
{
    int erros = 0;
    for (int j = 0; j < TB_N; j++) {
        int8_t xq[MLP_NUCLEO_N_IN];
        int32_t acc_f[1], acc_c[MLP_NUCLEO_N_CFG], cl[1];
        for (int i = 0; i < MLP_NUCLEO_N_IN; i++) xq[i] = TB_XQ[j * MLP_NUCLEO_N_IN + i];

        mlp_nucleo_q8(xq, acc_f, acc_c, cl);

        int dif = (acc_f[0] != TB_ACC_FREQ[j]);
        int esperado = 0;
        for (int k = 0; k < MLP_NUCLEO_N_CFG; k++) {
            int32_t r = TB_ACC_CFG[j * MLP_NUCLEO_N_CFG + k];
            dif |= (acc_c[k] != r);
            if (r > TB_ACC_CFG[j * MLP_NUCLEO_N_CFG + esperado]) esperado = k;
        }
        dif |= (cl[0] != esperado);
        if (dif) {
            if (erros < 10)
                printf("janela %d: acc_freq %d (esperado %d), classe %d (esperado %d)\n",
                       j, (int)acc_f[0], (int)TB_ACC_FREQ[j], (int)cl[0], esperado);
            erros++;
        }
    }
    printf("TB: %d janelas, %d com divergencia -> %s\n", TB_N, erros, erros ? "FALHOU" : "OK");
    return erros ? 1 : 0;
}
