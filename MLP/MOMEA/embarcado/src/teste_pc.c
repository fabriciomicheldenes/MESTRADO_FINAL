/* Teste de equivalência no PC: C (float e int8) x referência do Python, em todas as
 * janelas de gerado/vetores_teste.bin.
 *   int8 : acumuladores e classe devem ser IDÊNTICOS (bit-exato).
 *   float: frequência dentro de 0,01 Hz e classe igual (exceto empate numérico).
 * Uso: teste_pc ../gerado/vetores_teste.bin */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include "mlp_inf.h"

int main(int argc, char **argv)
{
    const char *arq = argc > 1 ? argv[1] : "../gerado/vetores_teste.bin";
    FILE *f = fopen(arq, "rb");
    if (!f) { perror(arq); return 2; }
    int32_t cab[3];
    if (fread(cab, 4, 3, f) != 3) return 2;
    int n = cab[0], nin = cab[1], ncl = cab[2];
    if (nin != MLP_N_ENTRADAS || ncl != MLP_N_CLASSES) {
        fprintf(stderr, "dimensões inesperadas %d/%d\n", nin, ncl);
        return 2;
    }
    int err_q8_acc = 0, err_q8_cls = 0, err_f32_cls = 0, err_xq = 0;
    double err_f32_max = 0.0, err_q8hz_max = 0.0;
    double t_f32 = 0.0, t_q8 = 0.0;

    for (int j = 0; j < n; j++) {
        float x[MLP_N_ENTRADAS], ref_freq;
        int32_t ref_cls_f32, ref_acc_f, ref_cls_q8, ref_acc_c[MLP_N_CLASSES];
        if (fread(x, 4, nin, f) != (size_t)nin) return 2;
        if (fread(&ref_freq, 4, 1, f) != 1) return 2;
        if (fread(&ref_cls_f32, 4, 1, f) != 1) return 2;
        if (fread(&ref_acc_f, 4, 1, f) != 1) return 2;
        if (fread(&ref_cls_q8, 4, 1, f) != 1) return 2;
        if (fread(ref_acc_c, 4, ncl, f) != (size_t)ncl) return 2;

        /* float */
        clock_t c0 = clock();
        float logits[MLP_N_CLASSES];
        float fr = mlp_freq_f32(x);
        int kf = mlp_cfg_f32(x, logits);
        t_f32 += (double)(clock() - c0);
        double e = fabs((double)fr - ref_freq);
        if (e > err_f32_max) err_f32_max = e;
        if (kf != ref_cls_f32) err_f32_cls++;

        /* int8 */
        c0 = clock();
        int8_t xq[MLP_N_ENTRADAS];
        int32_t acc_f[1], acc_c[MLP_N_CLASSES], cl[1];
        mlp_quantiza_entrada(x, xq);
        mlp_nucleo_q8(xq, acc_f, acc_c, cl);
        t_q8 += (double)(clock() - c0);
        int dif = (acc_f[0] != ref_acc_f);
        for (int k = 0; k < ncl; k++) dif |= (acc_c[k] != ref_acc_c[k]);
        if (dif) err_q8_acc++;
        if (cl[0] != ref_cls_q8) err_q8_cls++;
        double hz = mlp_freq_de_acc(acc_f[0]);
        double e2 = fabs(hz - fr);
        if (e2 > err_q8hz_max) err_q8hz_max = e2;
        (void)err_xq;
    }
    fclose(f);

    printf("Janelas testadas: %d\n", n);
    printf("int8 : janelas com acumulador diferente do Python: %d ; classe diferente: %d\n",
           err_q8_acc, err_q8_cls);
    printf("float: |freq C - Python| max = %.2e Hz ; classe diferente: %d\n", err_f32_max, err_f32_cls);
    printf("int8 x float (C): |freq| max = %.3f Hz\n", err_q8hz_max);
    printf("Tempo medio no PC: float %.2f us, int8 %.2f us por janela (as duas redes)\n",
           1e6 * t_f32 / CLOCKS_PER_SEC / n, 1e6 * t_q8 / CLOCKS_PER_SEC / n);
    int ok = (err_q8_acc == 0 && err_q8_cls == 0 && err_f32_max < 1e-2 && err_f32_cls == 0);
    printf("%s\n", ok ? "RESULTADO: OK" : "RESULTADO: FALHOU");
    return ok ? 0 : 1;
}
