/* Valida a cadeia embarcada completa no PC:
 *   hits gravados -> ea_descritores (C) -> MLP int8 (C)
 * contra resultados/janelas_descritores.csv e as saídas int8 do Python.
 *
 * Uso: teste_descritores ../gerado/hits_teste.bin ../gerado/vetores_teste.bin
 * Compilar duas vezes: modo exato (padrão) e -DEA_DESCR_HISTOGRAMA. */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include "ea_descritores.h"
#include "mlp_inf.h"
#include "../gerado/descritores_cfg.h"

#pragma pack(push, 1)
typedef struct {
    double rx_time_s;
    uint8_t canal, sat, maxdur, rovl;
    uint32_t counts, peak, rise, duration;
    uint64_t energy;
    float peak_v, amp_dbae, energy_eu, rise_us, duration_us;
} hit_arq_t;
#pragma pack(pop)

static const char *NOMES[EA_N_DESCR_CANAL] = {
    "taxa_hits", "frac_sat", "frac_maxdur", "log_energia_s", "fm_kHz_med", "amp_med",
    "amp_p10", "amp_p50", "amp_p90", "log_energia_p50", "log_energia_p90", "log_dur_p50",
    "log_dur_p90", "log_cont_p50", "log_cont_p90", "log_subida_p50", "fm_kHz_normais",
    "log_RA_p50"};

typedef struct {
    double err_max[EA_N_DESCR];
    int nan_dif, fora_tol, janelas;
    int cls_igual, nivel_igual;
    double hz_dif_max;
    double t_add, t_fecha, t_mlp;
    uint32_t hits;
} estat_t;

static ea_janela_t JAN;   /* estático, como no firmware */

static int ler(void *p, size_t sz, size_t n, FILE *f)
{
    return fread(p, sz, n, f) == n;
}

int main(int argc, char **argv)
{
    const char *a_hits = argc > 1 ? argv[1] : "../gerado/hits_teste.bin";
    const char *a_vet = argc > 2 ? argv[2] : "../gerado/vetores_teste.bin";
    /* opcional: grava, por janela, os 36 descritores do C (via registro bruto), a classe e o Hz */
    FILE *fs = argc > 3 ? fopen(argv[3], "wb") : NULL;
    FILE *fh = fopen(a_hits, "rb"), *fv = fopen(a_vet, "rb");
    if (!fh || !fv) { fprintf(stderr, "não abriu %s / %s\n", a_hits, a_vet); return 2; }

    char mag[4];
    int32_t n_runs, n_jan_tot;
    float thr_a, thr_b, dur;
    if (!ler(mag, 1, 4, fh) || memcmp(mag, "EAH1", 4)) { fprintf(stderr, "formato\n"); return 2; }
    if (!ler(&n_runs, 4, 1, fh) || !ler(&n_jan_tot, 4, 1, fh) || !ler(&thr_a, 4, 1, fh) ||
        !ler(&thr_b, 4, 1, fh) || !ler(&dur, 4, 1, fh)) return 2;
    int32_t cab[3];
    if (!ler(cab, 4, 3, fv) || cab[0] != n_jan_tot) { fprintf(stderr, "vetores_teste\n"); return 2; }

    /* 0: unidades físicas do hits.csv ; 1: conversão do registro bruto em C */
    estat_t E[2];
    memset(E, 0, sizeof E);

    for (int r = 0; r < n_runs; r++) {
        int32_t n_hits, n_jan;
        float cal4[4];
        if (!ler(&n_hits, 4, 1, fh) || !ler(&n_jan, 4, 1, fh) || !ler(cal4, 4, 4, fh)) return 2;
        ea_calib_t cal[2];
        ea_calib_prepara(&cal[0], cal4[0], cal4[2]);
        ea_calib_prepara(&cal[1], cal4[1], cal4[3]);
        hit_arq_t *H = malloc(sizeof(hit_arq_t) * (size_t)(n_hits ? n_hits : 1));
        float *esp = malloc(sizeof(float) * EA_N_DESCR * (size_t)n_jan);
        if (!ler(H, sizeof(hit_arq_t), (size_t)n_hits, fh) ||
            !ler(esp, sizeof(float), (size_t)EA_N_DESCR * n_jan, fh)) return 2;

        for (int k = 0; k < n_jan; k++) {
            float xin[EA_N_DESCR], ref_freq;
            int32_t ref_cf, ref_acc_f, ref_cq, ref_acc_c[MLP_N_CLASSES];
            if (!ler(xin, 4, EA_N_DESCR, fv) || !ler(&ref_freq, 4, 1, fv) || !ler(&ref_cf, 4, 1, fv) ||
                !ler(&ref_acc_f, 4, 1, fv) || !ler(&ref_cq, 4, 1, fv) ||
                !ler(ref_acc_c, 4, MLP_N_CLASSES, fv)) return 2;
            float ref_hz = mlp_freq_de_acc(ref_acc_f);

            for (int via = 0; via < 2; via++) {
                estat_t *e = &E[via];
                clock_t c0 = clock();
                ea_janela_inicia(&JAN, thr_a, thr_b);
                for (int i = 0; i < n_hits; i++) {
                    if ((int)floor(H[i].rx_time_s / dur) != k) continue;
                    ea_hit_t h;
                    if (via == 0) {
                        h.canal = H[i].canal; h.sat = H[i].sat; h.maxdur = H[i].maxdur;
                        h.counts = H[i].counts; h.peak_v = H[i].peak_v; h.amp_dbae = H[i].amp_dbae;
                        h.energy_eu = H[i].energy_eu; h.rise_us = H[i].rise_us;
                        h.duration_us = H[i].duration_us;
                    } else {
                        ea_hit_de_registro(H[i].canal, H[i].peak, H[i].counts, H[i].rise,
                                           H[i].duration, H[i].energy, H[i].sat, H[i].maxdur,
                                           &cal[H[i].canal & 1], &h);
                    }
                    e->hits += (uint32_t)ea_janela_adiciona(&JAN, &h);
                }
                clock_t c1 = clock();
                float x[EA_N_DESCR];
                ea_janela_fecha(&JAN, dur, x);
                clock_t c2 = clock();
                float fr; int cl; int32_t acc[MLP_N_CLASSES];
                mlp_q8(x, &fr, &cl, acc);
                clock_t c3 = clock();
                e->t_add += (double)(c1 - c0); e->t_fecha += (double)(c2 - c1);
                e->t_mlp += (double)(c3 - c2);
                if (fs && via == 1) {
                    int32_t cl32 = cl;
                    fwrite(x, sizeof(float), EA_N_DESCR, fs);
                    fwrite(&cl32, 4, 1, fs);
                    fwrite(&fr, 4, 1, fs);
                }

                const float *ex = &esp[k * EA_N_DESCR];
                int fora = 0;
                for (int d = 0; d < EA_N_DESCR; d++) {
                    int na = isnan(x[d]), nb = isnan(ex[d]);
                    if (na || nb) { if (na != nb) { e->nan_dif++; fora = 1; } continue; }
                    double err = fabs((double)x[d] - ex[d]);
                    if (err > e->err_max[d]) e->err_max[d] = err;
                    if (err > 1e-3 * (1.0 + fabs(ex[d]))) fora = 1;
                }
                e->fora_tol += fora;
                e->janelas++;
                e->cls_igual += (cl == ref_cq);
                e->nivel_igual += (lroundf(fr / 10.0f) == lroundf(ref_hz / 10.0f));
                double dh = fabs((double)fr - ref_hz);
                if (dh > e->hz_dif_max) e->hz_dif_max = dh;
            }
        }
        free(H); free(esp);
    }
    fclose(fh); fclose(fv);
    if (fs) fclose(fs);

#ifdef EA_DESCR_HISTOGRAMA
    const char *modo = "HISTOGRAMA";
#else
    const char *modo = "EXATO";
#endif
    printf("Modo de percentis: %s ; corridas %d ; janelas %d ; sizeof(ea_janela_t) = %u bytes\n",
           modo, n_runs, n_jan_tot, (unsigned)sizeof(ea_janela_t));
    const char *via_nome[2] = {"unidades do hits.csv", "registro bruto + calibração (C)"};
    int ok = 1;
    for (int via = 0; via < 2; via++) {
        estat_t *e = &E[via];
        printf("\n== Entrada: %s (%u hits aceitos)\n", via_nome[via], e->hits);
        printf("  descritores: janelas fora da tolerância (1e-3*(1+|ref|)) = %d ; NaN divergentes = %d\n",
               e->fora_tol, e->nan_dif);
        printf("  erro absoluto máximo por descritor (A | B):\n");
        for (int d = 0; d < EA_N_DESCR_CANAL; d++)
            printf("    %-16s %10.3g | %10.3g\n", NOMES[d], e->err_max[d],
                   e->err_max[d + EA_N_DESCR_CANAL]);
        printf("  MLP int8 sobre descritores do C x Python: classe igual %.2f %% ; "
               "nível de 10 Hz igual %.2f %% ; |dHz| máx %.3f\n",
               100.0 * e->cls_igual / e->janelas, 100.0 * e->nivel_igual / e->janelas, e->hz_dif_max);
        printf("  tempo médio no PC por janela: adicionar %.1f us, fechar %.1f us, MLP %.1f us\n",
               1e6 * e->t_add / CLOCKS_PER_SEC / e->janelas, 1e6 * e->t_fecha / CLOCKS_PER_SEC / e->janelas,
               1e6 * e->t_mlp / CLOCKS_PER_SEC / e->janelas);
#ifndef EA_DESCR_HISTOGRAMA
        if (via == 0 && (e->nan_dif || e->fora_tol)) ok = 0;
#endif
    }
    printf("\n%s\n", ok ? "RESULTADO: OK" : "RESULTADO: VERIFICAR");
    return ok ? 0 : 1;
}
