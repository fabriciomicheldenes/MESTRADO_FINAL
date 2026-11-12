/* Descritores de EA por janela - ver ea_descritores.h.
 * Referência: descritores_canal() em ../../mlp_ea.py. */
#include <math.h>
#include <string.h>
#include "ea_descritores.h"

#define FS_HZ 10.0e6f
#define EU_V2S 1.0e-14f

/* ------------------------------------------------------------------------- */
void ea_calib_prepara(ea_calib_t *cal, float volts_por_codigo, float ganho_db)
{
    float g = powf(10.0f, ganho_db / 20.0f);
    float r = volts_por_codigo / g;
    cal->volts_por_codigo = volts_por_codigo;
    cal->ganho_db = ganho_db;
    cal->fator_amp = 1.0f / (g * 1.0e-6f);
    cal->fator_energia = r * r / FS_HZ / EU_V2S;   /* em etapas para não estourar o float */
}

void ea_hit_de_registro(uint8_t canal, uint32_t pico_codigo, uint32_t contagens,
                        uint32_t subida_amostras, uint32_t duracao_amostras, uint64_t energia,
                        uint8_t sat, uint8_t maxdur, const ea_calib_t *cal, ea_hit_t *h)
{
    h->canal = canal;
    h->sat = sat;
    h->maxdur = maxdur;
    h->counts = contagens;
    h->peak_v = (float)pico_codigo * cal->volts_por_codigo;
    h->amp_dbae = 20.0f * log10f(h->peak_v * cal->fator_amp);
    /* sum(u^2) * vpc^2 / fs / g^2, em eu */
    h->energy_eu = (float)energia * cal->fator_energia;
    h->rise_us = (float)subida_amostras * (1.0e6f / FS_HZ);
    h->duration_us = (float)duracao_amostras * (1.0e6f / FS_HZ);
}

/* ------------------------------------------------------------------------- */
#ifdef EA_DESCR_HISTOGRAMA
/* limite inferior, largura e número de bins por variável (fora da faixa: bin extremo) */
static const float H_LO[EA_N_VAR]  = {0.0f, 40.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, -7.0f};
static const float H_W[EA_N_VAR]   = {2.0f, 0.25f, 0.05f, 0.025f, 0.025f, 0.025f, 2.0f, 0.05f};
static const uint16_t H_NB[EA_N_VAR] = {512, 280, 260, 240, 240, 240, 512, 280};
static uint16_t H_OFF[EA_N_VAR];   /* preenchido em ea_janela_inicia */

static void h_inicia_offsets(void)
{
    uint16_t o = 0;
    for (int v = 0; v < EA_N_VAR; v++) { H_OFF[v] = o; o += H_NB[v]; }
}

static void guarda(ea_canal_acc_t *c, int var, uint32_t idx, float val)
{
    (void)idx;
    int b = (int)floorf((val - H_LO[var]) / H_W[var]);
    if (b < 0) b = 0;
    if (b >= H_NB[var]) b = H_NB[var] - 1;
    uint16_t *p = &c->hist[H_OFF[var] + b];
    if (*p < 0xFFFFu) (*p)++;
}

/* valor do r-ésimo elemento (0-based) supondo distribuição uniforme dentro do bin */
static float h_valor_rank(const ea_canal_acc_t *c, int var, uint32_t r)
{
    const uint16_t *h = &c->hist[H_OFF[var]];
    uint32_t acum = 0;
    for (int b = 0; b < H_NB[var]; b++) {
        if (r < acum + h[b])
            return H_LO[var] + H_W[var] * ((float)b + ((float)(r - acum) + 0.5f) / (float)h[b]);
        acum += h[b];
    }
    return H_LO[var] + H_W[var] * (float)H_NB[var];
}

static float pct(ea_canal_acc_t *c, int var, uint32_t n, float q)
{
    if (n == 0) return NAN;
    float pos = q / 100.0f * (float)(n - 1);
    uint32_t lo = (uint32_t)pos;
    float fr = pos - (float)lo;
    float a = h_valor_rank(c, var, lo);
    if (fr <= 0.0f || lo + 1 >= n) return a;
    float b = h_valor_rank(c, var, lo + 1);
    return a + fr * (b - a);
}

#else  /* ---------------- modo exato ---------------- */
static void guarda(ea_canal_acc_t *c, int var, uint32_t idx, float val)
{
    if (idx < EA_MAX_HITS_CANAL) c->v[var][idx] = val;
}

/* heapsort in-place: O(n log n), sem recursão nem memória extra */
static void desce(float *a, uint32_t i, uint32_t n)
{
    for (;;) {
        uint32_t m = i, l = 2 * i + 1, r = l + 1;
        if (l < n && a[l] > a[m]) m = l;
        if (r < n && a[r] > a[m]) m = r;
        if (m == i) return;
        float t = a[i]; a[i] = a[m]; a[m] = t;
        i = m;
    }
}

static void ordena(float *a, uint32_t n)
{
    if (n < 2) return;
    for (uint32_t i = n / 2; i-- > 0;) desce(a, i, n);
    for (uint32_t k = n - 1; k > 0; k--) {
        float t = a[0]; a[0] = a[k]; a[k] = t;
        desce(a, 0, k);
    }
}

/* np.percentile(method="linear") sobre o vetor já ordenado */
static float pct_ord(const float *a, uint32_t n, float q)
{
    if (n == 0) return NAN;
    float pos = q / 100.0f * (float)(n - 1);
    uint32_t lo = (uint32_t)pos;
    float fr = pos - (float)lo;
    if (lo + 1 >= n) return a[n - 1];
    float d = a[lo + 1] - a[lo];
    return (fr >= 0.5f) ? a[lo + 1] - d * (1.0f - fr) : a[lo] + d * fr;
}

static float pct(ea_canal_acc_t *c, int var, uint32_t n, float q)
{
    return pct_ord(c->v[var], n, q);   /* chamado após ordenar */
}
#endif

/* ------------------------------------------------------------------------- */
static void zera_canal(ea_canal_acc_t *c)
{
    c->n = c->n_sat = c->n_maxdur = c->n_norm = c->n_fora = 0;
    c->soma_energia = c->soma_amp_norm = 0.0f;
#ifdef EA_DESCR_HISTOGRAMA
    memset(c->hist, 0, sizeof c->hist);
#endif
}

void ea_janela_inicia(ea_janela_t *j, float limiar_a_dbae, float limiar_b_dbae)
{
#ifdef EA_DESCR_HISTOGRAMA
    h_inicia_offsets();
#endif
    for (int k = 0; k < EA_N_CANAIS; k++) zera_canal(&j->c[k]);
    j->limiar_dbae[0] = limiar_a_dbae;
    j->limiar_dbae[1] = limiar_b_dbae;
    j->n_abaixo_limiar = 0;
}

static float fm_khz(const ea_hit_t *h)
{
    float d = h->duration_us < 1.0f ? 1.0f : h->duration_us;   /* clip(lower=1) */
    return (float)h->counts / d * 1.0e3f;
}

int ea_janela_adiciona(ea_janela_t *j, const ea_hit_t *h)
{
    if (h->canal >= EA_N_CANAIS) return 0;
    if (!(h->amp_dbae >= j->limiar_dbae[h->canal])) { j->n_abaixo_limiar++; return 0; }
    ea_canal_acc_t *c = &j->c[h->canal];

    float fm = fm_khz(h);
    guarda(c, EA_V_FM_TODOS, c->n, fm);
    c->n++;
    c->n_sat += h->sat ? 1u : 0u;
    c->n_maxdur += h->maxdur ? 1u : 0u;
    c->soma_energia += h->energy_eu;
    if (h->maxdur) return 1;            /* forma: só hits normais (sem MAXDUR) */

    uint32_t i = c->n_norm;
#ifndef EA_DESCR_HISTOGRAMA
    if (i >= EA_MAX_HITS_CANAL) c->n_fora++;
#endif
    float pv = h->peak_v < 1.0e-6f ? 1.0e-6f : h->peak_v;
    float ra = h->rise_us / 1.0e3f / pv;                       /* ms/V */
    guarda(c, EA_V_AMP,      i, h->amp_dbae);
    guarda(c, EA_V_LOG_E,    i, log10f(h->energy_eu + 1.0f));
    guarda(c, EA_V_LOG_DUR,  i, log10f(h->duration_us + 1.0f));
    guarda(c, EA_V_LOG_CONT, i, log10f((float)h->counts + 1.0f));
    guarda(c, EA_V_LOG_SUB,  i, log10f(h->rise_us + 1.0f));
    guarda(c, EA_V_FM,       i, fm);
    guarda(c, EA_V_LOG_RA,   i, log10f(ra + 1.0e-6f));
    c->soma_amp_norm += h->amp_dbae;
    c->n_norm++;
    return 1;
}

static void fecha_canal(ea_canal_acc_t *c, float dur_s, float *x)
{
    uint32_t n = c->n;
#ifdef EA_DESCR_HISTOGRAMA
    uint32_t nn = c->n_norm;
#else
    uint32_t n_t = n < EA_MAX_HITS_CANAL ? n : EA_MAX_HITS_CANAL;
    uint32_t nn = c->n_norm < EA_MAX_HITS_CANAL ? c->n_norm : EA_MAX_HITS_CANAL;
    ordena(c->v[EA_V_FM_TODOS], n_t);
    for (int v = EA_V_AMP; v < EA_N_VAR; v++) ordena(c->v[v], nn);
#endif
    x[0] = (float)n / dur_s;
    x[1] = n ? (float)c->n_sat / (float)n : NAN;
    x[2] = n ? (float)c->n_maxdur / (float)n : NAN;
    x[3] = log10f(c->soma_energia / dur_s + 1.0f);
#ifdef EA_DESCR_HISTOGRAMA
    x[4] = pct(c, EA_V_FM_TODOS, n, 50.0f);
#else
    x[4] = pct(c, EA_V_FM_TODOS, n_t, 50.0f);
#endif
    x[5] = c->n_norm ? c->soma_amp_norm / (float)c->n_norm : NAN;
    x[6] = pct(c, EA_V_AMP, nn, 10.0f);
    x[7] = pct(c, EA_V_AMP, nn, 50.0f);
    x[8] = pct(c, EA_V_AMP, nn, 90.0f);
    x[9] = pct(c, EA_V_LOG_E, nn, 50.0f);
    x[10] = pct(c, EA_V_LOG_E, nn, 90.0f);
    x[11] = pct(c, EA_V_LOG_DUR, nn, 50.0f);
    x[12] = pct(c, EA_V_LOG_DUR, nn, 90.0f);
    x[13] = pct(c, EA_V_LOG_CONT, nn, 50.0f);
    x[14] = pct(c, EA_V_LOG_CONT, nn, 90.0f);
    x[15] = pct(c, EA_V_LOG_SUB, nn, 50.0f);
    x[16] = pct(c, EA_V_FM, nn, 50.0f);
    x[17] = pct(c, EA_V_LOG_RA, nn, 50.0f);
}

void ea_janela_fecha(ea_janela_t *j, float dur_s, float x[EA_N_DESCR])
{
    for (int k = 0; k < EA_N_CANAIS; k++) {
        fecha_canal(&j->c[k], dur_s, &x[k * EA_N_DESCR_CANAL]);
        zera_canal(&j->c[k]);
    }
    j->n_abaixo_limiar = 0;
}
