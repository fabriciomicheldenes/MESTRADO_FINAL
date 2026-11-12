/* Descritores de EA por janela (porta em C de descritores_canal/janelar de mlp_ea.py).
 *
 * Uso no firmware:
 *   ea_janela_t jan;                       // estado da janela (estático: não cabe bem na pilha)
 *   ea_calib_t calib[2]; ea_calib_prepara(&calib[0], vpc_A, 34.0f); ... (uma vez)
 *   ea_janela_inicia(&jan, EA_LIMIAR_A_DBAE, EA_LIMIAR_B_DBAE);
 *   para cada registro de hit recebido da lógica:
 *       ea_hit_t h; ea_hit_de_registro(canal, pico, contagens, subida, duracao, energia,
 *                                      sat, maxdur, &calib[canal], &h);
 *       ea_janela_adiciona(&jan, &h);
 *   a cada EA_JANELA_S segundos:
 *       float x[EA_N_DESCR]; ea_janela_fecha(&jan, EA_JANELA_S, x);   // também reinicia
 *       mlp_q8(x, &freq_hz, &classe, acc);                             // ver mlp_inf.h
 *
 * Dois modos de cálculo dos percentis (compilação):
 *   padrão               : exato (igual ao numpy, interpolação linear). Guarda até
 *                          EA_MAX_HITS_CANAL hits por canal: 32 bytes/hit (1024 hits ~ 64 kB
 *                          para os 2 canais). Hits além disso entram nas contagens, médias e
 *                          somas, mas não nos percentis (ea_janela_t.c[].n_fora).
 *   -DEA_DESCR_HISTOGRAMA: histogramas de largura fixa, memória constante (~10 kB para os 2
 *                          canais) e custo O(1) por hit; percentis aproximados (erro < 1 bin).
 * NaN marca descritor indefinido (canal sem hits / sem hits normais), como no Python. */
#ifndef EA_DESCRITORES_H
#define EA_DESCRITORES_H
#include <stdint.h>

#define EA_N_CANAIS 2
#define EA_N_DESCR_CANAL 18
#define EA_N_DESCR (EA_N_CANAIS * EA_N_DESCR_CANAL)

#ifndef EA_MAX_HITS_CANAL
#define EA_MAX_HITS_CANAL 1024
#endif

typedef struct {
    float volts_por_codigo;   /* calibração do canal (config.json: calibracao.X.volts_per_code) */
    float ganho_db;           /* ganho do pré-amplificador (34 dB) */
    /* derivados, preenchidos por ea_calib_prepara (evita powf por hit) */
    float fator_amp;          /* 1 / (G * 1 uV) */
    float fator_energia;      /* (vpc/G)^2 / fs / 1e-14 */
} ea_calib_t;

void ea_calib_prepara(ea_calib_t *cal, float volts_por_codigo, float ganho_db);

typedef struct {
    uint8_t  canal;           /* 0 = A, 1 = B */
    uint8_t  sat, maxdur;
    uint32_t counts;
    float    peak_v;          /* V na entrada do ADC */
    float    amp_dbae;        /* dB ref. 1 uV na entrada do pré-amplificador */
    float    energy_eu;       /* 1 eu = 1e-14 V^2 s, referida à entrada do pré */
    float    rise_us, duration_us;
} ea_hit_t;

/* Variáveis guardadas por hit para os percentis */
enum { EA_V_FM_TODOS, EA_V_AMP, EA_V_LOG_E, EA_V_LOG_DUR, EA_V_LOG_CONT, EA_V_LOG_SUB,
       EA_V_FM, EA_V_LOG_RA, EA_N_VAR };

#ifdef EA_DESCR_HISTOGRAMA
#define EA_HIST_TOTAL 2564      /* soma dos bins das 8 variáveis (ver tabela em .c) */
#endif

typedef struct {
    uint32_t n, n_sat, n_maxdur, n_norm, n_fora;
    float soma_energia, soma_amp_norm;
#ifdef EA_DESCR_HISTOGRAMA
    uint16_t hist[EA_HIST_TOTAL];
#else
    float v[EA_N_VAR][EA_MAX_HITS_CANAL];
#endif
} ea_canal_acc_t;

typedef struct {
    ea_canal_acc_t c[EA_N_CANAIS];
    float limiar_dbae[EA_N_CANAIS];
    uint32_t n_abaixo_limiar;
} ea_janela_t;

/* Converte os campos do registro de 32 bytes em unidades físicas.
 * pico em códigos (|x - zero|), subida e duração em amostras de 10 MSa/s, energia bruta (48 bits).
 * cal deve ter sido preparada com ea_calib_prepara. */
void ea_hit_de_registro(uint8_t canal, uint32_t pico_codigo, uint32_t contagens,
                        uint32_t subida_amostras, uint32_t duracao_amostras, uint64_t energia,
                        uint8_t sat, uint8_t maxdur, const ea_calib_t *cal, ea_hit_t *h);

void ea_janela_inicia(ea_janela_t *j, float limiar_a_dbae, float limiar_b_dbae);
/* Retorna 1 se o hit entrou na janela, 0 se foi descartado (abaixo do limiar ou canal inválido). */
int  ea_janela_adiciona(ea_janela_t *j, const ea_hit_t *h);
/* Calcula os 36 descritores (ordem de gerado/descritores_cfg.h) e reinicia a janela. */
void ea_janela_fecha(ea_janela_t *j, float dur_s, float x[EA_N_DESCR]);

#endif
