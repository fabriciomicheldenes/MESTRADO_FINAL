/* Gerado por exportar_hits.py. Parâmetros do cálculo de descritores
 * usados no treino (mlp_ea.py, harmonização de limiar ligada). */
#ifndef DESCRITORES_CFG_H
#define DESCRITORES_CFG_H
#define EA_JANELA_S 5.0f
#define EA_LIMIAR_A_DBAE 65.1200f  /* hits abaixo são descartados */
#define EA_LIMIAR_B_DBAE 54.9900f
/* Ordem dos 18 descritores por canal (canal A: 0-17, canal B: 18-35):
 *   0 taxa_hits
 *   1 frac_sat
 *   2 frac_maxdur
 *   3 log_energia_s
 *   4 fm_kHz_med
 *   5 amp_med
 *   6 amp_p10
 *   7 amp_p50
 *   8 amp_p90
 *   9 log_energia_p50
 *  10 log_energia_p90
 *  11 log_dur_p50
 *  12 log_dur_p90
 *  13 log_cont_p50
 *  14 log_cont_p90
 *  15 log_subida_p50
 *  16 fm_kHz_normais
 *  17 log_RA_p50
 */
#endif
