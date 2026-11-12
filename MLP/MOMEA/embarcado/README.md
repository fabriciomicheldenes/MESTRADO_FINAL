# MLP de EA embarcada: prova de conceito (MicroBlaze e lógica programável)

Leva as duas MLPs do estudo (`../mlp_ea.py`) para o MOMEA (XC7A35T):

* **FREQ**: regressor da frequência de rotação, em Hz;
* **CFG**: classificador da configuração de válvulas (recalque-sucção), com 7 classes.

As duas redes são 36 → 64 → 32 → k, com ReLU, e têm cerca de 4,6 mil parâmetros cada. Rodam uma vez por janela de 5 s.

> **Escopo.** Isto demonstra que a inferência é *viável e verificável* no hardware. O modelo
> exportado foi treinado com todas as janelas e serve à implementação. O desempenho
> fora da amostra é o do estudo principal (LORO e LOSO em `../RESUMO_RESULTADOS.md`), que ainda não
> justifica uso em campo. Trocar o modelo exige apenas rodar de novo o exportador; o hardware não muda.

## Estrutura

| Caminho | Conteúdo |
|---|---|
| `exportar_modelo.py` | treina os modelos finais, emula a rede int8 em aritmética inteira e gera `gerado/` |
| `gerado/mlp_pesos_f32.h` | pesos float32; imputação aplicada antes e padronização e desnormalização dobradas nos pesos |
| `gerado/mlp_pesos_q8.h` | pesos int8, bias int32, multiplicadores de requantização e pré-processamento |
| `gerado/vetores_teste.bin` | as 3 540 janelas: entrada bruta e saídas esperadas (float e int8) |
| `gerado/vetores_tb.h`, `vetores_mb.h` | subconjuntos para o testbench HLS (256) e para a MicroBlaze (16) |
| `src/mlp_nucleo_q8.c/.h` | **núcleo int8**: o mesmo fonte para PC, MicroBlaze e Vitis HLS (função de topo) |
| `src/mlp_inf.c/.h` | caminho float32, quantização da entrada e funções de conveniência |
| `src/teste_pc.c`, `teste_pc.bat` | teste de equivalência C × Python no PC |
| `hls/run_hls.tcl`, `hls/tb_mlp_nucleo_q8.c` | csim → síntese → cossimulação C/RTL → IP AXI4-Lite |
| `microblaze/mb_bench.c` | aplicação Vitis: confere as saídas e mede ciclos com o AXI Timer |
| `exportar_hits.py` | exporta os hits das 56 corridas e os descritores esperados; confere as fórmulas registro → unidades físicas |
| `gerado/descritores_cfg.h` | limiares harmonizados (A ≥ 65,12 e B ≥ 54,99 dBAE), janela de 5 s e ordem dos descritores |
| `gerado/hits_teste.bin`, `hits_mb.h` | todos os hits para o teste no PC; 4 janelas reais (323 hits) para a placa |
| `src/ea_descritores.c/.h` | **descritores em C**: registro bruto → unidades físicas → 36 descritores por janela |
| `src/teste_descritores.c` | teste da cadeia completa no PC: hits → descritores em C → MLP int8, contra o Python |
| `microblaze/mb_cadeia.c` | cadeia completa na MicroBlaze com 4 janelas reais, mais o esboço de integração no firmware |

## Fluxo de dados

```
registro de hit (32 B) -> ea_hit_de_registro: pico*vpc, dBAE, energia em eu, us
   -> ea_janela_adiciona: limiar harmonizado; contagens/somas; guarda log10 dos descritores de forma
   -> a cada 5 s, ea_janela_fecha: percentis (exatos ou por histograma) -> 36 floats (NaN = indefinido)
   -> mlp_quantiza_entrada: NaN->mediana, (x-média)/escala, saturação +-8, int8
   -> mlp_nucleo_q8 (C na MicroBlaze OU IP HLS): 2 redes, acumuladores int32 + classe
   -> Hz = acc*FREQ_Q_S_OUT*FREQ_Y_ESCALA + FREQ_Y_MEDIA ; classe -> "rec-suc"
```

Aritmética int8:

* entrada com escala fixa 8/127;
* pesos simétricos, com uma escala por camada;
* acumulação em int32;
* requantização com ReLU, `min(127, (max(acc,0)*M + 2^30) >> 31)`, com as escalas de ativação calibradas no percentil 99,9.

## Como rodar

```bat
cd embarcado
..\.venv\Scripts\python.exe exportar_modelo.py      :: regenera os pesos em gerado/ (≈1 min)
..\.venv\Scripts\python.exe exportar_hits.py        :: regenera hits e descritores esperados
teste_pc.bat                                        :: compila com o gcc do Vitis HLS e roda os 3 testes
cd hls
E:\Xilinx\Vitis_HLS\2022.2\bin\vitis_hls.bat -f run_hls.tcl
```

Para pular a cossimulação, defina a variável de ambiente `HLS_SEM_COSIM=1`.

Na **MicroBlaze**, crie no Vitis uma aplicação *standalone* sobre o XSA do projeto e adicione:

* `microblaze/mb_bench.c`, `src/mlp_inf.c` e `src/mlp_nucleo_q8.c`;
* os diretórios `src/` e `gerado/` como *include paths*;
* as opções `-O2 -ffp-contract=off`.

Para a cadeia completa, use `microblaze/mb_cadeia.c` no lugar de `mb_bench.c` e acrescente `src/ea_descritores.c`, com `-DMLP_SEM_F32` e, se quiser, `-DEA_DESCR_HISTOGRAMA`.

O benchmark usa o AXI Timer 0. Sem timer, compile com `-DMB_SEM_TIMER`. Para usar só o int8, dispense `mlp_pesos_f32.h`: remova as funções `*_f32` de `mlp_inf.c`, o que economiza cerca de 37 kB.

## Resultados (10/11/2026)

### Equivalência (PC, todas as 3 540 janelas)

| Comparação | Resultado |
|---|---|
| float32 em C × scikit-learn, frequência | diferença máxima de 2,5×10⁻⁴ Hz; classe idêntica em 100 % |
| int8 em C × emulação int8 em Python | **bit-exato**: 0 acumuladores diferentes e 0 classes diferentes |
| int8 × float, frequência | diferença média de 0,81 Hz e máxima de 11,8 Hz; mesmo nível de 10 Hz em 96,6 % das janelas |
| int8 × float, classe | igual em 98,2 % das janelas |
| Acurácia no próprio treino (só referência) | CFG: 92,8 % em float e 92,5 % em int8; FREQ: MAE de 1,54 Hz em float e 1,81 Hz em int8 |

### Descritores em C (cadeia completa no PC, 56 corridas, 281 595 hits, 3 540 janelas)

| Comparação | Modo exato (padrão) | Modo histograma |
|---|---|---|
| Descritores × `janelas_descritores.csv` | todos dentro de 10⁻³·(1+\|ref\|); erro máximo de 8×10⁻⁵ partindo das unidades do CSV; NaN idênticos | erro máximo de 0,2 dB nos percentis de amplitude, 0,04 nos log e 1,5 kHz em fm; NaN idênticos |
| Partindo do registro bruto e da calibração | erro máximo de 5×10⁻⁴ dB na amplitude, do arredondamento do CSV | igual ao anterior |
| **MLP int8 sobre os descritores do C × Python** | **classe igual em 100 % e nível de 10 Hz igual em 100 %** | classe igual em 97,4 % e nível igual em 96,2 % |
| RAM de `ea_janela_t` (2 canais) | 65,6 kB (até 1024 hits por canal na janela) | 10,3 kB (qualquer taxa) |

* As fórmulas registro → unidades físicas (`ea_hit_de_registro`) conferem com o `hits.csv` em erro relativo ≤ 10⁻⁵, que é o arredondamento do CSV:
  * `peak_v = pico·vpc`;
  * `dBAE = 20·log10(peak_v / G / 1 µV)`, com G = 10^(34/20);
  * `E[eu] = energia·(vpc/G)² / 10 MSa/s / 10⁻¹⁴`.
* O modo exato basta para os dados atuais, que têm no máximo 50 hits por canal por janela.
* O histograma serve para taxas altas, por exemplo 333 hits/s, com 1 665 hits por janela. Bins mais finos reduzem o erro à custa de memória.

### MicroBlaze (mb-gcc 2022.2, `-O2`)

ELF de teste da cadeia completa (`mb_cadeia.c`, com `-DMLP_SEM_F32`, linkado com libm):

| Variante | Código e constantes | RAM da janela |
|---|---|---|
| Sem FPU, exato / histograma | 44,3 / 44,7 kB | 66,7 / 11,4 kB |
| Com FPU, exato / histograma | 37,9 / 38,4 kB | 66,7 / 11,4 kB |

Os valores incluem cerca de 10 kB de dados de teste (`hits_mb.h`), que não existem no firmware real.
O custo por hit é dominado por 7 `log10f`: estima-se ~15 a 25 mil ciclos sem FPU e ~2 a 3 mil com FPU.
* Na taxa dos dados atuais (~100 hits por janela), isso dá menos de 1 % da CPU a 100 MHz.
* A 333 hits/s por canal, seria da ordem de 10 a 15 % sem FPU.

Essas estimativas precisam ser medidas com `mb_cadeia.c`, que imprime ciclos por hit, por fechamento de janela e pela MLP.

Núcleo da MLP (`mb_bench.c`):

| Item | Valor |
|---|---|
| Núcleo int8 (código + pesos das 2 redes) | 11,2 kB |
| Caminho float32 (código + pesos float das 2 redes) | 39,5 kB |
| Laço interno do int8 | 14 instruções por multiplicação-acumulação, cerca de 9,2 mil MACs por janela |
| Estimativa int8 a 100 MHz, com código e dados na memória local (LMB) | ≈ 150 mil ciclos ≈ 1,5 ms por janela |
| Estimativa float sem FPU (ponto flutuante emulado) | ≈ 10 a 20 ms por janela |

Os tempos na MicroBlaze são estimativas: confirme com `mb_bench.c` na placa. Mesmo o pior caso fica abaixo de 0,5 % do período da janela (5 s).

### Lógica programável (Vitis HLS 2022.2, xc7a35ticsg324-1L, 100 MHz)

| Item | Valor |
|---|---|
| Simulação C (csim), 256 janelas | 0 divergências |
| **Cossimulação C/RTL (Verilog, xsim)** | **PASS**: 0 divergências nas 256 janelas |
| Latência (2 redes + argmax) | 11 391 ciclos ≈ **114 µs** (cossimulação: 11 268 a 11 393 ciclos) |
| Folga de temporização estimada | 0,30 ns (alvo de 10 ns) |
| BRAM_18K | 7 (7 %) |
| DSP48E1 | 10 (11 %) |
| FF | 1 648 (3 %) |
| LUT | 2 421 (11 %) |
| Interface | AXI4-Lite: controle, entrada `xq[36]` e saídas `acc_freq`, `acc_cfg[7]` e `classe` |
| IP exportado | `hls/proj_mlp_q8/sol_100MHz/impl/ip/momea_hls_mlp_nucleo_q8_1_0.zip` |

Os recursos são da estimativa da síntese HLS. Os números finais saem da implementação no Vivado, junto
com o projeto atual do MOMEA, cuja ocupação precisa ser verificada para confirmar que cabe.

Cada laço interno roda com II = 1, mas há cerca de 12 ciclos de enchimento do pipeline por neurônio.
Achatar os laços ou desenrolar o laço de entrada reduziria a latência. Para uma decisão a cada 5 s,
isso não é necessário.

## Verificação por simulação RTL (sem placa)

`microblaze/sim/` mede ciclos executando os binários finais numa MicroBlaze simulada em RTL
(Vivado/xsim 2022.2, xc7a35ticsg324-1L, sem caches, 128 kB de LMB, AXI GPIO para as marcas):

* `mb_sim.c`: programa de teste que marca cada etapa no GPIO e escreve os resultados;
* `tb_mb.v`: testbench que registra o ciclo de cada marca;
* `sim_mb.tcl`: monta o sistema e gera a imagem .mem de cada ELF, que deve estar no conjunto `sim_1`;
* `rodar_sim_mb.py`: compila, elabora, simula e confere com o PC (`ref_host_*.txt`);
* `sintese_mb.tcl`: recursos do processador sem e com FPU.

O diretório de trabalho fica em `I:\Momea\2026\mbsim`, porque os caminhos do Vivado estouram o limite de 260 caracteres. A simulação roda a ~2,5 mil ciclos/s, por isso a carga é reduzida pelas macros `SIM_N_JAN` e `SIM_N_MLP`:

* com FPU: 2 janelas e 2 vetores;
* sem FPU: 1 janela e 1 vetor.

Resultados (10/11/2026; as 6 simulações deram resultados idênticos aos do PC):

| Etapa (ciclos a 100 MHz) | Sem FPU | Com FPU |
|---|---|---|
| MLP float32, duas redes | 3 353 499 | 212 130 |
| Quantização da entrada + núcleo int8 | 66 296 + 139 042 | 7 232 + 139 042 |
| Por hit, exato / histograma | 38 287 / 45 323 | 4 232 / 5 122 |
| Fechar a janela, exato / histograma | 749 938 / 214 316 | 117 676 / 91 974 |
| Núcleo MicroBlaze (síntese) | 1 361 LUT, 857 FF, 4 DSP | 2 322 LUT, 1 599 FF, 6 DSP |

IP HLS após place & route (`hls/run_hls_impl.tcl`): 656 LUT, 682 FF, 14 DSP, 3,5 BRAM36, período de 6,84 ns.

O artigo está em `../ARTIGO_SIM_EMBARCADO-10.11.2026/`:

* figuras e `numeros_mb.tex` gerados por `figuras_sim.py`;
* PDF compilado com pdflatex.

## Próximos passos

1. Rodar `mb_cadeia.c` na placa para medir os ciclos reais e escolher entre o modo exato e o histograma, e entre o C na MicroBlaze e o IP HLS.
2. Integrar ao firmware conforme o esboço em `mb_cadeia.c`:
   * acumular os hits no tratamento de cada registro;
   * fechar a janela num temporizador de 5 s e enviar frequência e classe num datagrama de status.
   * A janela passa a usar o relógio da placa, e não a hora de chegada no PC (`rx_time_s`), como no treino.
3. Retreinar com dados melhores (sem saturação, com o ganho ajustado) e regenerar `gerado/`.
4. Opcional: escala por neurônio nos pesos, para reduzir o erro de 11,8 Hz da quantização int8 no pior caso.
