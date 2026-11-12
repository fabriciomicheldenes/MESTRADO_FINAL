# Classificação das condições da bomba por EA: MLP, RF e GB

Pasta: `MLP/`. Script: `mlp_ea.py`. Saídas: `resultados/` (métricas, predições, figuras e `relatorio_auto.md`).
Data: 10/11/2026.

## Como rodar

```
cd I:\Momea\2026\Artigo\Relatorio_Artigo\MLP
.venv\Scripts\python.exe mlp_ea.py                      # janela de 5 s (padrão), ~5 min
.venv\Scripts\python.exe mlp_ea.py --janela 2           # outra janela
.venv\Scripts\python.exe mlp_ea.py --sem-harmonizar     # sem corte de limiar comum
```

O ambiente `.venv` usa o Python 3.15 em `C:\Python315`, que não tinha nenhuma biblioteca
científica. Por isso o venv foi criado e recebeu numpy, pandas, scikit-learn, matplotlib e
tabulate (versões em `requirements.txt`). Para recriá-lo:
`C:\Python315\python.exe -m venv .venv` e depois `.venv\Scripts\python.exe -m pip install -r requirements.txt`.

## 1. O que há em `Dados`

* Os dados são do **MOMEA**, não do Vallen: `hits.csv` gerado pelo `ae_hit_dashboard.py rev 0.6`, com
  dois canais (0 = A e 1 = B), mais `config.json` e `summary.json`.
* São **56 corridas, uma por condição** (8 frequências × 7 configurações de válvula), com cerca de
  300 s e 4 000 a 7 000 hits cada (283 618 hits no total).
  **Não há repetições**: toda pasta de condição contém uma única sessão `data_hora` (prefixo `01_`).
* Rótulos lidos do caminho: a pasta de 1º nível dá as válvulas (`100REC_50SUC` …) e a de 2º nível dá a
  frequência (`01_40HZ-…`). As configurações presentes são 100-100, 100-75, 100-50, 100-25, 75-100,
  50-100 e 25-100 (rec-suc %). O "700rec-50suc" citado era de fato **100rec-50suc**.
  **Erro de nomes:** as subpastas de `100REC_50SUC` se chamam `…REC100-25SUC…`. Considerei a pasta de
  1º nível como a correta, porque os horários (05/11, 16h52–17h45) são distintos dos da série
  `100REC_25SUC` (06/11). Convém confirmar com quem fez o ensaio.
* Ordem dos ensaios (cada série de válvulas é um bloco contínuo no tempo):

  | série (rec-suc) | dia/horário | ordem das frequências | limiar A/B (dBAE) |
  |---|---|---|---|
  | 100-100 | 05/11 10h42–12h09 | 30→90, depois 20 | 63/50 (20 Hz: 65/55) |
  | 25-100 | 05/11 15h54–16h41 | 90→20 (decrescente) | 65/55 |
  | 100-50 | 05/11 16h52–17h45 | 90→20 (decrescente) | 65/55 |
  | 100-25 | 06/11 09h49–10h32 | 20→90 | 50/50 |
  | 100-75 | 06/11 11h17–12h01 | 20→90 | 50/50 |
  | 75-100 | 06/11 14h49–15h32 | 20→90 | 50/50 |
  | 50-100 | 06/11 15h52–16h41 | 20→90 | 50/50 |

  O ganho foi de 34 dB em todas as corridas, HDT de 250 µs, HLT de 2 ms e MAXDUR de 100 ms.

### Problemas de qualidade nos dados (afetam a interpretação)

1. **ROVL = 1 em 100 % dos hits**, nos dois canais e em todas as corridas. No extrator, ROVL indica
   que houve cruzamento do limiar durante o HLT re-armável, o que reinicia a espera (ver artigo do
   sistema, seção 2). Com a emissão quase contínua da bomba, isso ocorre em todo hit. Por isso a flag
   não separa nada nestes dados e foi ignorada como descritor.
2. **A taxa de hits está limitada pelo tempo morto do detector.** Por canal, os hits chegam
   espaçados em cerca de 100 ms (≈10 hits/s). No canal A, quando o hit dura 100 ms (MAXDUR), o
   espaçamento sobe para cerca de 200 ms (≈5 hits/s). O canal B tem praticamente 3 000 hits por 300 s
   em todas as condições. Essa cadência é compatível com um bloqueio re-armável que se estende enquanto
   houver cruzamentos, até cerca de 100 ms. Assim, o detector registra aproximadamente um hit a cada
   100 ms, mais a duração do hit. Não encontrei o HDL da versão ensaiada para confirmar o limite. A
   consequência é que a "taxa de hits" não mede atividade acústica: ela é função da duração do hit.
   Os hits são uma amostragem do sinal contínuo, e não eventos isolados.
3. **Saturação:** 70 % dos hits têm SAT = 1. A partir de 40–50 Hz quase todos os hits dos dois canais
   saturam (amplitude presa em ~102,8 dBAE), e a amplitude deixa de informar. Com 34 dB de ganho, a
   faixa dinâmica é insuficiente acima de 40 Hz.
4. **MAXDUR:** 17,8 % dos hits, quase todos no canal A, com duração truncada em 100 ms. Esses hits
   representam emissão quase contínua (coerente com cavitação ou ruído de escoamento).
5. **O limiar mudou entre sessões** (63/50, 65/55 e 50/50 dBAE) e, por isso, está confundido com as
   válvulas e o dia. O script descarta os hits abaixo do maior limiar (A ≥ 65,1 dBAE,
   B ≥ 55,0 dBAE), mas esse corte **remove só 0,001 % dos hits**: os picos ficam quase sempre muito
   acima de qualquer dos limiares. A diferença de limiar age, portanto, sobre a duração, as
   contagens e a energia, que o FPGA mede acima do limiar da sessão, e isso não tem correção a
   posteriori. Ver a seção de sensibilidade.
6. A coluna `timestamp` está zerada. O tempo usado é `rx_time_s`, o tempo de chegada no PC.

## 2. Método

* **Janelas de 5 s** sem sobreposição: 3 540 janelas, cerca de 63 por corrida.
* **36 descritores** por janela (18 por canal):
  - com todos os hits: taxa, fração saturada, fração MAXDUR, log₁₀ da energia por segundo (inclui os
    hits contínuos) e frequência média (contagens/duração);
  - só com os hits sem MAXDUR, cuja duração não está truncada: média e percentis 10/50/90 da
    amplitude, percentis 50/90 de log(energia), log(duração) e log(contagens), mediana de
    log(tempo de subida), frequência média e log(RA), com RA = subida/amplitude em ms/V.

  Os valores faltantes (janela sem hit normal, 6,7 %) são imputados pela mediana. Os descritores
  são padronizados. Imputação e padronização são ajustadas só no treino de cada fold.
* **Alvos:**
  - frequência: **regressão**, com leitura ordinal por arredondamento ao nível mais próximo
    (20 a 90 Hz);
  - válvulas: **uma classe por configuração (7 classes, 8 corridas cada)**; % recalque e % sucção são
    derivados da classe prevista. Como o delineamento é em cruz, apenas 7 das 16 combinações
    (rec, suc) existem. Dois classificadores independentes poderiam prever combinações nunca
    ensaiadas e teriam classes muito desbalanceadas (100 % em 4 de 7).
* **Modelos:**
  - MLP de 64-32 neurônios, ReLU, Adam, L2 = 1e-3 e parada antecipada;
  - Random Forest com 300 árvores;
  - HistGradientBoosting.

  Os hiperparâmetros são fixos, sem busca, para não otimizar sobre o próprio teste.
* **Validação.** Nenhuma das três sorteia hits ou janelas. Como não há repetições, a
  "repetição deixada de fora" foi substituída por:
  - **BLOCOS (pseudo-repetição):** cada corrida é dividida em 5 trechos consecutivos de ~60 s; deixa-se
    um trecho de fora por vez, para todas as corridas, com 1 janela de exclusão entre treino e teste.
    É o mais próximo de "deixar uma repetição de fora", mas a mesma corrida aparece no treino.
  - **LORO (deixa uma corrida de fora, 56 folds):** a condição testada nunca foi vista. Sua frequência
    aparece no treino com outras válvulas, e sua configuração de válvulas aparece com outras
    frequências. **É o teste honesto de generalização.**
  - **LOSO (deixa uma série de válvulas de fora, 7 folds; só frequência):** retira uma sessão inteira
    (outro horário, outro limiar, outra configuração). É o teste mais duro contra deriva temporal.
* **Importância:** queda de desempenho ao permutar cada descritor, medida nos blocos de teste.
* **Não supervisionado:** k-means (k = 2, 3, 7, 8, 14 e 56) sobre os descritores padronizados, sem
  rótulos, e ARI/NMI contra frequência, válvulas, corrida, dia e limiar. Inclui também a PCA.

## 3. Resultados (janela de 5 s, limiar harmonizado)

### Frequência do motor

| validação | modelo | MAE (Hz) | R² | acerto do nível exato | acerto em ±10 Hz | acerto do nível por corrida |
|---|---|---|---|---|---|---|
| BLOCOS | MLP | 2,1 | 0,96 | 93 % | 99 % | 98 % |
| BLOCOS | RF | 1,1 | 0,99 | 94 % | 99 % | 98 % |
| BLOCOS | GB | 1,2 | 0,98 | 95 % | 99 % | 98 % |
| **LORO** | **MLP** | **5,6** | **0,79** | **69 %** | **92 %** | **71 %** |
| LORO | RF | 4,8 | 0,88 | 61 % | 94 % | 59 % |
| LORO | GB | 5,0 | 0,88 | 61 % | 95 % | 63 % |
| LOSO | MLP | 5,1 | 0,85 | 70 % | 94 % | 77 % |
| LOSO | RF | 4,2 | 0,89 | 69 % | 95 % | 68 % |
| LOSO | GB | 4,2 | 0,89 | 70 % | 95 % | 71 % |

O "acerto por corrida" usa a mediana das janelas de cada corrida.

* A frequência **generaliza** para condições e sessões não vistas: o erro médio fica em ~5 Hz, e
  92–95 % das janelas caem a até um nível (±10 Hz) do valor real. 20 e 30 Hz são reconhecidos quase
  sempre. Os erros se concentram entre níveis vizinhos de 40 a 90 Hz, faixa em que a saturação
  achata a amplitude. Ver `fig_freq_previsto_real_LORO.png` e `fig_confusao_LORO.png`.
* Em LOSO, o pior caso é a série **100-25** (sucção a 25 %), com MAE de ~10 Hz. Ela é a que mais
  difere das demais: a energia do canal A cai acima de 70 Hz e a fração MAXDUR é zero. Isso é
  coerente com outro regime de escoamento ou de cavitação na sucção estrangulada.
* As séries decrescentes (25-100 e 100-50) também generalizam (MAE de 3,6 a 7,9 Hz). Portanto, a
  previsão da frequência não depende só da ordem crescente do ensaio.
* A MLP não supera RF/GB em MAE, mas acerta mais o nível exato em LORO (69 % contra 61 %).
  Com 56 corridas, essa diferença não é conclusiva.

### Válvulas (7 configurações)

| validação | modelo | acurácia | acurácia balanceada | acerto % rec | acerto % suc | acerto por corrida |
|---|---|---|---|---|---|---|
| BLOCOS | MLP | 89 % | 89 % | 92 % | 95 % | 100 % |
| BLOCOS | RF | 95 % | 94 % | 96 % | 97 % | 100 % |
| BLOCOS | GB | 93 % | 93 % | 95 % | 96 % | 100 % |
| **LORO** | **MLP** | **28 %** | **28 %** | **57 %** | **59 %** | **30 %** |
| LORO | RF | 21 % | 21 % | 54 % | 53 % | 18 % |
| LORO | GB | 22 % | 22 % | 55 % | 53 % | 23 % |

Nível de acaso: 14 % para 7 classes; 57 % para % rec ou % suc chutando sempre "100".

* **Resultado principal:** com BLOCOS as válvulas parecem bem classificadas (89–95 %), mas em LORO
  o desempenho cai para 21–28 %, pouco acima do acaso. Os acertos de % rec e % suc ficam no nível do
  chute.
  Portanto, **o que se aprende em BLOCOS é a "assinatura" de cada corrida, e não um efeito de válvula
  que se transfira para outra frequência**. Essa assinatura inclui a deriva no tempo, o limiar e o
  estado do acoplamento.
* Em LORO, alguns pares se confundem de modo sistemático: 75-100 ↔ 50-100, e 100-75/100-50 → 100-25.
  O par 100-25 e o par 25-100 têm acerto próximo de 50 %, ou seja, as restrições mais severas
  têm algum sinal próprio.

### Importância dos descritores (`fig_importancia_descritores.png`)

* **Frequência:** dominam a energia por segundo do canal A (`A_log_energia_s`), o RA do canal B, os
  percentis de energia dos dois canais e a amplitude baixa (p10) do canal B. Ou seja, o nível de
  energia contínua cresce com a rotação.
* **Válvulas (em BLOCOS):** dominam a frequência média do canal A, a fração MAXDUR do canal A e a
  duração e as contagens dos dois canais. São descritores de forma e de continuidade do sinal.
  Pelo resultado em LORO, porém, eles identificam a corrida, e não a válvula.
* A importância por permutação subestima descritores correlacionados (energia, contagens e duração
  andam juntas). As ordens relativas são mais confiáveis que os valores absolutos.

### Verificação não supervisionada (`fig_nao_supervisionado.png`, `nao_supervisionado_ARI_NMI.csv`)

| k-means | ARI frequência | ARI válvulas | ARI corrida | ARI dia | ARI limiar |
|---|---|---|---|---|---|
| k = 2 | 0,16 | 0,00 | 0,02 | 0,01 | 0,01 |
| k = 8 | **0,31** | 0,10 | 0,11 | 0,04 | 0,01 |
| k = 7 | 0,31 | 0,10 | 0,11 | 0,04 | 0,01 |
| k = 56 | 0,20 | 0,12 | **0,47** | 0,02 | 0,03 |

* Sem rótulos, a estrutura dominante é a **frequência**: PC1 explica 56 % da variância e separa
  20–30 Hz (sem saturação) de ≥ 40 Hz (saturado). As válvulas quase não aparecem (ARI ≈ 0,1).
* Com k = 56, os clusters coincidem bastante com as corridas (ARI = 0,47). Cada corrida forma um
  aglomerado próprio, o que confirma a "assinatura de corrida" vista em BLOCOS.
* O ARI contra dia e contra limiar é próximo de zero: a sessão não é o fator dominante na estrutura
  global. Isso não exclui que ela pese na separação fina entre corridas, onde estão as válvulas.

### Sensibilidade (LORO/LOSO, `resultados_sensib/`)

| variante | freq LORO MAE (MLP / RF / GB) | freq LOSO MAE (MLP / RF / GB) | válvulas LORO acc (MLP / RF / GB) |
|---|---|---|---|
| principal (5 s, corte de limiar) | 5,6 / 4,8 / 5,0 Hz | 5,1 / 4,2 / 4,2 Hz | 28 / 21 / 22 % |
| sem corte de limiar | 5,4 / 4,8 / 5,0 Hz | 5,1 / 4,2 / 4,2 Hz | 28 / 21 / 22 % |
| janela de 2 s | 5,1 / 5,7 / 5,6 Hz | 5,0 / 4,9 / 5,0 Hz | 28 / 21 / 19 % |

* Sem o corte de limiar, o resultado é praticamente o mesmo, como esperado, já que o corte remove
  0,001 % dos hits.
* Com janela de 2 s, a MLP melhora um pouco (nível exato em LORO: 72 %; por corrida: 79 %). RF e GB
  pioram, porque a janela curta tem cerca de 20 hits por canal e descritores mais ruidosos.
* A conclusão principal não muda em nenhuma variante: a frequência generaliza (MAE ≈ 5 Hz) e as
  válvulas não generalizam entre corridas (≈ 20–28 %).
* Rodado com `--esquemas LORO,LOSO --saida resultados_sensib/...`.

## 4. Limitações

1. **Sem repetições.** Uma corrida por condição não permite estimar a variabilidade entre repetições
   nem a validação pedida (deixar uma repetição de fora). O BLOCOS é otimista por construção. O LORO
   é o número honesto, mas cada fold testa uma única corrida.
2. **Ordem fixa e deriva.** Cada série de válvulas foi ensaiada em bloco contínuo, em dias e horários
   diferentes, e a ordem das frequências é monótona dentro da série. Válvulas, dia, horário, limiar e
   temperatura estão **confundidos**. Nenhuma validação com estes dados separa o efeito da válvula
   do efeito do tempo. O LOSO mostra que a frequência resiste à troca de sessão; para as válvulas
   não há como testar.
3. **Limiar diferente entre sessões (50 a 65 dBAE).** Duração, contagens e energia foram medidas
   acima do limiar de cada sessão, e o corte comum em 65/55 dBAE não corrige isso. Com esses dados,
   o efeito do limiar não pode ser separado do efeito das válvulas.
4. **Taxa de hits limitada (~10 hits/s por canal) pelo rearme, e ROVL sempre ativo.** Os hits são
   uma amostragem do sinal contínuo.
   A taxa não é um descritor físico, e descritores de contagem ou atividade (Hits/s, contagem
   acumulada) ficam comprometidos.
5. **Saturação a partir de 40 Hz.** A amplitude e o pico deixam de discriminar nos níveis altos.
   Isso explica parte da confusão entre níveis vizinhos de 40 a 90 Hz.
6. **Janelas de uma mesma corrida são correlacionadas.** Por isso as métricas por janela inflam o
   número efetivo de amostras. A unidade independente é a corrida (56 no total).
7. Hiperparâmetros fixos, sem busca aninhada. Os resultados servem de linha de base, não de
   desempenho otimizado.

## 5. Recomendações para o próximo ensaio

* **Repetir cada condição pelo menos 2–3 vezes, em ordem aleatorizada** (ou em blocos aleatorizados
  por dia), incluindo a volta a condições já ensaiadas. Só assim é possível validar deixando uma
  repetição de fora e separar o efeito da válvula da deriva.
* Manter **limiar e ganho fixos** em todas as sessões. Reduzir o ganho, ou usar um limiar maior no
  canal A, para evitar a saturação acima de 40 Hz.
* Confirmar no HDL da versão ensaiada o **limite do bloqueio re-armável (~100 ms)**, que explicaria a
  cadência dos hits. Para sinal contínuo, um HLT não re-armável ou descritores contínuos (RMS e
  energia por intervalo fixo) representam melhor a atividade. Gravar o `timestamp` do FPGA.
* Para as válvulas, considerar descritores de forma de onda e espectro (bandas de frequência, RMS
  contínuo) em vez de hits, porque a cavitação é um sinal quase contínuo que o extrator de hits
  trunca (MAXDUR).

## 6. Arquivos gerados (`resultados/`)

| arquivo | conteúdo |
|---|---|
| `relatorio_auto.md` | tabelas de métricas geradas pelo script |
| `metricas.csv` | métricas por esquema, alvo, modelo e fold (inclui TOTAL) |
| `predicoes.csv` | predição de cada janela em cada validação |
| `janelas_descritores.csv` | tabela de descritores por janela (entrada dos modelos) |
| `corridas.csv` | metadados das 56 corridas (rótulos, horário, limiar, ganho) |
| `importancias.csv` | importância por permutação |
| `nao_supervisionado_ARI_NMI.csv`, `contingencia_k*_*.csv` | k-means × fatores |
| `fig_freq_previsto_real_{BLOCOS,LORO,LOSO}.png` | frequência prevista × real |
| `fig_confusao_{BLOCOS,LORO}.png` | matrizes de confusão (frequência e válvulas) |
| `fig_importancia_descritores.png` | importâncias |
| `fig_nao_supervisionado.png` | PCA e contingência do k-means |
| `fig_descritores_por_condicao.png` | descritores selecionados × frequência × válvulas |
