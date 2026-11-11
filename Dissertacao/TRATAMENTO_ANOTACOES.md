# Tratamento das anotações da banca de qualificação

Banca: Dr. Pedro/Thiago Berticelli Ló (IFPR, avaliador externo; arquivo `AnotacaosPedroLo.pdf`, 93 comentários) e Dr. Guilherme Galante (Unioeste; arquivo `AnotacoesGuilherme.pdf`, 90 comentários digitados e riscos).
As páginas citadas são as dos PDFs anotados (versão de fevereiro de 2026). O texto revisado está em `Dissertacao/` (compilar `Main.tex`).

## 1. Observações gerais (repetidas ao longo do texto)

| Observação | Quem | Como foi tratada |
|---|---|---|
| Faltam referências (marcadas como "ref", "REF?", "Refs?", "faltam mais refs", "todo conceito deve ser referenciado") | Ambos | Acrescentadas referências verificadas no Crossref em Introdução, EA, AM e FPGA (Grosse e Ohtsu 2008; Mba e Rao 2006; Brennen 1995; Bishop 2006; Goodfellow et al. 2016; LeCun et al. 2015; Rumelhart et al. 1986; Kingma e Ba 2015; Jolliffe e Cadima 2016; Trimberger 2015; Kuon et al. 2008; Cong et al. 2011; Jacob et al. 2018; Nagel et al. 2021). Todas as entradas novas têm DOI conferido. |
| Siglas: definir na primeira ocorrência, usar sempre depois, padronizar português/inglês | Ambos | Convenção adotada: nome em português, termo em inglês e sigla na primeira ocorrência; depois só a sigla. EA, AM (em vez de ML), RNA, DL (*Deep Learning*), MLP (*Multilayer Perceptron*), PCA, ADC, CLB, LUT. Lista de siglas refeita. |
| Uso de travessão "—" é "muito ChatGPT" | Guilherme (p. 22) | Todos os travessões do texto ativo foram removidos (pares viram parênteses; isolados, vírgula). |
| Espaços depois da vírgula; "na Equação" | Guilherme (p. 25) | Corrigido em EA (`hit,Matematicamente`, `,por`, `,essa`) e referências a equações com "Equação~\eqref". |
| Termos em inglês em itálico | Guilherme (p. 36, 38) | Aplicado em AM e FPGA (*momentum*, *dropout*, *weight decay*, *Field-Programmable Gate Arrays*, etc.). |
| Fonte das figuras ("O autor.") | Guilherme (p. 33) | Legendas com "Fonte: o autor." ou "Adaptado de...". |

## 2. Estrutura e conteúdo

| Página | Quem | Observação | Tratamento |
|---|---|---|---|
| 1 | Guilherme | Título: riscar "na Borda", acrescentar "em FPGAs" | Título: *Uma Arquitetura de Processamento em FPGA para Detecção de Falhas em Motobombas* (**confirmar com o orientador**). |
| 16 | Pedro | Faltou um parágrafo sobre trabalhos que levam a problemática para sistemas embarcados | Novo parágrafo na Motivação (Kang et al. 2015; Wirtz et al. 2018; Toumi et al. 2022). |
| 18 | Guilherme | Objetivo 4: "isso vai ser feito por simulação também?" | Objetivos reescritos conforme o que foi feito; a parte embarcada é verificada por simulação RTL, e isso é dito explicitamente (Seções 1.3 e 3.7; discussão em 4.5). |
| 18–19 | Pedro | "Por que não uma 1D-CNN / Temp-CNN?"; CNN-1D e CNN-2D "não estavam nos objetivos" | CNN removida dos objetivos e contribuições; justificativa na Seção 4.5 ("Por que uma MLP, e não uma rede convolucional?"): a entrada é um vetor de descritores de *hits*, não a forma de onda. |
| 21–22 | Guilherme | Parágrafo de abertura da seção de EA riscado ("Não é necessário") | Removido. |
| 21 | Pedro | "Técnica de EA" — EA é a emissão acústica, não a técnica | Corrigido ("A EA é um método..."). |
| 24 | Pedro | MARSE: traduzir | "(*measured area under the rectified signal envelope*, área medida sob o envelope retificado do sinal)". |
| 24 | Guilherme | Deixar claro que o nome da técnica é HIT | Texto explicita "detecção de *hits*" e remete à nova Seção 2.2 (Detecção e parametrização de *hits*). Corrigida também a definição de fim do *hit* (DDT/HDT) e do bloqueio (HLT). |
| 26 | Pedro | "da rede MLP" → deixar mais amplo (algoritmos de AM) | Reescrito: os descritores por janela servem a diferentes algoritmos de AM; aqui, MLP. Removida a afirmação incoerente de que a MLP aprende de formas de onda/espectrogramas. |
| 27 | Pedro | Citar Santos (2025), *Modelagem computacional dirigida por dados...* | Mantida a citação `santos2025modelagem` (**conferir dados da referência**). |
| 28 | Pedro | "Vai ter dados de acelerômetros também?" | Esclarecido: os ensaios do MOMEA usam só EA; vibração aparece só na validação externa (CWRU). |
| 28 | Pedro | Importante pesquisar a faixa de frequência necessária | Faixa de EA (dezenas de kHz a ~1 MHz) e consequência para a amostragem tratadas na Introdução e na EA; o extrator usa 10 MSa/s e filtro digital (Seção 3.4). |
| 29 | Guilherme | Mover o parágrafo sobre o escopo do projeto para depois da tabela de trabalhos e deixar clara a diferença deste trabalho | Movido; novo parágrafo com as três diferenças em relação aos trabalhos da tabela. |
| 30 | Guilherme | "2015 não é recente" | A referência (Zhang 2015) foi retirada da tabela: tratava de aceleradores de CNN em FPGA, não de cavitação. O mesmo para Nurvitadhi 2017 e Wang 2019. |
| 31 | Pedro | Tabela: e os resultados obtidos? Quais modelos tiveram maior sucesso? | Nova coluna "Resultados relatados", linha "Este trabalho" e frase sobre a ausência de um modelo superior absoluto. |
| 31 | Guilherme | Faltam referências na seção de AM; parágrafo inicial riscado | Parágrafo removido; referências acrescentadas. |
| 32 | Guilherme | "Melhorar a transição do assunto" (três paradigmas) | Nova frase de transição apresentando os três paradigmas com referências. |
| 36–37 | Guilherme | "Algoritmo 1 / pseudo-algoritmo"; "??? Não está explicado isso" | O pseudoalgoritmo virou Algoritmo 1, citado no texto; explicados \(J_{\mathcal{B}}(\theta)\) e a inicialização (Xavier/He). |
| 38 | Guilherme | A seção de FPGA deve vir antes da subseção de MLP em FPGA; precisa de referências | Subseção "MLP em FPGA para diagnóstico de falhas" movida para o fim da seção de FPGA; referências na abertura. |
| 43 | Guilherme | Figura 8 não referenciada; puxar a Figura 9 para o início | Figura do CLB citada no texto (legenda corrigida: "CBL" → CLB); figura do fluxo de desenvolvimento movida para o início da subseção. Na versão enxuta, a figura do CLB foi retirada. |
| 46 | Pedro | "Já tens libs para implementar MLP no FPGA?" | Resposta na Seção 4.5: núcleo próprio em C com fonte única para MicroBlaze e Vitis HLS (em vez de FINN/hls4ml), com verificação bit a bit. |
| 51 | Guilherme | Qual o tamanho das bases, número de entradas, % treino e teste? | Nova Tabela 3 (bases, tamanho, divisão treino/teste); detalhes nas Seções 3.6 e 3.8. |
| 53 | Guilherme | Incluir uma figura do pipeline | Nova Figura 7 (cadeia proposta), com cores distinguindo o que já existia e o que é contribuição. |
| 54–55 | Pedro | "Foi explorada a granularidade das janelas?"; "MLP convolucional?"; "o que seria a fase de Mineração?" | Metodologia da qualificação substituída pelos métodos efetivamente usados; janelas de 5 s e sensibilidade a 2 s (Seções 3.6 e 4.2); sem "MLP convolucional". |
| 57 | Pedro | Verificar as métricas | Métricas definidas por estudo: detecção, perdas com limite de confiança, concordância (extrator); MAE, acurácia, acurácia balanceada, F1 e acerto por corrida (bomba); fidelidade bit a bit, ciclos e recursos (embarcado). |
| 57–58 | Guilherme | Marcar onde começa o fluxo; legenda; trechos riscados | Figura da metodologia antiga substituída pela Figura 7; trechos riscados removidos. |
| 58 | Guilherme | Quanto tempo de coleta é necessário para detectar um defeito? | Resposta na Seção 4.5 ("Quanto tempo de coleta é necessário..."). |
| 59 | Guilherme | Deixar claro o que já estava implementado e o que é a contribuição | Seção 3.1 (parágrafo de delimitação das contribuições e Figura 7) e Seção 3.2 "Antecedentes". **Conferir com o autor** a atribuição de cada item. |
| 60 | Guilherme | Aumentar a figura | Figuras do MomeaProbe/MomeaCapture refeitas como subfiguras. |
| 62–63 | Pedro e Guilherme | Latência? A análise não é em tempo real? Os dados vão para o banco e depois são analisados? | Resposta na Seção 4.5 ("O sistema opera em tempo real?") e na Seção 3.2: com o FPGA, os registros chegam à aplicação à medida que os eventos ocorrem; a decisão é por janela. |
| 63 | Guilherme | Padronizar o nome da técnica; destacar a diferença da figura FPGA em relação à USB | Nome padronizado (detecção de *hits*); legenda e texto da Figura 13 destacam o bloco "FPGA + HIT". |
| 63 | Pedro | "Os dados brutos não são maiores?"; "não estava claro que seria compactado no FPGA" | Esclarecido: o FPGA não compacta; ele reduz o sinal a registros de *hits* (redução de cerca de 770 vezes, Seção 4.1). |
| 65 | Guilherme | Se só identifica sinais acima do limiar, por que precisa de ML? Não pode ser um método mais simples? | Resposta na Seção 3.1 e na Seção 4.5: o limiar decide o que é registrado; o modelo interpreta a população de *hits* (condição de operação). |
| 66 | Guilherme | O FPGA tem capacidade para processar tudo isso? | Resposta na Seção 4.5 ("O FPGA tem capacidade para a cadeia completa?"), com recursos medidos (extrator ~92 % das LUTs; IP HLS +656 LUTs; FPU +961 LUTs). |
| 68–71 | Guilherme | Confusão entre o que já foi implementado e a proposta; treino com dados brutos ou filtrados pelo HIT? | Capítulo de Resultados Parciais substituído pelos Capítulos 3 e 4; o treino usa descritores dos *hits* (dito na Seção 3.6). |
| 72 | Guilherme | Cronograma | Removido (não cabe na dissertação). |

## 3. Correções de referências encontradas na revisão

* `Alfayez2005` (qualificação) misturava dois artigos e tinha DOI inexistente; corrigido para o artigo do *Proc. IMechE Part E* (2005) e criada a entrada `Alfayez2005ndt` (NDT & E International, 2005).
* `krichen2023survey`, `Espinel2021`: DOIs e dados corrigidos.
* `Osete2024`: duplicata incorreta de `FernandezOsete2024`; citação trocada.
* `Zhang2015`, `Nurvitadhi2017`, `Wang2019`: eram citadas como estudos de cavitação, mas tratam de aceleradores de redes neurais em FPGA; retiradas da tabela de síntese da literatura.

## 4. Pendências para o autor

1. Confirmar o título e a banca/data da folha de aprovação (em vermelho no PDF).
2. Confirmar a delimitação das contribuições (Seção 3.1): o que é seu, o que é do projeto e o que é de outros membros.
3. Conferir referências sem DOI que não foram encontradas no Crossref (podem estar corretas, mas convém checar): `santos2025modelagem`, `haykin2001redes`, `libralao2003tecnicas`, `fayyad1996data`, `shapiro:2003`, `salami2020power`, `jaiswal2024multilayer`, `yan2024survey`, `wang2024review`, `Li2023SurveyFPGA`, `silva2024detecccao`, `Gomes2018`, `Galli2017`, `AnaliseVibracoesMotobomba2021`.
4. Completar a descrição da bancada da bomba (trechos `\completar{...}` em vermelho na Seção 3.6).
5. Revisão final de estilo das Seções 4.1 a 4.4 (texto vindo dos relatórios): o tom ainda é de relatório técnico em alguns parágrafos.

## 5. Versão enxuta (branch `dissertacao/versao-enxuta`)

Introdução, Fundamentação Teórica e Materiais e Métodos foram reduzidos ao essencial (de cerca de 110 para 28 páginas; total de 178 para 95 páginas); Resultados e Conclusões não mudaram. A versão completa continua no branch `dissertacao/versao-inicial`.

* Introdução: três seções (problema e motivação, objetivos, contribuições e organização).
* Fundamentação: EA e trabalhos relacionados (com a tabela de síntese), nova Seção 2.2 de *hits* condensada (`FundamentacaoTeorica/04-Hits.tex`), AM e MLP sem o histórico e sem os trechos comentados, FPGA com arquitetura, fluxo, quantização e MLP em FPGA. Saíram a figura de paradigmas de AM, a figura do CLB, a lista longa de otimizações e a tabela de otimização de MLP em FPGA (os trabalhos continuam citados em um parágrafo).
* Materiais e Métodos: o extrator (Seções 3.3 a 3.5) foi condensado a partir do artigo de 20 páginas, **sem a subseção "Arquitetura inicial baseada em streaming contínuo"**; os antecedentes USB ficam resumidos na Seção 3.2. Os métodos da bomba, da inferência embarcada e do CWRU foram resumidos, mantendo todos os rótulos citados nos Resultados.
* Os arquivos `importado/v2_decisoes.tex`, `v2_implementacao.tex`, `v2_delineamento.tex` e `v2_hits_fundamentos.tex` não são mais incluídos, mas foram mantidos no repositório.
