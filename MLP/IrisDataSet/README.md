# MLP aplicada ao conjunto de dados Iris

Versão do modelo MLP do projeto (originalmente em `../mlp_ea.py`, para emissão acústica)
adaptada ao clássico conjunto Iris de Fisher: 150 amostras, 4 atributos, 3 classes.

Além do modelo, esta pasta contém o relatório técnico em LaTeX — um texto didático sobre redes
perceptron multicamadas, com derivações matemáticas, figuras e tabelas geradas automaticamente
pela execução.

## Como executar

A partir da pasta `MLP/`:

```bat
.venv\Scripts\python.exe iris\mlp_iris.py
```

A execução leva cerca de 4 minutos e produz:

| Pasta / arquivo | Conteúdo |
|---|---|
| `resultados/` | métricas em CSV, `resumo.md`, log da execução |
| `artigo/figuras/` | 22 figuras em PNG (220 dpi), usadas no relatório |
| `artigo/tabelas/` | tabelas em LaTeX (`booktabs`) |
| `artigo/numeros.tex` | macros LaTeX com os números citados no texto |

## Como compilar o relatório

```bat
cd iris\artigo
pdflatex relatorio.tex
pdflatex relatorio.tex
```

A segunda passagem é necessária para atualizar o sumário, as referências cruzadas e a lista de
figuras. O resultado é `relatorio.pdf` (56 páginas).

## O que o script faz

1. análise exploratória (estatísticas, matriz de dispersão, boxplots, correlação, PCA);
2. varredura de arquiteturas (de 4 a 128-64 neurônios);
3. varredura de funções de ativação (ReLU, tanh, logística, identidade);
4. varredura da regularização L2 (α de 1e-6 a 1e2);
5. curvas de treinamento e de aprendizado;
6. divisão treino/teste estratificada (70/30) e métricas finais (acurácia, F1 macro,
   entropia cruzada, intervalo de confiança de Wilson);
7. fronteiras de decisão e importância por permutação;
8. visualização dos pesos aprendidos;
9. verificação numérica do gradiente e implementação manual em NumPy;
10. comparação com RF, gradient boosting e regressão logística, com teste de Wilcoxon pareado.

## Resultado principal

| Métrica | Valor |
|---|---|
| Acurácia (validação cruzada 5 × 5) | 96,8 % |
| Acurácia (teste, n = 45) | 93,3 % (IC 95 %: 82,1 – 97,7 %) |
| F1 macro (teste) | 0,933 |
| Parâmetros da rede | 131 (4–16–3) |
| Erro relativo do gradiente | 1,2 × 10⁻⁵ |

A rede empata com floresta aleatória e gradient boosting e fica ligeiramente atrás da regressão
logística (97,5 %; p = 0,025 no teste de Wilcoxon). O relatório discute por que isso acontece e o
que o resultado significa — o Iris é quase linearmente separável, de modo que a não linearidade da
MLP não tem estrutura para explorar.

## Requisitos

Python 3.15 com as versões de `../requirements.txt` (NumPy 2.5, pandas 3.1, scikit-learn 1.9,
SciPy 1.18, Matplotlib 3.11). Para o PDF, uma distribuição LaTeX com `pdflatex`, `tikz`,
`pgfplots`, `booktabs` e `listings` (testado com TeX Live 2026).
