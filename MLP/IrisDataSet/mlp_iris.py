# -*- coding: utf-8 -*-
"""
mlp_iris.py - Rede neural MLP aplicada ao conjunto de dados Iris.

Esta é a versão Iris do modelo MLP desenvolvido em ``MLP/mlp_ea.py``. A estrutura
geral foi mantida (leitura -> descritores -> validação sem vazamento -> figuras e
relatório automático), mas o problema agora é o clássico Iris de Fisher
(150 amostras, 4 atributos contínuos, 3 classes), usado aqui como estudo
didático de uma rede MLP.

Perguntas que o script responde
  1. Como os dados se distribuem e quão separáveis são as classes?
  2. Qual arquitetura (número de camadas/neurônios) é adequada?
  3. Como a função de ativação e a regularização L2 afetam o resultado?
  4. Como o treinamento converge (curvas de perda e de acurácia)?
  5. Quais são os erros do modelo (matriz de confusão, precisão/revocação/F1)?
  6. O modelo se compara favoravelmente a RF, gradient boosting e regressão
     logística?
  7. A implementação manual em NumPy (com retropropagação escrita à mão)
     reproduz o que o scikit-learn faz?

Protocolo
  * Divisão estratificada treino/teste (70/30) para a estimativa final.
  * Validação cruzada estratificada repetida (5 dobras x 5 repetições) para
    comparar configurações, sempre com as mesmas dobras entre modelos.
  * Padronização (z-score) ajustada apenas no treino de cada dobra, dentro de
    um Pipeline, para não vazar informação do teste.

Saídas
  resultados/  metricas em CSV, resumo em Markdown e previsões
  artigo/figuras/  figuras em PNG (usadas pelo relatório LaTeX)
  artigo/tabelas/  tabelas em LaTeX (booktabs)
  artigo/numeros.tex  macros LaTeX com os números citados no texto

Uso
  .venv\\Scripts\\python.exe mlp_iris.py
  .venv\\Scripts\\python.exe mlp_iris.py --sem-numpy   # pula a implementação manual
"""
from __future__ import annotations

import argparse
import json
import os
import time
import warnings
from pathlib import Path

# Limita o número de threads das bibliotecas numéricas: com 150 amostras o ganho
# é nulo e o paralelismo por processos pode esbarrar em restrições do sistema.
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import ListedColormap  # noqa: E402
from scipy import stats  # noqa: E402
from sklearn.base import clone  # noqa: E402
from sklearn.datasets import load_iris  # noqa: E402
from sklearn.decomposition import PCA  # noqa: E402
from sklearn.ensemble import (GradientBoostingClassifier,  # noqa: E402
                              RandomForestClassifier)
from sklearn.exceptions import ConvergenceWarning  # noqa: E402
from sklearn.inspection import permutation_importance  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import (accuracy_score, balanced_accuracy_score,  # noqa: E402
                             confusion_matrix, f1_score, log_loss,
                             precision_recall_fscore_support)
from sklearn.model_selection import (StratifiedKFold, cross_val_predict,  # noqa: E402
                                     learning_curve, train_test_split)
from sklearn.neural_network import MLPClassifier  # noqa: E402
from sklearn.pipeline import make_pipeline  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

warnings.filterwarnings("ignore", category=ConvergenceWarning)

# --------------------------------------------------------------------------- #
# Caminhos e constantes
# --------------------------------------------------------------------------- #
BASE = Path(__file__).resolve().parent
RES = BASE / "resultados"
FIG = BASE / "artigo" / "figuras"
TAB = BASE / "artigo" / "tabelas"
for _d in (RES, FIG, TAB):
    _d.mkdir(parents=True, exist_ok=True)

SEMENTE = 42
REPETICOES, DOBRAS = 5, 5          # validação cruzada: 5 x 5 = 25 dobras

# Configuração adotada como modelo principal. Não é um palpite: resulta das
# varreduras de arquitetura, ativação e regularização feitas neste mesmo script
# (ver resultados/arquiteturas.csv, ativacoes.csv e regularizacao.csv).
MLP_PRINCIPAL = dict(hidden=(16,), ativacao="tanh", alpha=1e-2)

ATRIBUTOS = ["SL", "SW", "PL", "PW"]
ROTULO = {
    "SL": "comprimento da sépala (cm)",
    "SW": "largura da sépala (cm)",
    "PL": "comprimento da pétala (cm)",
    "PW": "largura da pétala (cm)",
}
CURTO = {"SL": "comp. sépala", "SW": "larg. sépala",
         "PL": "comp. pétala", "PW": "larg. pétala"}
CLASSES = ["setosa", "versicolor", "virginica"]
COR = {"setosa": "#2a78d6", "versicolor": "#eb6834", "virginica": "#1baf7a"}
COR_MOD = {"MLP": "#2a78d6", "RF": "#eb6834", "GB": "#1baf7a", "LR": "#eda100"}
TINTA, TINTA2, GRADE = "#0b0b0b", "#52514e", "#e4e3df"
CMAP3 = ListedColormap([COR[c] for c in CLASSES])

plt.rcParams.update({
    "figure.dpi": 110, "savefig.dpi": 220, "savefig.bbox": "tight",
    "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9,
    "axes.edgecolor": TINTA2, "axes.labelcolor": TINTA,
    "xtick.color": TINTA2, "ytick.color": TINTA2,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRADE, "grid.linewidth": 0.6,
    "axes.axisbelow": True, "legend.frameon": False,
    "text.usetex": False,
})


# --------------------------------------------------------------------------- #
# Utilidades de formatação e saída
# --------------------------------------------------------------------------- #
def brl(x: float, d: int = 3) -> str:
    """Número no padrão brasileiro, seguro em modo texto e matemático do LaTeX."""
    s = f"{x:.{d}f}"
    return s.replace(".", "{,}")


def br(x: float, d: int = 3) -> str:
    """Número no padrão brasileiro (texto simples, para CSV/Markdown)."""
    return f"{x:.{d}f}".replace(".", ",")


def salvar(fig, nome: str) -> None:
    fig.savefig(FIG / nome, dpi=220)
    plt.close(fig)
    print(f"    figura: {nome}")


def cient(v: float, d: int = 1) -> str:
    """Notação científica em LaTeX, com vírgula decimal."""
    if v == 0 or not np.isfinite(v):
        return "$0$"
    e = int(np.floor(np.log10(abs(v))))
    return "$%s\\times10^{%d}$" % (brl(v / 10 ** e, d), e)


def tab_latex(df: pd.DataFrame, nome: str, colspec: str, decimais: int | None = None) -> None:
    """Grava uma tabela booktabs; números float saem com vírgula decimal."""
    def esc(s):
        s = str(s)
        for a, b in (("_", r"\_"), ("%", r"\%"), ("#", r"\#"), ("&", r"\&")):
            s = s.replace(a, b)
        return s

    linhas = ["% gerado automaticamente por mlp_iris.py - nao editar", ""]
    linhas.append(r"\begin{tabular}{%s}" % colspec)
    linhas.append(r"\toprule")
    linhas.append(" & ".join(r"\textbf{%s}" % esc(c) for c in df.columns) + r" \\")
    linhas.append(r"\midrule")
    for _, r in df.iterrows():
        vals = []
        for v in r:
            if isinstance(v, (float, np.floating)):
                vals.append(brl(v, decimais if decimais is not None else (3 if abs(v) < 1 else 1)))
            else:
                vals.append(esc(v))
        linhas.append(" & ".join(vals) + r" \\")
    linhas.append(r"\bottomrule")
    linhas.append(r"\end{tabular}")
    (TAB / f"{nome}.tex").write_text("\n".join(linhas) + "\n", encoding="utf-8")


def n_parametros(rede) -> int:
    """Número de parâmetros treináveis de um MLPClassifier."""
    return int(sum(w.size for w in rede.coefs_) + sum(b.size for b in rede.intercepts_))


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Intervalo de confiança de Wilson para uma proporção."""
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return c - h, c + h


def dobras_repetidas(X, y, repeticoes=REPETICOES, dobras=DOBRAS, semente=100):
    """Gera (treino, teste) de uma CV estratificada repetida, sempre na mesma ordem."""
    for r in range(repeticoes):
        skf = StratifiedKFold(n_splits=dobras, shuffle=True, random_state=semente + r)
        yield from skf.split(X, y)


# --------------------------------------------------------------------------- #
# 1. Dados e análise exploratória
# --------------------------------------------------------------------------- #
def carregar() -> tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray]:
    d = load_iris()
    df = pd.DataFrame(d.data, columns=ATRIBUTOS)
    df["especie"] = [d.target_names[t] for t in d.target]
    return df, d.data, d.target, d.target_names


def eda(df: pd.DataFrame, y: np.ndarray) -> pd.DataFrame:
    print("1) Análise exploratória")
    # estatísticas descritivas por classe
    linhas = []
    for esp in CLASSES:
        g = df[df.especie == esp]
        for a in ATRIBUTOS:
            linhas.append(dict(especie=esp, atributo=a, media=g[a].mean(), desvio=g[a].std(),
                               minimo=g[a].min(), mediana=g[a].median(), maximo=g[a].max()))
    est = pd.DataFrame(linhas)
    est.to_csv(RES / "eda_estatisticas_por_classe.csv", index=False)

    geral = []
    for a in ATRIBUTOS:
        geral.append({'atributo': CURTO[a], 'média': df[a].mean(), 'desvio': df[a].std(),
                      'mínimo': df[a].min(), 'mediana': df[a].median(),
                      'máximo': df[a].max()})
    tab_latex(pd.DataFrame(geral), "tab_estatisticas", "lrrrrr", decimais=2)

    # tabela por classe (média +- desvio)
    piv = est.pivot(index="atributo", columns="especie", values="media").loc[ATRIBUTOS]
    piv_s = est.pivot(index="atributo", columns="especie", values="desvio").loc[ATRIBUTOS]
    t = pd.DataFrame(index=[CURTO[a] for a in ATRIBUTOS])
    for esp in CLASSES:
        t[esp] = [f"{brl(piv.loc[a, esp], 2)} $\\pm$ {brl(piv_s.loc[a, esp], 2)}" for a in ATRIBUTOS]
    tab_latex(t.reset_index().rename(columns={"index": "atributo"}),
              "tab_estatisticas_classe", "lccc")

    # 1.1 matriz de dispersão
    fig, axs = plt.subplots(4, 4, figsize=(9.4, 9.0))
    lim = {a: (df[a].min() - 0.15, df[a].max() + 0.15) for a in ATRIBUTOS}
    for i in range(4):
        for j in range(4):
            ax = axs[i, j]
            if i < j:
                ax.axis("off")
                continue
            if i == j:
                for esp in CLASSES:
                    g = df[df.especie == esp]
                    ax.hist(g[ATRIBUTOS[j]], bins=11, range=lim[ATRIBUTOS[j]],
                            color=COR[esp], histtype="step", linewidth=1.4,
                            label=esp)
                ax.set_xlim(*lim[ATRIBUTOS[j]])
                ax.set_yticks([])
                ax.set_title(CURTO[ATRIBUTOS[j]], fontsize=9)
            else:
                for esp in CLASSES:
                    g = df[df.especie == esp]
                    ax.scatter(g[ATRIBUTOS[j]], g[ATRIBUTOS[i]], s=11, color=COR[esp],
                               alpha=0.75, linewidths=0, label=esp)
                ax.set_xlim(*lim[ATRIBUTOS[j]])
                ax.set_ylim(*lim[ATRIBUTOS[i]])
            if i == 3:
                ax.set_xlabel(CURTO[ATRIBUTOS[j]], fontsize=8)
            if j == 0 and i > 0:
                ax.set_ylabel(CURTO[ATRIBUTOS[i]], fontsize=8)
            if j > 0:
                ax.tick_params(labelleft=False)
            ax.tick_params(labelsize=7)
    h, l = axs[1, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper right", bbox_to_anchor=(0.995, 0.99), fontsize=9,
               title="espécie", title_fontsize=9)
    fig.suptitle("Matriz de dispersão do conjunto Iris (150 amostras)", y=1.0, fontsize=11)
    salvar(fig, "fig_iris_dispersao.png")

    # 1.2 boxplots
    fig, axs = plt.subplots(1, 4, figsize=(11, 3.1))
    for ax, a in zip(axs, ATRIBUTOS):
        dados = [df.loc[df.especie == e, a].to_numpy() for e in CLASSES]
        bp = ax.boxplot(dados, patch_artist=True, widths=0.6, showfliers=False)
        for b, e in zip(bp["boxes"], CLASSES):
            b.set(facecolor=COR[e], alpha=0.5, edgecolor=COR[e], linewidth=1.1)
        for k in ("whiskers", "caps", "medians"):
            for ln in bp[k]:
                ln.set(color=TINTA2, linewidth=1.1)
        ax.set_xticks([1, 2, 3], ["setosa", "versicol.", "virgin."], fontsize=8)
        ax.set_title(CURTO[a])
    fig.suptitle("Distribuição dos quatro atributos por espécie", fontsize=11)
    fig.tight_layout()
    salvar(fig, "fig_iris_boxplots.png")

    # 1.3 correlação
    C = np.corrcoef(df[ATRIBUTOS].to_numpy().T)
    fig, ax = plt.subplots(figsize=(4.6, 4.0))
    im = ax.imshow(C, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(4), [CURTO[a] for a in ATRIBUTOS], rotation=40, ha="right")
    ax.set_yticks(range(4), [CURTO[a] for a in ATRIBUTOS])
    for i in range(4):
        for j in range(4):
            ax.text(j, i, br(C[i, j], 2), ha="center", va="center", fontsize=8,
                    color="white" if abs(C[i, j]) > 0.6 else TINTA)
    ax.grid(False)
    ax.set_title("Correlação de Pearson entre atributos")
    fig.colorbar(im, ax=ax, shrink=0.82)
    salvar(fig, "fig_iris_correlacao.png")

    # 1.4 PCA
    Z = StandardScaler().fit_transform(df[ATRIBUTOS].to_numpy())
    pca = PCA().fit(Z)
    P = pca.transform(Z)
    fig, axs = plt.subplots(1, 2, figsize=(10.4, 3.9),
                            gridspec_kw={"width_ratios": [1.35, 1]})
    for esp in CLASSES:
        m = df.especie == esp
        axs[0].scatter(P[m, 0], P[m, 1], s=20, color=COR[esp], alpha=0.8,
                       linewidths=0.4, edgecolors="white", label=esp)
    axs[0].set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%})")
    axs[0].set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%})")
    axs[0].legend(fontsize=8, title="espécie", title_fontsize=8)
    axs[0].set_title("Projeção nas duas primeiras componentes principais")
    axs[1].bar(range(1, 5), pca.explained_variance_ratio_, color="#2a78d6", width=0.6,
               label="por componente")
    axs[1].plot(range(1, 5), np.cumsum(pca.explained_variance_ratio_), "o-",
                color="#eb6834", ms=4, lw=1.4, label="acumulada")
    axs[1].set_xticks(range(1, 5), [f"PC{i}" for i in range(1, 5)])
    axs[1].set_ylim(0, 1.05)
    axs[1].set_ylabel("variância explicada")
    axs[1].legend(fontsize=8)
    axs[1].set_title("Variância explicada")
    fig.tight_layout()
    salvar(fig, "fig_iris_pca.png")

    return est


# --------------------------------------------------------------------------- #
# 2. Modelo base e protocolo
# --------------------------------------------------------------------------- #
def mlp(hidden=(16, 8), ativacao="relu", alpha=1e-3, semente=SEMENTE,
        solver="adam", lr=1e-3, max_iter=3000, early_stopping=False) -> object:
    """Pipeline padronização + MLP. A padronização é ajustada dentro de cada dobra."""
    return make_pipeline(
        StandardScaler(),
        MLPClassifier(hidden_layer_sizes=hidden, activation=ativacao, alpha=alpha,
                      solver=solver, learning_rate_init=lr, max_iter=max_iter,
                      early_stopping=early_stopping, validation_fraction=0.15,
                      n_iter_no_change=40, random_state=semente),
    )


def mlp_principal(**kw) -> object:
    """O MLP adotado no restante do trabalho."""
    p = dict(MLP_PRINCIPAL)
    p.update(kw)
    return mlp(**p)


def modelos_referencia() -> dict:
    return {
        "MLP": mlp_principal(),
        "RF": make_pipeline(StandardScaler(), RandomForestClassifier(
            n_estimators=400, min_samples_leaf=1, random_state=SEMENTE, n_jobs=1)),
        "GB": make_pipeline(StandardScaler(), GradientBoostingClassifier(
            n_estimators=300, learning_rate=0.06, max_depth=3, subsample=0.9,
            random_state=SEMENTE)),
        "LR": make_pipeline(StandardScaler(), LogisticRegression(
            max_iter=2000, C=10.0, random_state=SEMENTE)),
    }


def cv_emparelhada(modelos: dict, X, y) -> tuple[pd.DataFrame, dict]:
    """CV repetida com as mesmas dobras para todos os modelos (comparação justa)."""
    nomes = list(modelos)
    acc = {k: [] for k in nomes}
    f1m = {k: [] for k in nomes}
    tempo = {k: 0.0 for k in nomes}
    for k, m in modelos.items():
        t0 = time.time()
        for tr, te in dobras_repetidas(X, y):
            mm = clone(m).fit(X[tr], y[tr])
            p = mm.predict(X[te])
            acc[k].append(accuracy_score(y[te], p))
            f1m[k].append(f1_score(y[te], p, average="macro"))
        tempo[k] = time.time() - t0
    linhas = []
    for k in nomes:
        linhas.append(dict(modelo=k, acuracia=np.mean(acc[k]), desvio=np.std(acc[k], ddof=1),
                           f1_macro=np.mean(f1m[k]), f1_desvio=np.std(f1m[k], ddof=1),
                           tempo_25_dobras=tempo[k]))
    return pd.DataFrame(linhas), acc


# --------------------------------------------------------------------------- #
# 3. Estudo de hiperparâmetros
# --------------------------------------------------------------------------- #
ARQUITETURAS = [(4,), (8,), (16,), (32,), (8, 4), (16, 8), (32, 16), (64, 32), (128, 64)]


def estudo_arquiteturas(X, y) -> pd.DataFrame:
    print("2) Arquiteturas")
    linhas = []
    for h in ARQUITETURAS:
        m = mlp(h)
        accs, perdas = [], []
        t0 = time.time()
        mm = None
        for tr, te in dobras_repetidas(X, y, repeticoes=3):
            mm = clone(m).fit(X[tr], y[tr])
            accs.append(accuracy_score(y[te], mm.predict(X[te])))
            perdas.append(log_loss(y[te], mm.predict_proba(X[te]), labels=[0, 1, 2]))
        fits = 3 * DOBRAS
        npar = n_parametros(mm[-1])
        linhas.append(dict(arquitetura="(" + ", ".join(map(str, h)) + ")",
                           neuronios=int(sum(h)), profundidade=len(h),
                           parametros=npar, acuracia=np.mean(accs),
                           desvio=np.std(accs, ddof=1), log_loss=np.mean(perdas),
                           ms_por_ajuste=1000 * (time.time() - t0) / fits))
        print(f"    {str(h):12s} acc={np.mean(accs):.3f}  p={npar}")
    df = pd.DataFrame(linhas)

    fig, axs = plt.subplots(1, 2, figsize=(11.4, 3.8))
    x = np.arange(len(df))
    axs[0].errorbar(x, df.acuracia, yerr=df.desvio, fmt="o-", color="#2a78d6",
                    ms=5, lw=1.5, capsize=3)
    axs[0].set_xticks(x, df.arquitetura, rotation=40, ha="right", fontsize=8)
    axs[0].set_ylabel("acurácia (CV 5$\\times$3)")
    axs[0].set_title("Acurácia por arquitetura")
    axs[0].set_ylim(0.83, 1.03)
    axs[1].errorbar(df.parametros, df.acuracia, yerr=df.desvio, fmt="o", color="#eb6834",
                    ms=5, capsize=3)
    for _, r in df.iterrows():
        axs[1].annotate(r.arquitetura, (r.parametros, r.acuracia), fontsize=6.5,
                        xytext=(4, 3), textcoords="offset points", color=TINTA2)
    axs[1].set_xscale("log")
    axs[1].set_xlabel("número de parâmetros (escala log)")
    axs[1].set_ylabel("acurácia")
    axs[1].set_title("Custo $\\times$ desempenho")
    fig.tight_layout()
    salvar(fig, "fig_arquiteturas.png")

    t = df[['arquitetura', 'profundidade', 'parametros', 'acuracia', 'desvio', 'log_loss']].copy()
    t.columns = ['arquitetura', 'profundidade', 'parâmetros', 'acurácia', 'desvio', 'log-loss']
    t['acurácia'] = (100 * t['acurácia']).round(1)
    t['desvio'] = (100 * t['desvio']).round(1)
    tab_latex(t, "tab_arquiteturas", "lrrrrr")
    df.to_csv(RES / "arquiteturas.csv", index=False)
    return df


def estudo_ativacoes(X, y) -> pd.DataFrame:
    print("3) Funções de ativação")
    linhas = []
    for a in ("relu", "tanh", "logistic", "identity"):
        m = mlp((16, 8), ativacao=a)
        accs, perdas = [], []
        for tr, te in dobras_repetidas(X, y, repeticoes=3):
            mm = clone(m).fit(X[tr], y[tr])
            accs.append(accuracy_score(y[te], mm.predict(X[te])))
            perdas.append(log_loss(y[te], mm.predict_proba(X[te]), labels=[0, 1, 2]))
        linhas.append(dict(ativacao=a, acuracia=np.mean(accs), desvio=np.std(accs, ddof=1),
                           log_loss=np.mean(perdas)))
    df = pd.DataFrame(linhas)

    fig, axs = plt.subplots(1, 2, figsize=(10.2, 3.6))
    cores = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
    axs[0].bar(df.ativacao, df.acuracia, yerr=df.desvio, color=cores, width=0.6,
               capsize=4, error_kw={"elinewidth": 1})
    axs[0].set_ylim(0.85, 1.0)
    axs[0].set_ylabel("acurácia (CV 5$\\times$3)")
    axs[0].set_title("Acurácia por ativação")
    axs[1].bar(df.ativacao, df.log_loss, color=cores, width=0.6)
    axs[1].set_ylabel("entropia cruzada")
    axs[1].set_title("Perda por ativação")
    fig.tight_layout()
    salvar(fig, "fig_ativacoes.png")

    t = df.rename(columns={'ativacao': 'ativação', 'acuracia': 'acurácia',
                           'log_loss': 'log-loss'}).copy()
    t['acurácia'] = (100 * t['acurácia']).round(1)
    t['desvio'] = (100 * t['desvio']).round(1)
    t['log-loss'] = t['log-loss'].round(3)
    tab_latex(t, "tab_ativacoes", "lrrr")
    df.to_csv(RES / "ativacoes.csv", index=False)
    return df


def estudo_regularizacao(X, y) -> pd.DataFrame:
    print("4) Regularização L2")
    alfas = np.logspace(-6, 2, 17)
    linhas = []
    for a in alfas:
        m = mlp((32, 16), alpha=float(a))
        accs, perdas = [], []
        for tr, te in dobras_repetidas(X, y, repeticoes=3):
            mm = clone(m).fit(X[tr], y[tr])
            accs.append(accuracy_score(y[te], mm.predict(X[te])))
            perdas.append(log_loss(y[te], mm.predict_proba(X[te]), labels=[0, 1, 2]))
        linhas.append(dict(alpha=float(a), acuracia=np.mean(accs),
                           desvio=np.std(accs, ddof=1), log_loss=np.mean(perdas)))
    df = pd.DataFrame(linhas)

    fig, axs = plt.subplots(1, 2, figsize=(10.6, 3.7))
    for ax, col, cor, tit in ((axs[0], "acuracia", "#2a78d6", "Acurácia"),
                              (axs[1], "log_loss", "#eb6834", "Entropia cruzada")):
        ax.errorbar(df.alpha, df[col], yerr=df.desvio if col == "acuracia" else None,
                    fmt="o-", color=cor, ms=3.5, lw=1.4, capsize=2)
        ax.set_xscale("log")
        ax.set_xlabel("$\\alpha$ (força da regularização L2)")
        ax.set_title(tit)
    axs[0].set_ylabel("acurácia (CV 5$\\times$3)")
    axs[1].set_ylabel("perda")
    fig.tight_layout()
    salvar(fig, "fig_regularizacao.png")
    df.to_csv(RES / "regularizacao.csv", index=False)
    return df


def estudo_taxa_aprendizado(X, y) -> pd.DataFrame:
    print("5) Taxa de aprendizado (SGD com momento)")
    taxas = [1e-3, 1e-2, 5e-2, 1e-1, 5e-1]
    fig, axs = plt.subplots(1, 2, figsize=(10.6, 3.7))
    linhas = []
    for i, lr in enumerate(taxas):
        m = make_pipeline(StandardScaler(), MLPClassifier(
            hidden_layer_sizes=(16, 8), solver="sgd", learning_rate_init=lr,
            momentum=0.9, nesterovs_momentum=True, max_iter=300, random_state=SEMENTE))
        m.fit(X, y)
        ativ = m[-1]
        cor = plt.cm.viridis(i / (len(taxas) - 1) * 0.85)
        axs[0].plot(ativ.loss_curve_, color=cor, lw=1.6,
                    label=f"$\\eta_0$ = {lr:g}")
        linhas.append(dict(taxa=lr, perda_final=float(ativ.loss_curve_[-1]),
                           acuracia_treino=accuracy_score(y, m.predict(X))))
    axs[0].set_xlabel("época")
    axs[0].set_ylabel("perda de treino")
    axs[0].set_yscale("log")
    axs[0].legend(fontsize=7, title="taxa inicial", title_fontsize=7)
    axs[0].set_title("Convergência com SGD $+$ momento")
    axs[1].bar([f"{t:g}" for t in taxas], [r["perda_final"] for r in linhas],
               color=[plt.cm.viridis(i / (len(taxas) - 1) * 0.85) for i in range(len(taxas))],
               width=0.6)
    axs[1].set_xlabel("taxa de aprendizado inicial $\\eta_0$")
    axs[1].set_ylabel("perda final (300 épocas)")
    axs[1].set_title("Perda final por taxa de aprendizado")
    fig.tight_layout()
    salvar(fig, "fig_taxas_aprendizado.png")
    df = pd.DataFrame(linhas)
    df.to_csv(RES / "taxas_aprendizado.csv", index=False)
    return df


# --------------------------------------------------------------------------- #
# 4. Treinamento final, curvas e avaliação
# --------------------------------------------------------------------------- #
def curva_perda(Xtr, ytr) -> dict:
    print("6) Curva de treinamento (Adam)")
    m = make_pipeline(StandardScaler(), MLPClassifier(
        hidden_layer_sizes=(16,), activation="tanh", alpha=1e-2, solver="adam",
        learning_rate_init=1e-3, max_iter=1500, early_stopping=True,
        validation_fraction=0.2, n_iter_no_change=60, random_state=SEMENTE))
    m.fit(Xtr, ytr)
    ativ = m[-1]
    fig, axs = plt.subplots(1, 2, figsize=(10.6, 3.7))
    axs[0].plot(ativ.loss_curve_, color="#2a78d6", lw=1.7, label="treino")
    axs[0].set_xlabel("época")
    axs[0].set_ylabel("entropia cruzada")
    axs[0].set_title("Perda de treino")
    axs[1].plot(ativ.validation_scores_, color="#eb6834", lw=1.7, label="validação")
    axs[1].axhline(1.0, color=TINTA2, lw=0.8, ls="--")
    axs[1].set_xlabel("época")
    axs[1].set_ylabel("acurácia")
    axs[1].set_ylim(0.55, 1.02)
    axs[1].set_title("Acurácia de validação (20% do treino)")
    fig.tight_layout()
    salvar(fig, "fig_curva_perda.png")
    return dict(n_iter=int(ativ.n_iter_), perda_final=float(ativ.loss_curve_[-1]),
                parada=bool(ativ.n_iter_ < 1500))


def curva_aprendizado(X, y) -> pd.DataFrame:
    print("7) Curva de aprendizado")
    m = mlp_principal()
    # o Iris está ordenado por espécie: embaralhar antes evita que os primeiros
    # subconjuntos de treino contenham apenas uma ou duas classes
    rng = np.random.default_rng(SEMENTE)
    perm = rng.permutation(len(y))
    Xp, yp = X[perm], y[perm]
    tamanhos, tr_sc, te_sc = learning_curve(
        m, Xp, yp, train_sizes=np.linspace(0.1, 1.0, 10),
        cv=StratifiedKFold(5, shuffle=True, random_state=SEMENTE),
        scoring="accuracy", n_jobs=1)
    df = pd.DataFrame(dict(n= tamanhos, treino=tr_sc.mean(1), treino_dp=tr_sc.std(1),
                           validacao=te_sc.mean(1), validacao_dp=te_sc.std(1)))
    fig, ax = plt.subplots(figsize=(6.6, 3.8))
    ax.plot(df.n, df.treino, "o-", color="#2a78d6", ms=4, lw=1.5, label="treino")
    ax.fill_between(df.n, df.treino - df.treino_dp, df.treino + df.treino_dp,
                    color="#2a78d6", alpha=0.18, lw=0)
    ax.plot(df.n, df.validacao, "o-", color="#eb6834", ms=4, lw=1.5, label="validação")
    ax.fill_between(df.n, df.validacao - df.validacao_dp, df.validacao + df.validacao_dp,
                    color="#eb6834", alpha=0.18, lw=0)
    ax.set_xlabel("número de amostras de treinamento")
    ax.set_ylabel("acurácia")
    ax.set_ylim(0.68, 1.02)
    ax.legend(fontsize=8)
    ax.set_title("Curva de aprendizado (MLP 16, tanh, CV 5 dobras)")
    fig.tight_layout()
    salvar(fig, "fig_curva_aprendizado.png")
    df.to_csv(RES / "curva_aprendizado.csv", index=False)
    return df


def avaliacao_final(X, y, nomes) -> dict:
    print("8) Divisão treino/teste e métricas finais")
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.30, stratify=y,
                                          random_state=SEMENTE)
    modelo = mlp_principal().fit(Xtr, ytr)
    prob = modelo.predict_proba(Xte)
    pred = modelo.predict(Xte)
    acc = accuracy_score(yte, pred)
    lo, hi = wilson(int(acc * len(yte)), len(yte))
    met = dict(n_treino=len(ytr), n_teste=len(yte), acuracia=acc,
               acuracia_bal=balanced_accuracy_score(yte, pred),
               f1_macro=f1_score(yte, pred, average="macro"),
               log_loss=log_loss(yte, prob, labels=[0, 1, 2]),
               ic95=(lo, hi))

    # figura da divisão
    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    for esp, cor in COR.items():
        k = CLASSES.index(esp)
        mtr = ytr == k
        mte = yte == k
        ax.scatter(Xtr[mtr, 2], Xtr[mtr, 3], s=26, color=cor, alpha=0.75,
                   linewidths=0.5, edgecolors="white", label=esp)
        ax.scatter(Xte[mte, 2], Xte[mte, 3], s=44, facecolors="none", edgecolors=cor,
                   linewidths=1.4, marker="s")
    ax.set_xlabel(ROTULO["PL"])
    ax.set_ylabel(ROTULO["PW"])
    ax.set_title(f"Divisão estratificada: {len(ytr)} treino (círculos) e "
                 f"{len(yte)} teste (quadrados)")
    ax.legend(fontsize=8, title="espécie", title_fontsize=8, loc="upper left")
    fig.tight_layout()
    salvar(fig, "fig_divisao_treino_teste.png")

    # matriz de confusão: CV out-of-fold (150) e teste (45)
    oof = cross_val_predict(mlp_principal(), X, y,
                            cv=StratifiedKFold(10, shuffle=True, random_state=SEMENTE))
    cm_cv = confusion_matrix(y, oof)
    cm_te = confusion_matrix(yte, pred)
    fig, axs = plt.subplots(1, 2, figsize=(9.6, 3.9))
    for ax, cm, tit in ((axs[0], cm_cv, f"Validação cruzada 10 dobras ($n$ = {len(y)})"),
                        (axs[1], cm_te, f"Conjunto de teste ($n$ = {len(yte)})")):
        im = ax.imshow(cm, cmap="Blues", vmin=0, vmax=cm.max())
        ax.grid(False)
        for i in range(3):
            for j in range(3):
                ax.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=11,
                        color="white" if cm[i, j] > 0.55 * cm.max() else TINTA)
        ax.set_xticks(range(3), CLASSES, rotation=20)
        ax.set_yticks(range(3), CLASSES)
        ax.set_xlabel("previsto")
        ax.set_ylabel("real")
        ax.set_title(tit)
    fig.tight_layout()
    salvar(fig, "fig_confusao.png")

    pd.DataFrame(cm_cv, index=CLASSES, columns=CLASSES).to_csv(RES / "matriz_confusao_cv.csv")
    pd.DataFrame(cm_te, index=CLASSES, columns=CLASSES).to_csv(RES / "matriz_confusao_teste.csv")

    # tabela da matriz de confusão (CV)
    t = pd.DataFrame(cm_cv, index=CLASSES, columns=CLASSES)
    t.insert(0, "real $\\backslash$ previsto", CLASSES)
    tab_latex(t, "tab_confusao", "lccc")

    # métricas por classe (CV out-of-fold)
    P, R, F, S = precision_recall_fscore_support(y, oof, labels=[0, 1, 2])
    t = pd.DataFrame(dict(Classe=CLASSES, Suporte=S,
                          **{'Precisão': (100 * P).round(1),
                             'Revocação': (100 * R).round(1),
                             'F1': (100 * F).round(1)}))
    tab_latex(t, "tab_por_classe", "lrrrr")
    t2 = pd.DataFrame(dict(classe=CLASSES, precisao=P, revocacao=R, f1=F, suporte=S))
    t2.to_csv(RES / "metricas_por_classe_cv.csv", index=False)

    fig, ax = plt.subplots(figsize=(7.4, 3.5))
    w = 0.26
    x = np.arange(3)
    for i, (v, rot, cor) in enumerate(((P, "precisão", "#2a78d6"),
                                       (R, "revocação", "#eb6834"),
                                       (F, "F1", "#1baf7a"))):
        ax.bar(x + (i - 1) * w, v, width=w, color=cor, label=rot)
        for xi, vi in zip(x + (i - 1) * w, v):
            ax.text(xi, vi + 0.01, br(vi, 2), ha="center", fontsize=7, color=TINTA2)
    ax.set_xticks(x, CLASSES)
    ax.set_ylim(0, 1.16)
    ax.set_ylabel("valor")
    ax.legend(fontsize=8, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.02))
    ax.set_title("Precisão, revocação e F1 por classe (validação cruzada)")
    fig.tight_layout()
    salvar(fig, "fig_metricas_por_classe.png")

    # relatório de classificação no teste
    rep = pd.DataFrame(
        precision_recall_fscore_support(yte, pred, labels=[0, 1, 2]),
        index=["precisao", "revocacao", "f1", "suporte"]).T
    rep.index = CLASSES
    rep.to_csv(RES / "relatorio_teste.csv")
    return dict(modelo=modelo, Xtr=Xtr, Xte=Xte, ytr=ytr, yte=yte, pred=pred,
                prob=prob, met=met, oof=oof, cm_cv=cm_cv, cm_te=cm_te)


# --------------------------------------------------------------------------- #
# 5. Fronteiras de decisão, importância e pesos
# --------------------------------------------------------------------------- #
def fronteira(ax, modelo, X2, y, resolucao=300, alpha=0.18):
    x0, x1 = X2[:, 0].min() - 0.4, X2[:, 0].max() + 0.4
    y0, y1 = X2[:, 1].min() - 0.4, X2[:, 1].max() + 0.4
    xx, yy = np.meshgrid(np.linspace(x0, x1, resolucao), np.linspace(y0, y1, resolucao))
    Z = modelo.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
    ax.contourf(xx, yy, Z, levels=[-0.5, 0.5, 1.5, 2.5], cmap=CMAP3, alpha=alpha)
    try:
        ax.contour(xx, yy, Z, levels=[0.5, 1.5], colors="white", linewidths=0.7, alpha=0.9)
    except ValueError:
        pass
    for k, esp in enumerate(CLASSES):
        m = y == k
        ax.scatter(X2[m, 0], X2[m, 1], s=14, color=COR[esp], edgecolors="white",
                   linewidths=0.35, zorder=3, label=esp)


def figuras_2d(X, y) -> None:
    print("9) Fronteiras de decisão (2 atributos: pétala)")
    idx = [2, 3]
    X2 = X[:, idx]
    Xtr, Xte, ytr, yte = train_test_split(X2, y, test_size=0.30, stratify=y,
                                          random_state=SEMENTE)

    modelos = {
        "MLP": mlp_principal(),
        "RF": modelos_referencia()["RF"],
        "GB": modelos_referencia()["GB"],
        "LR": modelos_referencia()["LR"],
    }
    fig, axs = plt.subplots(2, 2, figsize=(9.6, 8.2))
    for ax, (nome, m) in zip(axs.flat, modelos.items()):
        mm = clone(m).fit(Xtr, ytr)
        ac = mm.score(Xte, yte)
        fronteira(ax, mm, X2, y)
        ax.set_title(f"{nome} — acurácia no teste: {ac:.1%}")
        ax.set_xlabel(ROTULO["PL"])
        ax.set_ylabel(ROTULO["PW"])
    axs[0, 0].legend(fontsize=7, loc="upper left", markerscale=1.4)
    fig.suptitle("Regiões de decisão usando apenas comprimento e largura da pétala",
                 fontsize=11)
    fig.tight_layout()
    salvar(fig, "fig_fronteira_comparacao.png")

    # capacidade do modelo
    fig, axs = plt.subplots(2, 2, figsize=(9.6, 8.2))
    for ax, h in zip(axs.flat, [(2,), (4,), (16, 8), (64, 32, 16)]):
        mm = clone(mlp(h)).fit(Xtr, ytr)
        fronteira(ax, mm, X2, y)
        ax.set_title(f"MLP {h} — teste: {mm.score(Xte, yte):.1%}")
        ax.set_xlabel(ROTULO["PL"])
        ax.set_ylabel(ROTULO["PW"])
    axs[0, 0].legend(fontsize=7, loc="upper left", markerscale=1.4)
    fig.suptitle("Efeito da capacidade da rede na fronteira de decisão", fontsize=11)
    fig.tight_layout()
    salvar(fig, "fig_fronteira_capacidade.png")

    # fronteira única, maior, do modelo principal
    fig, ax = plt.subplots(figsize=(6.6, 4.9))
    mm = clone(mlp_principal()).fit(Xtr, ytr)
    fronteira(ax, mm, X2, y, alpha=0.22)
    ax.set_xlabel(ROTULO["PL"])
    ax.set_ylabel(ROTULO["PW"])
    ax.set_title(f"MLP 16 (tanh): fronteiras no plano da pétala "
                 f"(acurácia no teste: {mm.score(Xte, yte):.1%})")
    ax.legend(fontsize=8, loc="upper left")
    fig.tight_layout()
    salvar(fig, "fig_fronteira_mlp.png")

    # importância por permutação (modelo de 4 atributos, no teste)
    Xtr4, Xte4, ytr4, yte4 = train_test_split(X, y, test_size=0.30, stratify=y,
                                              random_state=SEMENTE)
    m4 = mlp_principal().fit(Xtr4, ytr4)
    r = permutation_importance(m4, Xte4, yte4, n_repeats=40, random_state=SEMENTE,
                               scoring="accuracy")
    ordem = np.argsort(r.importances_mean)
    fig, ax = plt.subplots(figsize=(6.4, 3.3))
    ax.barh(np.arange(4), r.importances_mean[ordem], xerr=r.importances_std[ordem],
            color="#2a78d6", height=0.6, capsize=3, error_kw={"elinewidth": 0.9})
    ax.set_yticks(np.arange(4), [CURTO[ATRIBUTOS[i]] for i in ordem])
    ax.set_xlabel("queda de acurácia ao permutar o atributo")
    ax.set_title("Importância por permutação (MLP, conjunto de teste, 40 repetições)")
    fig.tight_layout()
    salvar(fig, "fig_importancia.png")
    imp = pd.DataFrame(dict(atributo=[ATRIBUTOS[i] for i in np.argsort(-r.importances_mean)],
                            importancia=r.importances_mean[np.argsort(-r.importances_mean)],
                            desvio=r.importances_std[np.argsort(-r.importances_mean)]))
    imp.to_csv(RES / "importancia.csv", index=False)


def figura_pesos(X, y) -> None:
    print("10) Pesos aprendidos")
    m = mlp((8,), max_iter=4000).fit(X, y)
    ativ = m[-1]
    W1, W2 = ativ.coefs_[0], ativ.coefs_[1]
    fig, axs = plt.subplots(1, 2, figsize=(9.8, 3.4),
                            gridspec_kw={"width_ratios": [1.35, 1]})
    for ax, W, xt, yt, tit in (
            (axs[0], W1, [f"h{i+1}" for i in range(8)], [CURTO[a] for a in ATRIBUTOS],
             "Camada de entrada $\\rightarrow$ oculta ($W^{(1)}$)"),
            (axs[1], W2, CLASSES, [f"h{i+1}" for i in range(8)],
             "Camada oculta $\\rightarrow$ saída ($W^{(2)}$)")):
        v = np.abs(W).max()
        im = ax.imshow(W, cmap="RdBu_r", vmin=-v, vmax=v, aspect="auto")
        ax.set_xticks(range(W.shape[1]), xt, fontsize=7,
                      rotation=45 if W.shape[1] == 8 else 0,
                      ha="right" if W.shape[1] == 8 else "center")
        ax.set_yticks(range(W.shape[0]), yt, fontsize=7)
        for i in range(W.shape[0]):
            for j in range(W.shape[1]):
                v_ij = W[i, j]
                txt = "0" if abs(v_ij) < 0.05 else br(v_ij, 1)
                ax.text(j, i, txt, ha="center", va="center", fontsize=6,
                        color="white" if abs(v_ij) > 0.6 * v else TINTA)
        ax.grid(False)
        ax.set_title(tit, fontsize=9)
        fig.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
    fig.tight_layout()
    salvar(fig, "fig_pesos.png")

    t = pd.DataFrame(W1, index=ATRIBUTOS, columns=[f"h{i+1}" for i in range(8)]).reset_index()
    t = t.rename(columns={"index": "atributo"})
    tab_latex(t, "tab_pesos", "l" + "r" * 8, decimais=2)


# --------------------------------------------------------------------------- #
# 6. Implementação manual em NumPy
# --------------------------------------------------------------------------- #
def _softmax(z):
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def _onehot(y, k=3):
    Y = np.zeros((len(y), k))
    Y[np.arange(len(y)), y] = 1.0
    return Y


class MLPNumPy:
    """MLP de uma camada oculta: ReLU (ou tanh) na oculta e softmax na saída.

    A perda é a entropia cruzada categórica com regularização L2:

        L = -1/m * sum_i sum_k y_ik log p_ik + (lambda/2) * (||W1||^2 + ||W2||^2)

    A retropropagação usa  delta^(2) = p - y  e
    delta^(1) = (delta^(2) W2^T) * f'(z1).
    """

    def __init__(self, n_in=4, n_hid=8, n_out=3, alpha=1e-4, semente=0, ativacao="relu"):
        rng = np.random.default_rng(semente)
        self.W1 = rng.normal(0, np.sqrt(2.0 / n_in), (n_in, n_hid))
        self.b1 = np.zeros(n_hid)
        self.W2 = rng.normal(0, np.sqrt(2.0 / n_hid), (n_hid, n_out))
        self.b2 = np.zeros(n_out)
        self.alpha = alpha
        self.ativacao = ativacao
        self.hist = {"perda": [], "acuracia": []}

    def _f(self, z):
        return np.maximum(z, 0) if self.ativacao == "relu" else np.tanh(z)

    def _df(self, z):
        return (z > 0).astype(float) if self.ativacao == "relu" else 1 - np.tanh(z) ** 2

    def forward(self, X):
        z1 = X @ self.W1 + self.b1
        a1 = self._f(z1)
        z2 = a1 @ self.W2 + self.b2
        return z1, a1, z2, _softmax(z2)

    def perda(self, X, y):
        _, _, _, p = self.forward(X)
        Y = _onehot(y)
        ce = -np.mean(np.sum(Y * np.log(p + 1e-12), axis=1))
        reg = 0.5 * self.alpha * (np.sum(self.W1 ** 2) + np.sum(self.W2 ** 2))
        return ce + reg

    def backward(self, X, y):
        m = len(y)
        z1, a1, _, p = self.forward(X)
        Y = _onehot(y)
        d2 = (p - Y) / m                      # (m, 3)
        gW2 = a1.T @ d2 + self.alpha * self.W2
        gb2 = d2.sum(0)
        d1 = (d2 @ self.W2.T) * self._df(z1)   # (m, h)
        gW1 = X.T @ d1 + self.alpha * self.W1
        gb1 = d1.sum(0)
        return {"W1": gW1, "b1": gb1, "W2": gW2, "b2": gb2}

    def fit(self, X, y, epocas=500, lote=16, lr=0.05, b1=0.9, b2=0.999, eps=1e-8,
            semente=0, verboso=False):
        rng = np.random.default_rng(semente)
        params = {"W1": self.W1, "b1": self.b1, "W2": self.W2, "b2": self.b2}
        m = {k: np.zeros_like(v) for k, v in params.items()}
        v = {k: np.zeros_like(v) for k, v in params.items()}
        t = 0
        for ep in range(epocas):
            ordem = rng.permutation(len(y))
            for i in range(0, len(y), lote):
                idx = ordem[i:i + lote]
                g = self.backward(X[idx], y[idx])
                t += 1
                for k in params:
                    m[k] = b1 * m[k] + (1 - b1) * g[k]
                    v[k] = b2 * v[k] + (1 - b2) * g[k] ** 2
                    mh = m[k] / (1 - b1 ** t)
                    vh = v[k] / (1 - b2 ** t)
                    params[k] -= lr * mh / (np.sqrt(vh) + eps)
            self.W1, self.b1, self.W2, self.b2 = (params["W1"], params["b1"],
                                                  params["W2"], params["b2"])
            self.hist["perda"].append(self.perda(X, y))
            self.hist["acuracia"].append(accuracy_score(y, self.predict(X)))
        return self

    def predict_proba(self, X):
        return self.forward(X)[3]

    def predict(self, X):
        return self.predict_proba(X).argmax(1)


def checagem_gradiente(X, y, semente=0) -> pd.DataFrame:
    print("11) Verificação numérica do gradiente")
    rede = MLPNumPy(n_hid=5, alpha=1e-3, semente=semente, ativacao="tanh")
    g = rede.backward(X, y)
    eps = 1e-6
    linhas = []
    rng = np.random.default_rng(3)
    for nome in ("W1", "b1", "W2", "b2"):
        P = getattr(rede, nome)
        idxs = [tuple(rng.integers(0, s) for s in P.shape) for _ in range(60)] \
            if P.ndim == 2 else [(i,) for i in rng.choice(P.size, min(8, P.size), replace=False)]
        for ix in idxs:
            orig = P[ix]
            P[ix] = orig + eps
            lp = rede.perda(X, y)
            P[ix] = orig - eps
            lm = rede.perda(X, y)
            P[ix] = orig
            num = (lp - lm) / (2 * eps)
            ana = g[nome][ix]
            linhas.append(dict(parametro=nome, analitico=ana, numerico=num,
                               erro_relativo=abs(ana - num) / max(abs(ana) + abs(num), 1e-12)))
    df = pd.DataFrame(linhas)

    fig, axs = plt.subplots(1, 2, figsize=(10.2, 3.8))
    axs[0].loglog(np.abs(df.numerico) + 1e-12, np.abs(df.analitico) + 1e-12, "o",
                  ms=3.5, color="#2a78d6", alpha=0.85)
    lim = [1e-8, 1e-1]
    axs[0].plot(lim, lim, "--", color=TINTA2, lw=0.9, label="identidade")
    axs[0].set_xlabel("gradiente numérico (diferenças centrais)")
    axs[0].set_ylabel("gradiente analítico (retropropagação)")
    axs[0].set_title("Gradiente analítico $\\times$ numérico")
    axs[0].legend(fontsize=8)
    axs[1].hist(np.log10(df.erro_relativo), bins=18, color="#1baf7a", edgecolor="white")
    axs[1].set_xlabel("$\\log_{10}$ do erro relativo $|a-n| / (|a|+|n|)$")
    axs[1].set_ylabel("ocorrências")
    axs[1].set_title("Distribuição do erro relativo")
    fig.tight_layout()
    salvar(fig, "fig_gradiente.png")

    tab_latex(pd.DataFrame([
        {"parâmetro": nome, "amostras": int((df.parametro == nome).sum()),
         "erro máximo": cient(float(df[df.parametro == nome].erro_relativo.max())),
         "erro médio": cient(float(df[df.parametro == nome].erro_relativo.mean()))}
        for nome in ("W1", "b1", "W2", "b2")]), "tab_gradiente", "lrrr")
    df.to_csv(RES / "checagem_gradiente.csv", index=False)
    return df


def figura_numpy(X, y) -> dict:
    print("12) Implementação em NumPy $\\times$ scikit-learn")
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.30, stratify=y,
                                          random_state=SEMENTE)
    esc = StandardScaler().fit(Xtr)
    Ztr, Zte = esc.transform(Xtr), esc.transform(Xte)

    rede = MLPNumPy(n_hid=8, alpha=1e-4, semente=SEMENTE, ativacao="relu")
    rede.fit(Ztr, ytr, epocas=400, lote=16, lr=0.03)
    ac_np = accuracy_score(yte, rede.predict(Zte))

    sk = mlp((8,), max_iter=4000).fit(Xtr, ytr)
    ac_sk = accuracy_score(yte, sk.predict(Xte))

    fig, axs = plt.subplots(1, 2, figsize=(11.0, 3.9))
    axs[0].plot(rede.hist["perda"], color="#2a78d6", lw=1.7, label="NumPy (400 épocas)")
    axs[0].set_xlabel("época")
    axs[0].set_ylabel("perda de treino")
    axs[0].set_title(f"Curva de perda da implementação manual")
    axs[0].legend(fontsize=8)
    axs[1].bar(["NumPy", "scikit-learn"], [ac_np, ac_sk], color=["#2a78d6", "#eb6834"],
               width=0.55)
    for i, v in enumerate([ac_np, ac_sk]):
        axs[1].text(i, v + 0.005, f"{v:.1%}", ha="center", fontsize=9)
    axs[1].set_ylim(0.7, 1.05)
    axs[1].set_ylabel("acurácia no teste")
    axs[1].set_title("Mesma arquitetura (8 neurônios)")
    fig.tight_layout()
    salvar(fig, "fig_implementacao.png")
    return dict(acuracia_numpy=ac_np, acuracia_sklearn=ac_sk)


def figura_escala(X, y) -> pd.DataFrame:
    print("13) Efeito da padronização")
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.30, stratify=y,
                                          random_state=SEMENTE)

    def base(solver, lr):
        return MLPClassifier(hidden_layer_sizes=(16,), activation="tanh", alpha=1e-2,
                             solver=solver, learning_rate_init=lr, momentum=0.9,
                             max_iter=800, tol=1e-5, random_state=SEMENTE)

    variantes = [
        ("Adam", "padronizado", make_pipeline(StandardScaler(), base("adam", 1e-3))),
        ("Adam", "sem padronização", base("adam", 1e-3)),
        ("SGD", "padronizado", make_pipeline(StandardScaler(), base("sgd", 5e-2))),
        ("SGD", "sem padronização", base("sgd", 5e-2)),
    ]
    cores = {"Adam": "#2a78d6", "SGD": "#eb6834"}
    estilos = {"padronizado": "-", "sem padronização": "--"}
    linhas = []
    fig, ax = plt.subplots(figsize=(7.0, 3.8))
    for opt, esc, m in variantes:
        m.fit(Xtr, ytr)
        ac = accuracy_score(yte, m.predict(Xte))
        ativ = m.steps[-1][1] if hasattr(m, "steps") else m
        ax.plot(ativ.loss_curve_, color=cores[opt], ls=estilos[esc], lw=1.6,
                label=f"{opt}, {esc} (teste {ac:.1%})")
        linhas.append(dict(otimizador=opt, padronizacao=esc, acuracia=ac,
                           epocas=int(ativ.n_iter_),
                           perda_final=float(ativ.loss_curve_[-1])))
    ax.set_xlabel("época")
    ax.set_ylabel("perda de treino")
    ax.set_yscale("log")
    ax.legend(fontsize=7.5)
    ax.set_title("Convergência com e sem padronização: Adam versus SGD com momento")
    fig.tight_layout()
    salvar(fig, "fig_escala.png")
    df = pd.DataFrame(linhas)
    df.to_csv(RES / "escala.csv", index=False)
    return df


# --------------------------------------------------------------------------- #
# 7. Relatório automático
# --------------------------------------------------------------------------- #
def relatorio(res: dict, arq, ativ, reg, cmp_df, apr, grad, npres, esc, testes) -> dict:
    m = res["met"]
    melhor_arq = arq.loc[arq.acuracia.idxmax()]
    melhor_ativ = ativ.loc[ativ.acuracia.idxmax()]
    N = dict(
        accMLP=cmp_df.loc[cmp_df.modelo == "MLP", "acuracia"].iloc[0],
        accRF=cmp_df.loc[cmp_df.modelo == "RF", "acuracia"].iloc[0],
        accGB=cmp_df.loc[cmp_df.modelo == "GB", "acuracia"].iloc[0],
        accLR=cmp_df.loc[cmp_df.modelo == "LR", "acuracia"].iloc[0],
        sdMLP=cmp_df.loc[cmp_df.modelo == "MLP", "desvio"].iloc[0],
        sdLR=cmp_df.loc[cmp_df.modelo == "LR", "desvio"].iloc[0],
        accTeste=m["acuracia"], fmacroTeste=m["f1_macro"], llteste=m["log_loss"],
        icLo=m["ic95"][0], icHi=m["ic95"][1],
        nTreino=m["n_treino"], nTeste=m["n_teste"],
        melhorArq=melhor_arq.arquitetura, melhorArqAcc=melhor_arq.acuracia,
        melhorParam=int(melhor_arq.parametros), melhorAtiv=melhor_ativ.ativacao,
        melhorAtivAcc=melhor_ativ.acuracia,
        alphaOtimo=float(reg.loc[reg.log_loss.idxmin(), "alpha"]),
        nIter=apr["n_iter"], perdaFinal=apr["perda_final"],
        accNumpy=npres["acuracia_numpy"], accSklearn=npres["acuracia_sklearn"],
        accComEscala=float(esc.loc[(esc.otimizador == "Adam") &
                                   (esc.padronizacao == "padronizado"), "acuracia"].iloc[0]),
        accSemEscala=float(esc.loc[(esc.otimizador == "Adam") &
                                   (esc.padronizacao == "sem padronização"), "acuracia"].iloc[0]),
        parametrosMLP=n_parametros(res["modelo"][-1]),
        pLR=float(testes.loc[testes.comparacao == "MLP vs LR", "p"].iloc[0]),
        pRF=float(testes.loc[testes.comparacao == "MLP vs RF", "p"].iloc[0]),
        pGB=float(testes.loc[testes.comparacao == "MLP vs GB", "p"].iloc[0]),
    )

    # macros LaTeX
    defs = []
    for k, v in N.items():
        if isinstance(v, str):
            defs.append(r"\newcommand{\%s}{%s}" % (k, v))
        elif isinstance(v, (int, np.integer)):
            defs.append(r"\newcommand{\%s}{%d}" % (k, int(v)))
        else:
            defs.append(r"\newcommand{\%s}{%s}" % (k, brl(float(v), 3)))
    # o erro da checagem de gradiente é muito pequeno para 3 decimais
    v = float(grad.erro_relativo.max())
    if v > 0:
        e = int(np.floor(np.log10(abs(v))))
        mant = v / 10 ** e
        defs.append(r"\newcommand{\erroGrad}{$%s\times10^{%d}$}" % (brl(mant, 1), e))
    else:
        defs.append(r"\newcommand{\erroGrad}{$0$}")
    (BASE / "artigo" / "numeros.tex").write_text("\n".join(defs) + "\n", encoding="utf-8")

    # resumo em Markdown
    L = ["# Relatório automático - MLP no Iris\n",
         f"Gerado em {pd.Timestamp.now():%Y-%m-%d %H:%M}. "
         f"CV = {REPETICOES} x {DOBRAS} dobras estratificadas; semente = {SEMENTE}.\n",
         "## Comparação de modelos (validação cruzada)\n",
         cmp_df.round(4).to_markdown(index=False),
         "\n## Arquiteturas\n", arq.round(4).to_markdown(index=False),
         "\n## Ativações\n", ativ.round(4).to_markdown(index=False),
         "\n## Divisão treino/teste\n",
         f"- treino: {m['n_treino']}, teste: {m['n_teste']}",
         f"- acurácia: {br(m['acuracia'])} (IC95 Wilson: {br(m['ic95'][0])} a {br(m['ic95'][1])})",
         f"- F1 macro: {br(m['f1_macro'])}; entropia cruzada: {br(m['log_loss'])}",
         f"- verificação do gradiente: erro relativo máximo {grad.erro_relativo.max():.2e}",
         f"- NumPy: {br(npres['acuracia_numpy'])} x scikit-learn: {br(npres['acuracia_sklearn'])}"]
    (RES / "resumo.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    return N


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sem-numpy", action="store_true", help="pula a implementação manual")
    args = ap.parse_args()
    t0 = time.time()

    df, X, y, _ = carregar()
    print(f"Iris: {X.shape[0]} amostras, {X.shape[1]} atributos, 3 classes")
    eda(df, y)

    arq = estudo_arquiteturas(X, y)
    ativ = estudo_ativacoes(X, y)
    reg = estudo_regularizacao(X, y)
    taxas = estudo_taxa_aprendizado(X, y)

    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.30, stratify=y,
                                          random_state=SEMENTE)
    apr = curva_perda(Xtr, ytr)
    apr_df = curva_aprendizado(X, y)

    res = avaliacao_final(X, y, CLASSES)
    figuras_2d(X, y)
    figura_pesos(X, y)

    if not args.sem_numpy:
        grad = checagem_gradiente(Xtr, ytr)
        npres = figura_numpy(X, y)
    else:
        grad = pd.DataFrame(dict(parametro=[], erro_relativo=[np.nan]))
        npres = dict(acuracia_numpy=np.nan, acuracia_sklearn=np.nan)

    esc = figura_escala(X, y)

    cmp_df, accs = cv_emparelhada(modelos_referencia(), X, y)
    cmp_df.to_csv(RES / "comparacao_modelos.csv", index=False)

    # teste de Wilcoxon para amostras pareadas: mesmas 25 dobras para todos
    linhas_t = []
    for k in ("RF", "GB", "LR"):
        st = stats.wilcoxon(accs["MLP"], accs[k])
        linhas_t.append(dict(comparacao=f"MLP vs {k}",
                             estatistica_W=float(st.statistic), p=float(st.pvalue),
                             diferenca=float(np.mean(np.array(accs["MLP"]) - np.array(accs[k])))))
    testes = pd.DataFrame(linhas_t)
    testes.to_csv(RES / "testes_estatisticos.csv", index=False)
    tt = pd.DataFrame({
        "comparação": [v.replace(" vs ", r" $\times$ ") for v in testes.comparacao],
        "estatística W": [br(float(v), 1) for v in testes.estatistica_W],
        "valor p": [brl(v, 3) if v >= 1e-3 else "$<$ 0,001" for v in testes.p],
        "diferença (p.p.)": [br(float(100 * v), 2) for v in testes.diferenca],
    })
    tab_latex(tt, "tab_testes", "lrrr")
    t = cmp_df.drop(columns=["f1_desvio"]).copy()
    t.columns = ["modelo", "acurácia (%)", "desvio (p.p.)", "F1 macro (%)", "tempo (s)"]
    for c in t.columns[1:]:
        t[c] = (100 * t[c]).round(1) if "(%)" in c or "p.p." in c else t[c].round(1)
    tab_latex(t, "tab_modelos", "lrrrr")

    fig, axs = plt.subplots(1, 2, figsize=(10.6, 3.7))
    for ax, col, dp, tit in ((axs[0], "acuracia", "desvio", "Acurácia"),
                             (axs[1], "f1_macro", "f1_desvio", "F1 macro")):
        ordem = cmp_df.sort_values(col, ascending=False)
        ax.bar(ordem.modelo, ordem[col], yerr=ordem[dp], capsize=4,
               color=[COR_MOD[k] for k in ordem.modelo], width=0.6,
               error_kw={"elinewidth": 1})
        ax.set_ylim(0.80, 1.0)
        ax.set_title(tit)
        ax.set_ylabel(f"{tit.lower()} (CV {REPETICOES}$\\times${DOBRAS})")
    fig.tight_layout()
    salvar(fig, "fig_comparacao_modelos.png")

    N = relatorio(res, arq, ativ, reg, cmp_df, apr, grad, npres, esc, testes)

    with pd.option_context("display.width", 200):
        print("\n=== Comparação de modelos ===")
        print(cmp_df.round(4).to_string(index=False))
    print(f"\nConcluído em {time.time() - t0:.0f} s.")
    print(json.dumps({k: (round(v, 4) if isinstance(v, float) else v)
                      for k, v in N.items()}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
