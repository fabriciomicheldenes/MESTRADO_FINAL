# -*- coding: utf-8 -*-
"""
figuras_artigo.py - Figuras complementares para o artigo e para o relatório de
configuração/uso. Lê as saídas de mlp_ea.py (resultados/ e resultados_sensib/)
e os hits brutos em Dados/. Também roda dois experimentos curtos:

  * validação interna × teste: escore da validação interna do early stopping da MLP
    (15 % de janelas sorteadas dentro do treino) comparado ao escore no fold de teste;
  * holdout fora do tempo (OOT): treina com as séries de 05/11 e testa nas de 06/11
    (e o inverso), só para a frequência.

Uso:  .venv\\Scripts\\python.exe figuras_artigo.py   (depois de mlp_ea.py)
Saída: artigo/figuras/*.png e artigo/figuras/*.csv
"""
from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Patch, Rectangle  # noqa: E402
from sklearn.base import clone  # noqa: E402
from sklearn.decomposition import PCA  # noqa: E402
from sklearn.impute import SimpleImputer  # noqa: E402
from sklearn.metrics import (accuracy_score, confusion_matrix,  # noqa: E402
                             mean_absolute_error, r2_score)
from sklearn.preprocessing import StandardScaler  # noqa: E402

import mlp_ea as M  # noqa: E402

AQUI = Path(__file__).resolve().parent
RES = AQUI / "resultados"
SEN = AQUI / "resultados_sensib"
FIG = AQUI / "artigo" / "figuras"
FIG.mkdir(parents=True, exist_ok=True)

FREQS, CONFIGS = M.FREQS, M.CONFIGS
COR_MOD = M.COR_MODELO
COR_CFG = dict(zip(CONFIGS, ["#2a78d6", "#eb6834", "#1baf7a", "#eda100",
                             "#e87ba4", "#008300", "#6250d6"]))
COR_CH = {"A": "#2a78d6", "B": "#eb6834"}
C_TREINO, C_VAL, C_TESTE, C_GAP = "#2a78d6", "#eda100", "#e34948", "#c9c8c2"
TINTA, TINTA2 = M.TINTA, M.TINTA2


def salvar(fig, nome):
    fig.savefig(FIG / nome)
    plt.close(fig)
    print("  ", nome)


# --------------------------------------------------------------------------- #
def carregar():
    df = pd.read_csv(RES / "janelas_descritores.csv")
    cor = pd.read_csv(RES / "corridas.csv", parse_dates=["inicio"])
    pred = pd.read_csv(RES / "predicoes.csv")
    met = pd.read_csv(RES / "metricas.csv")
    hits = []
    for _, c in cor.iterrows():
        h = pd.read_csv(AQUI / "Dados" / c.arquivo,
                        usecols=["rx_time_s", "channel", "amp_dbae", "duration_us", "sat", "maxdur"])
        h["run"], h["freq"], h["config"] = c.run, c.freq, c.config
        hits.append(h)
    hits = pd.concat(hits, ignore_index=True)
    dur = hits.groupby("run").rx_time_s.max()
    cor["dur_s"] = cor.run.map(dur)
    return df, cor, pred, met, hits


# --------------------------------------------------------------------------- #
# Dados e qualidade
# --------------------------------------------------------------------------- #
def fig_ordem_temporal(cor):
    cores_thr = {"63/50": "#6250d6", "65/55": "#eb6834", "50/50": "#2a78d6"}
    cor = cor.assign(thr=cor.thr_a.round().astype(int).astype(str) + "/" +
                     cor.thr_b.round().astype(int).astype(str))
    dias = sorted(cor.inicio.dt.date.unique())
    fig, axs = plt.subplots(1, len(dias), figsize=(12, 4.2), sharey=True)
    ordem_series = list(dict.fromkeys(cor.sort_values("inicio").config))
    for ax, dia in zip(axs, dias):
        g = cor[cor.inicio.dt.date == dia]
        for _, c in g.iterrows():
            y = ordem_series.index(c.config)
            h0 = c.inicio.hour + c.inicio.minute / 60 + c.inicio.second / 3600
            ax.add_patch(Rectangle((h0, y - 0.32), c.dur_s / 3600, 0.64,
                                   color=cores_thr[c.thr], lw=0))
            ax.text(h0 + c.dur_s / 7200, y + 0.38, f"{c.freq}", ha="center", va="bottom",
                    fontsize=6, color=TINTA, rotation=90)
        ax.set_xlim(g.inicio.dt.hour.min() - 0.2, (g.inicio + pd.to_timedelta(g.dur_s, "s")).dt.hour.max() + 1)
        ax.set_title(f"{pd.Timestamp(dia):%d/%m/%Y}")
        ax.set_xlabel("hora do dia")
        ax.grid(axis="y", visible=False)
    axs[0].set_yticks(range(len(ordem_series)), ordem_series)
    axs[0].set_ylim(-0.7, len(ordem_series) + 0.1)
    axs[0].set_ylabel("série de válvulas (rec-suc %)")
    fig.legend(handles=[Patch(color=v, label=f"limiar A/B = {k} dBAE") for k, v in cores_thr.items()],
               loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.04))
    fig.suptitle("Ordem temporal das 56 corridas (número = frequência do motor, Hz)", fontsize=10)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    salvar(fig, "fig_ordem_temporal.png")


def fig_exemplo_hits(hits, df):
    runs = ["100-25@20Hz", "100-100@80Hz"]
    fig, axs = plt.subplots(2, 2, figsize=(12, 5.4), sharex=True)
    for j, r in enumerate(runs):
        h = hits[(hits.run == r) & (hits.rx_time_s.between(60, 70))]
        for i, (ch, nome) in enumerate([(0, "A"), (1, "B")]):
            ax = axs[i, j]
            hh = h[h.channel == ch]
            ax.vlines(hh.rx_time_s, 0, hh.amp_dbae, color=COR_CH[nome], lw=1.2)
            mx = hh[hh.maxdur == 1]
            ax.scatter(mx.rx_time_s, mx.amp_dbae, s=10, color=TINTA, zorder=3,
                       label="hit com MAXDUR (100 ms)")
            for t in np.arange(60, 71, 5):
                ax.axvline(t, color=TINTA2, lw=0.6, ls=":")
            ax.set_ylim(40, 108)
            ax.axhline(102.8, color=TINTA2, lw=0.8, ls="--")
            ax.set_ylabel(f"canal {nome}\namplitude (dBAE)")
            if i == 0:
                ax.set_title(f"corrida {r}")
        axs[1, j].set_xlabel("tempo de chegada (s); linhas pontilhadas = janelas de 5 s")
    axs[0, 1].legend(loc="lower right", fontsize=7)
    axs[0, 0].text(60.3, 103.5, "saturação do ADC", fontsize=7, color=TINTA2)
    fig.suptitle("Trecho de 10 s de hits: 100-25 a 20 Hz (sem saturação) × 100-100 a 80 Hz (saturado)", fontsize=10)
    fig.tight_layout()
    salvar(fig, "fig_exemplo_hits.png")


def fig_intervalo_hits(hits):
    fig, axs = plt.subplots(1, 2, figsize=(11, 3.5), sharey=True)
    bins = np.logspace(-2, 0, 120)
    for ax, (ch, nome) in zip(axs, [(0, "A"), (1, "B")]):
        h = hits[hits.channel == ch].sort_values(["run", "rx_time_s"])
        dt = h.groupby("run").rx_time_s.diff().dropna()
        prev_mx = h.groupby("run").maxdur.shift(1).loc[dt.index]
        ax.hist(dt[prev_mx == 0], bins=bins, color=COR_CH[nome], alpha=0.85, label="após hit normal")
        ax.hist(dt[prev_mx == 1], bins=bins, color=TINTA2, alpha=0.7, label="após hit com MAXDUR")
        ax.set_xscale("log")
        ax.set_xlim(0.01, 1)
        for t in (0.1, 0.2):
            ax.axvline(t, color=TINTA, lw=0.8, ls="--")
        ax.set_title(f"canal {nome}: intervalo entre hits consecutivos")
        ax.set_xlabel("intervalo (s, escala log)")
        ax.legend(fontsize=7, loc="upper left")
    axs[0].set_ylabel("número de hits")
    fig.suptitle("Cadência dos hits: picos em 0,1 s e 0,2 s (tempo morto do rearme), todas as corridas",
                 fontsize=10)
    fig.tight_layout()
    salvar(fig, "fig_intervalo_hits.png")


def fig_amplitude(hits):
    fig, axs = plt.subplots(1, 2, figsize=(11, 3.5), sharey=True)
    bins = np.arange(45, 105, 0.5)
    grupos = [(20, 30, "20-30 Hz", "#9ec5ef"), (40, 50, "40-50 Hz", "#2a78d6"),
              (60, 90, "60-90 Hz", "#0d3a73")]
    for ax, (ch, nome) in zip(axs, [(0, "A"), (1, "B")]):
        h = hits[hits.channel == ch]
        for a, b, lab, cor in grupos:
            x = h[h.freq.between(a, b)].amp_dbae
            ax.hist(x, bins=bins, histtype="step", lw=1.6, color=cor, label=lab,
                    weights=np.full(len(x), 1 / len(x)))
        ax.set_title(f"canal {nome}")
        ax.set_xlabel("amplitude de pico (dBAE)")
        ax.set_yscale("log")
    axs[0].set_ylabel("fração dos hits (log)")
    axs[0].legend(title="frequência do motor", fontsize=7, title_fontsize=7)
    fig.suptitle("Distribuição da amplitude: acúmulo em 102,8 dBAE (fundo de escala) a partir de 40 Hz",
                 fontsize=10)
    fig.tight_layout()
    salvar(fig, "fig_amplitude_saturacao.png")


def _mapa(ax, tab, titulo, fmt="{:.2f}", cmap="Blues", vmin=None, vmax=None):
    tab = tab.reindex(index=CONFIGS, columns=FREQS)
    im = ax.imshow(tab.to_numpy(float), cmap=cmap, aspect="auto", vmin=vmin, vmax=vmax)
    ax.grid(False)
    v = tab.to_numpy(float)
    lim = np.nanmin(v) + 0.6 * (np.nanmax(v) - np.nanmin(v))
    for i in range(v.shape[0]):
        for j in range(v.shape[1]):
            if np.isfinite(v[i, j]):
                ax.text(j, i, fmt.format(v[i, j]), ha="center", va="center", fontsize=6.5,
                        color="white" if v[i, j] > lim else TINTA)
    ax.set_xticks(range(len(FREQS)), FREQS)
    ax.set_yticks(range(len(CONFIGS)), CONFIGS)
    ax.set_title(titulo)
    return im


def fig_mapas_condicao(hits, df):
    g = hits.groupby(["config", "freq", "channel"])
    dur = hits.groupby(["config", "freq"]).rx_time_s.max()
    taxa = (g.size() / dur.reindex(g.size().index.droplevel(2)).to_numpy()).unstack(2)
    sat = g.sat.mean().unstack(2)
    mxd = g.maxdur.mean().unstack(2)
    en = df.groupby(["config", "freq"])[["A_log_energia_s", "B_log_energia_s"]].median()
    fig, axs = plt.subplots(2, 3, figsize=(14, 7))
    _mapa(axs[0, 0], taxa[0].unstack(1), "taxa de hits, canal A (hits/s)", "{:.1f}")
    _mapa(axs[1, 0], taxa[1].unstack(1), "taxa de hits, canal B (hits/s)", "{:.1f}")
    _mapa(axs[0, 1], sat[0].unstack(1), "fração SAT, canal A", vmin=0, vmax=1)
    _mapa(axs[1, 1], sat[1].unstack(1), "fração SAT, canal B", vmin=0, vmax=1)
    _mapa(axs[0, 2], mxd[0].unstack(1), "fração MAXDUR, canal A", vmin=0, vmax=1)
    _mapa(axs[1, 2], en.A_log_energia_s.unstack(1), "log10 energia/s, canal A (mediana)", "{:.1f}")
    for ax in axs[1]:
        ax.set_xlabel("frequência do motor (Hz)")
    for ax in axs[:, 0]:
        ax.set_ylabel("válvulas rec-suc (%)")
    fig.suptitle("Indicadores por condição (uma corrida por célula)", fontsize=10)
    fig.tight_layout()
    salvar(fig, "fig_mapas_condicao.png")


def fig_boxplots(df):
    sel = [("A_log_energia_s", "log10 energia/s, A"), ("B_log_energia_s", "log10 energia/s, B"),
           ("A_fm_kHz_med", "freq. média A (kHz)"), ("B_log_dur_p50", "log10 duração p50, B (µs)"),
           ("B_log_RA_p50", "log10 RA p50, B"), ("A_frac_maxdur", "fração MAXDUR, A")]
    for fator, rotulos, nome, xl in [("freq", FREQS, "fig_boxplot_por_frequencia.png", "frequência (Hz)"),
                                     ("config", CONFIGS, "fig_boxplot_por_valvula.png", "válvulas rec-suc (%)")]:
        fig, axs = plt.subplots(2, 3, figsize=(13, 6.4))
        for ax, (c, tit) in zip(axs.flat, sel):
            dados = [df.loc[df[fator] == r, c].dropna() for r in rotulos]
            bp = ax.boxplot(dados, widths=0.6, showfliers=False, patch_artist=True)
            for k, b in enumerate(bp["boxes"]):
                cor = COR_CFG[rotulos[k]] if fator == "config" else plt.cm.Blues(0.3 + 0.7 * k / 7)
                b.set(facecolor=cor, alpha=0.75, edgecolor=TINTA2)
            for l in bp["medians"]:
                l.set(color=TINTA, lw=1.2)
            ax.set_xticks(range(1, len(rotulos) + 1), rotulos, rotation=0 if fator == "freq" else 40)
            ax.set_title(tit)
        for ax in axs[1]:
            ax.set_xlabel(xl)
        fig.suptitle(f"Distribuição por janela de 5 s de descritores selecionados, por "
                     f"{'frequência' if fator == 'freq' else 'configuração de válvulas'}", fontsize=10)
        fig.tight_layout()
        salvar(fig, nome)


def fig_pca(df):
    cols = [c for c in M.colunas_descritores(df) if df[c].nunique() > 1]
    Z = StandardScaler().fit_transform(SimpleImputer(strategy="median").fit_transform(df[cols]))
    pca = PCA(n_components=3, random_state=0).fit(Z)
    P = pca.transform(Z)
    fig, axs = plt.subplots(1, 3, figsize=(14, 4.3))
    sc = axs[0].scatter(P[:, 0], P[:, 1], c=df.freq, cmap="Blues", s=4, vmin=0, vmax=95)
    fig.colorbar(sc, ax=axs[0], label="frequência (Hz)")
    axs[0].set_title("cor = frequência")
    for cfg in CONFIGS:
        m = (df.config == cfg).to_numpy()
        axs[1].scatter(P[m, 0], P[m, 1], s=3, color=COR_CFG[cfg], label=cfg, alpha=0.7, lw=0)
    axs[1].legend(fontsize=6.5, markerscale=3, title="rec-suc", title_fontsize=7, loc="lower left")
    axs[1].set_title("cor = válvulas")
    for d, cor in zip(sorted(df.dia.unique()), ["#eb6834", "#2a78d6"]):
        m = (df.dia == d).to_numpy()
        axs[2].scatter(P[m, 0], P[m, 1], s=3, color=cor, label=d, alpha=0.6, lw=0)
    axs[2].legend(fontsize=7, markerscale=3, title="dia", title_fontsize=7)
    axs[2].set_title("cor = dia do ensaio")
    v = pca.explained_variance_ratio_
    for ax in axs:
        ax.set_xlabel(f"PC1 ({v[0]:.0%})")
        ax.set_ylabel(f"PC2 ({v[1]:.0%})")
    fig.suptitle("Projeção PCA das 3 540 janelas (36 descritores padronizados)", fontsize=10)
    fig.tight_layout()
    salvar(fig, "fig_pca.png")
    # carregamentos
    load = pd.DataFrame(pca.components_[:2].T, index=cols, columns=["PC1", "PC2"])
    top = load.abs().max(axis=1).sort_values(ascending=False).index[:14][::-1]
    fig, ax = plt.subplots(figsize=(7, 4.6))
    y = np.arange(len(top))
    ax.barh(y + 0.2, load.loc[top, "PC1"], height=0.38, color="#2a78d6", label="PC1")
    ax.barh(y - 0.2, load.loc[top, "PC2"], height=0.38, color="#eb6834", label="PC2")
    ax.set_yticks(y, top)
    ax.axvline(0, color=TINTA2, lw=0.8)
    ax.grid(axis="y", visible=False)
    ax.legend()
    ax.set_title("Carregamentos das duas primeiras componentes")
    fig.tight_layout()
    salvar(fig, "fig_pca_carregamentos.png")


def fig_ari():
    ns = pd.read_csv(RES / "nao_supervisionado_ARI_NMI.csv")
    nomes = {"freq": "frequência", "config": "válvulas", "corrida": "corrida", "dia": "dia",
             "limiar": "limiar"}
    cores = {"freq": "#2a78d6", "config": "#eb6834", "corrida": "#1baf7a", "dia": "#eda100",
             "limiar": "#6250d6"}
    fig, axs = plt.subplots(1, 2, figsize=(11, 3.6))
    for ax, met in zip(axs, ["ARI", "NMI"]):
        for f in nomes:
            g = ns[ns.fator == f]
            ax.plot(g.k, g[met], marker="o", ms=4, lw=1.6, color=cores[f], label=nomes[f])
        ax.set_xscale("log")
        ax.minorticks_off()
        ax.set_xticks([2, 3, 7, 8, 14, 56], [2, 3, 7, 8, 14, 56])
        ax.set_xlabel("número de clusters k (k-means)")
        ax.set_ylabel(met)
        ax.set_title(f"{met}: clusters × fatores")
    axs[0].legend(fontsize=7)
    fig.tight_layout()
    salvar(fig, "fig_kmeans_ari.png")


# --------------------------------------------------------------------------- #
# Resultados supervisionados
# --------------------------------------------------------------------------- #
def fig_confusao_loso(pred):
    g = pred[(pred.esquema == "LOSO") & (pred.tarefa == "freq")]
    fig, axs = plt.subplots(1, 3, figsize=(13, 3.9))
    for ax, mod in zip(axs, ["MLP", "RF", "GB"]):
        gm = g[g.modelo == mod]
        cm = confusion_matrix(gm.freq, M.arred_freq(gm.pred), labels=FREQS)
        M._matriz(ax, cm, FREQS, FREQS, f"{mod} - frequência, LOSO")
        ax.set_xlabel("previsto (Hz)")
    axs[0].set_ylabel("real (Hz)")
    fig.tight_layout()
    salvar(fig, "fig_confusao_LOSO.png")


def fig_mapa_loro(pred):
    fig, axs = plt.subplots(1, 4, figsize=(17, 3.9))
    for ax, mod in zip(axs, ["MLP", "RF", "GB"]):
        g = pred[(pred.esquema == "LORO") & (pred.tarefa == "freq") & (pred.modelo == mod)]
        e = (g.pred.astype(float) - g.freq).abs().groupby([g.config, g.freq]).mean().unstack()
        _mapa(ax, e, f"{mod}: MAE da frequência (Hz)", "{:.1f}", cmap="Reds", vmin=0, vmax=20)
    g = pred[(pred.esquema == "LORO") & (pred.tarefa == "config") & (pred.modelo == "MLP")]
    a = (g.pred.astype(str) == g.config).groupby([g.config, g.freq]).mean().unstack()
    _mapa(axs[3], a, "MLP: acerto das válvulas", "{:.2f}", vmin=0, vmax=1)
    for ax in axs:
        ax.set_xlabel("frequência (Hz)")
    axs[0].set_ylabel("válvulas rec-suc (%)")
    fig.suptitle("Desempenho por corrida deixada de fora (LORO, média das janelas)", fontsize=10)
    fig.tight_layout()
    salvar(fig, "fig_mapa_LORO.png")


def fig_mae_folds(met):
    fig, axs = plt.subplots(1, 2, figsize=(12, 3.8), gridspec_kw=dict(width_ratios=[5, 7]))
    for ax, esq, ordem in [(axs[0], "BLOCOS", [f"bloco{i}" for i in range(1, 6)]),
                           (axs[1], "LOSO", [f"serie {c}" for c in CONFIGS])]:
        g = met[(met.esquema == esq) & (met.tarefa == "freq") & (met.fold != "TOTAL")]
        t = g.pivot(index="fold", columns="modelo", values="MAE_Hz").reindex(ordem)
        x = np.arange(len(t))
        for i, mod in enumerate(["MLP", "RF", "GB"]):
            ax.bar(x + (i - 1) * 0.27, t[mod], width=0.25, color=COR_MOD[mod], label=mod)
        ax.set_xticks(x, [o.replace("serie ", "") for o in ordem])
        ax.set_ylabel("MAE da frequência (Hz)")
        ax.set_title(f"{esq}: MAE por fold")
        ax.grid(axis="x", visible=False)
    axs[1].set_xlabel("série deixada de fora (rec-suc %)")
    axs[0].set_xlabel("trecho de tempo deixado de fora")
    axs[0].legend()
    fig.tight_layout()
    salvar(fig, "fig_mae_por_fold.png")


def fig_blocos_vs_loro(met):
    t = met[(met.fold == "TOTAL") & (met.tarefa == "config")].pivot(index="esquema", columns="modelo",
                                                                    values="acc")
    f = met[(met.fold == "TOTAL") & (met.tarefa == "freq")].pivot(index="esquema", columns="modelo",
                                                                  values="MAE_Hz")
    fig, axs = plt.subplots(1, 2, figsize=(11, 3.7))
    for ax, tab, esqs, yl, tit in [
            (axs[0], f, ["BLOCOS", "LORO", "LOSO"], "MAE (Hz)", "Frequência: erro médio absoluto"),
            (axs[1], t, ["BLOCOS", "LORO"], "acurácia (7 classes)", "Válvulas: acurácia")]:
        x = np.arange(len(esqs))
        for i, mod in enumerate(["MLP", "RF", "GB"]):
            v = tab.loc[esqs, mod]
            ax.bar(x + (i - 1) * 0.27, v, width=0.25, color=COR_MOD[mod], label=mod)
            for xi, vi in zip(x, v):
                ax.text(xi + (i - 1) * 0.27, vi, f"{vi:.2f}" if tab is t else f"{vi:.1f}",
                        ha="center", va="bottom", fontsize=6.5, color=TINTA)
        ax.set_xticks(x, esqs)
        ax.set_ylabel(yl)
        ax.set_title(tit)
        ax.grid(axis="x", visible=False)
    axs[1].axhline(1 / 7, color=TINTA, lw=1, ls="--")
    axs[1].text(1.45, 1 / 7 + 0.015, "acaso (1/7)", fontsize=7, ha="right")
    axs[0].legend()
    fig.suptitle("Mesmo modelo, validações diferentes: BLOCOS (corrida no treino) × LORO/LOSO "
                 "(corrida ou série inédita)", fontsize=10)
    fig.tight_layout()
    salvar(fig, "fig_blocos_vs_loro.png")


def fig_sensibilidade():
    var = [("principal\n(5 s, corte)", RES), ("sem corte\nde limiar", SEN / "sem_harmonizar"),
           ("janela\n2 s", SEN / "janela_2s")]
    linhas = []
    for nome, p in var:
        m = pd.read_csv(p / "metricas.csv")
        m = m[m.fold == "TOTAL"]
        for _, r in m.iterrows():
            if r.esquema == "BLOCOS":
                continue
            linhas.append(dict(var=nome, esq=r.esquema, tar=r.tarefa, mod=r.modelo,
                               v=r.MAE_Hz if r.tarefa == "freq" else r.acc))
    s = pd.DataFrame(linhas)
    s.to_csv(FIG / "sensibilidade.csv", index=False)
    fig, axs = plt.subplots(1, 3, figsize=(14, 3.7))
    for ax, (esq, tar, yl) in zip(axs, [("LORO", "freq", "MAE (Hz)"), ("LOSO", "freq", "MAE (Hz)"),
                                        ("LORO", "config", "acurácia")]):
        g = s[(s.esq == esq) & (s.tar == tar)]
        x = np.arange(len(var))
        for i, mod in enumerate(["MLP", "RF", "GB"]):
            v = [g[(g["var"] == n) & (g["mod"] == mod)].v.iloc[0] for n, _ in var]
            ax.bar(x + (i - 1) * 0.27, v, width=0.25, color=COR_MOD[mod], label=mod)
        ax.set_xticks(x, [n for n, _ in var])
        ax.set_ylabel(yl)
        ax.set_title(f"{'frequência' if tar == 'freq' else 'válvulas'} - {esq}")
        ax.grid(axis="x", visible=False)
        if tar == "config":
            ax.axhline(1 / 7, color=TINTA, lw=1, ls="--")
    axs[0].legend()
    fig.suptitle("Sensibilidade ao corte de limiar e ao tamanho da janela", fontsize=10)
    fig.tight_layout()
    salvar(fig, "fig_sensibilidade.png")


# --------------------------------------------------------------------------- #
# Diagramas da divisão dos dados
# --------------------------------------------------------------------------- #
def _grade(ax, estado, titulo, legenda=True):
    """estado: dict (config, freq) -> lista de cores das janelas (63 por corrida)."""
    for i, cfg in enumerate(CONFIGS):
        for j, f in enumerate(FREQS):
            cores = estado[(cfg, f)]
            n = len(cores)
            for k, c in enumerate(cores):
                ax.add_patch(Rectangle((j + 0.05 + 0.9 * k / n, i + 0.12), 0.9 / n, 0.76,
                                       color=c, lw=0))
    ax.set_xlim(0, len(FREQS))
    ax.set_ylim(len(CONFIGS), 0)
    ax.set_xticks(np.arange(len(FREQS)) + 0.5, FREQS)
    ax.set_yticks(np.arange(len(CONFIGS)) + 0.5, CONFIGS)
    ax.set_xlabel("frequência do motor (Hz); cada célula = 1 corrida, faixas = janelas de 5 s no tempo")
    ax.set_ylabel("válvulas rec-suc (%)")
    ax.grid(False)
    ax.set_title(titulo, fontsize=9.5)
    for s in ax.spines.values():
        s.set_visible(False)


def _leg(fig, extra=()):
    h = [Patch(color=C_TREINO, label="treino"), Patch(color=C_VAL, label="validação interna (early stopping MLP)"),
         Patch(color=C_TESTE, label="teste"), Patch(color=C_GAP, label="exclusão (não usada)")] + list(extra)
    fig.legend(handles=h, loc="lower center", ncol=len(h), fontsize=8, bbox_to_anchor=(0.5, -0.01))


def diagramas_divisao(df):
    rng = np.random.default_rng(42)
    nj = 61

    def treino_com_val(n):
        c = [C_TREINO] * n
        for k in rng.choice(n, int(round(0.15 * n)), replace=False):
            c[k] = C_VAL
        return c

    # aleatória 70/15/15
    est = {}
    for cfg in CONFIGS:
        for f in FREQS:
            est[(cfg, f)] = list(rng.choice([C_TREINO, C_VAL, C_TESTE], nj, p=[0.7, 0.15, 0.15]))
    fig, ax = plt.subplots(figsize=(11, 5))
    _grade(ax, est, "Divisão aleatória 70/15/15 por janela (NÃO usada): janelas vizinhas da mesma corrida "
                    "caem em treino e teste")
    _leg(fig)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    salvar(fig, "fig_divisao_aleatoria.png")

    # BLOCOS (fold 3)
    est = {}
    for cfg in CONFIGS:
        for f in FREQS:
            b = np.minimum((np.arange(nj) * 5 // nj), 4)
            c = treino_com_val(nj)
            te = np.where(b == 2)[0]
            for k in te:
                c[k] = C_TESTE
            for k in (te.min() - 1, te.max() + 1):
                c[k] = C_GAP
            est[(cfg, f)] = c
    fig, ax = plt.subplots(figsize=(11, 5))
    _grade(ax, est, "BLOCOS, fold 3 de 5: o 3º trecho (~60 s) de TODAS as corridas é teste; "
                    "1 janela de exclusão de cada lado")
    _leg(fig)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    salvar(fig, "fig_divisao_blocos.png")

    # LORO (fold 100-50 @ 60 Hz)
    est = {}
    for cfg in CONFIGS:
        for f in FREQS:
            est[(cfg, f)] = [C_TESTE] * nj if (cfg, f) == ("100-50", 60) else treino_com_val(nj)
    fig, ax = plt.subplots(figsize=(11, 5))
    _grade(ax, est, "LORO, 1 dos 56 folds: a corrida 100-50 @ 60 Hz inteira é teste; "
                    "60 Hz e 100-50 aparecem no treino em outras combinações")
    _leg(fig)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    salvar(fig, "fig_divisao_loro.png")

    # LOSO (série 100-25)
    est = {}
    for cfg in CONFIGS:
        for f in FREQS:
            est[(cfg, f)] = [C_TESTE] * nj if cfg == "100-25" else treino_com_val(nj)
    fig, ax = plt.subplots(figsize=(11, 5))
    _grade(ax, est, "LOSO, 1 dos 7 folds: a série 100-25 inteira (sessão de 06/11, 09h49-10h32) é teste "
                    "(só frequência)")
    _leg(fig)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    salvar(fig, "fig_divisao_loso.png")

    # OOT proposto: dia 05/11 treino, 06/11 teste
    dia1 = {"100-100", "25-100", "100-50"}
    est = {}
    for cfg in CONFIGS:
        for f in FREQS:
            est[(cfg, f)] = treino_com_val(nj) if cfg in dia1 else [C_TESTE] * nj
    fig, ax = plt.subplots(figsize=(11, 5))
    _grade(ax, est, "Holdout fora do tempo (OOT): treino com as séries de 05/11, teste nas de 06/11 "
                    "(só frequência)")
    _leg(fig)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    salvar(fig, "fig_divisao_oot.png")

    # validação interna: sorteada × por corrida (uma linha de corridas)
    fig, axs = plt.subplots(2, 1, figsize=(11, 2.8))
    runs = 10
    for ax, modo in zip(axs, ["sorteio", "grupo"]):
        for r in range(runs):
            if modo == "sorteio":
                c = treino_com_val(nj)
            else:
                c = [C_VAL if r in (3, 8) else C_TREINO] * nj
            for k, cc in enumerate(c):
                ax.add_patch(Rectangle((r + 0.05 + 0.9 * k / nj, 0.1), 0.9 / nj, 0.8, color=cc, lw=0))
        ax.set_xlim(0, runs)
        ax.set_ylim(0, 1)
        ax.axis("off")
    axs[0].set_title("Atual (sklearn early_stopping): 15 % das janelas de treino sorteadas, "
                     "misturadas às vizinhas da mesma corrida", fontsize=9)
    axs[1].set_title("Recomendado: validação interna por corrida inteira (ex.: 15 % das corridas de treino)",
                     fontsize=9)
    fig.legend(handles=[Patch(color=C_TREINO, label="treino efetivo da MLP"),
                        Patch(color=C_VAL, label="validação interna")], loc="lower center", ncol=2,
               fontsize=8, bbox_to_anchor=(0.5, -0.06))
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    salvar(fig, "fig_validacao_interna_esquema.png")


# --------------------------------------------------------------------------- #
# Experimentos curtos
# --------------------------------------------------------------------------- #
def experimento_validacao_interna(df):
    cols = [c for c in M.colunas_descritores(df) if df[c].nunique() > 1]
    X = df[cols].to_numpy(float)
    linhas = []
    for esq in ("BLOCOS", "LORO"):
        folds = M.gerar_folds(df, esq, 5)
        if esq == "LORO":
            folds = folds[::4]          # 14 dos 56 folds, espalhados no tempo
        for tarefa in ("freq", "config"):
            y = df.freq.to_numpy(float) if tarefa == "freq" else df.config.to_numpy()
            mod = M.modelos(tarefa)["MLP"]
            for nome, tr, te in folds:
                m = clone(mod).fit(X[tr], y[tr])
                est = m[-1].regressor_ if tarefa == "freq" else m[-1]
                p = m.predict(X[te])
                teste = (r2_score(y[te], p) if tarefa == "freq" and np.std(y[te]) > 0 else
                         (np.nan if tarefa == "freq" else accuracy_score(y[te], p)))
                linhas.append(dict(esquema=esq, tarefa=tarefa, fold=nome,
                                   metrica="R2" if tarefa == "freq" else "acurácia",
                                   interna=est.best_validation_score_, teste=teste,
                                   MAE_teste=mean_absolute_error(y[te], p) if tarefa == "freq" else np.nan,
                                   epocas=est.n_iter_))
    r = pd.DataFrame(linhas)
    r.to_csv(FIG / "validacao_interna_vs_teste.csv", index=False)
    res = r.groupby(["esquema", "tarefa", "metrica"])[["interna", "teste", "MAE_teste", "epocas"]].mean()
    res.round(3).to_csv(FIG / "validacao_interna_vs_teste_resumo.csv")
    print(res.round(3))
    fig, axs = plt.subplots(1, 2, figsize=(11, 3.6))
    for ax, tarefa, yl in [(axs[0], "config", "acurácia (válvulas)"), (axs[1], "freq", "R² (frequência)")]:
        g = r[r.tarefa == tarefa]
        for k, esq in enumerate(["BLOCOS", "LORO"]):
            gg = g[g.esquema == esq]
            ax.scatter(np.full(len(gg), k * 3) + np.random.default_rng(1).uniform(-.25, .25, len(gg)),
                       gg.interna, color=C_VAL, s=18, label="validação interna" if k == 0 else None)
            ax.scatter(np.full(len(gg), k * 3 + 1) + np.random.default_rng(2).uniform(-.25, .25, len(gg)),
                       gg.teste, color=C_TESTE, s=18, label="teste do fold" if k == 0 else None)
        ax.set_xlim(-0.7, 4.9)
        ax.set_xticks([0, 1, 3, 4], ["BLOCOS\ninterna", "BLOCOS\nteste", "LORO\ninterna", "LORO\nteste"])
        ax.set_ylabel(yl)
        ax.grid(axis="x", visible=False)
        if tarefa == "config":
            ax.axhline(1 / 7, color=TINTA, lw=1, ls="--")
        else:
            gl = g[g.esquema == "LORO"]
            ax.text(4, 0.5 * (ax.get_ylim()[0] + ax.get_ylim()[1]),
                    f"LORO: cada fold tem uma só\nfrequência, R² do teste\nindefinido; MAE do teste\n"
                    f"= {gl.MAE_teste.mean():.1f} Hz".replace(".", ","),
                    ha="center", va="center", fontsize=7, color=TINTA2)
    axs[0].legend(fontsize=7, loc="lower left")
    fig.suptitle("MLP: escore da validação interna do early stopping × escore no teste (cada ponto = 1 fold)",
                 fontsize=10)
    fig.tight_layout()
    salvar(fig, "fig_validacao_interna_vs_teste.png")
    return res


def experimento_oot(df):
    cols = [c for c in M.colunas_descritores(df) if df[c].nunique() > 1]
    X = df[cols].to_numpy(float)
    y = df.freq.to_numpy(float)
    dias = sorted(df.dia.unique())
    linhas, preds = [], []
    for dtr, dte in [(dias[0], dias[1]), (dias[1], dias[0])]:
        tr = np.where(df.dia == dtr)[0]
        te = np.where(df.dia == dte)[0]
        for nome, mod in M.modelos("freq").items():
            p = clone(mod).fit(X[tr], y[tr]).predict(X[te])
            gr = pd.DataFrame(dict(run=df.run.iloc[te], f=y[te], p=p)).groupby("run").agg(
                f=("f", "first"), p=("p", "median"))
            linhas.append(dict(treino=dtr, teste=dte, modelo=nome, MAE_Hz=mean_absolute_error(y[te], p),
                               acc_nivel=accuracy_score(y[te], M.arred_freq(p)),
                               acc_10Hz=float(np.mean(np.abs(M.arred_freq(p) - y[te]) <= 10)),
                               MAE_Hz_corrida=mean_absolute_error(gr.f, gr.p)))
            preds.append(pd.DataFrame(dict(treino=dtr, modelo=nome, f=y[te], p=p)))
    r = pd.DataFrame(linhas)
    r.to_csv(FIG / "oot_por_dia.csv", index=False)
    print(r.round(3).to_string(index=False))
    pr = pd.concat(preds)
    fig, axs = plt.subplots(1, 2, figsize=(11, 3.8), sharey=True)
    for ax, dtr in zip(axs, dias):
        g = pr[pr.treino == dtr]
        for i, mod in enumerate(["MLP", "RF", "GB"]):
            gm = g[g.modelo == mod]
            med = gm.groupby("f").p.median()
            q1, q3 = gm.groupby("f").p.quantile(0.25), gm.groupby("f").p.quantile(0.75)
            x = med.index + (i - 1) * 2
            ax.errorbar(x, med, yerr=[med - q1, q3 - med], fmt="o", ms=4, color=COR_MOD[mod],
                        label=mod, capsize=2, lw=1)
        ax.plot([15, 95], [15, 95], ls="--", color=TINTA2, lw=1)
        dte = [d for d in dias if d != dtr][0]
        ax.set_title(f"treino {pd.Timestamp(dtr):%d/%m} → teste {pd.Timestamp(dte):%d/%m}")
        ax.set_xlabel("frequência real (Hz)")
        ax.set_xticks(FREQS, FREQS)
    axs[0].set_ylabel("prevista (mediana e quartis das janelas)")
    axs[0].legend()
    fig.suptitle("Holdout fora do tempo (OOT) entre dias, frequência do motor", fontsize=10)
    fig.tight_layout()
    salvar(fig, "fig_oot_por_dia.png")
    return r


def copiar_existentes():
    for p in RES.glob("fig_*.png"):
        shutil.copy2(p, FIG / p.name)


def main():
    print("Carregando...")
    df, cor, pred, met, hits = carregar()
    print("Figuras de dados e qualidade")
    fig_ordem_temporal(cor)
    fig_exemplo_hits(hits, df)
    fig_intervalo_hits(hits)
    fig_amplitude(hits)
    fig_mapas_condicao(hits, df)
    fig_boxplots(df)
    fig_pca(df)
    fig_ari()
    print("Figuras de resultados")
    fig_confusao_loso(pred)
    fig_mapa_loro(pred)
    fig_mae_folds(met)
    fig_blocos_vs_loro(met)
    fig_sensibilidade()
    print("Diagramas da divisão")
    diagramas_divisao(df)
    print("Experimento: validação interna × teste")
    experimento_validacao_interna(df)
    print("Experimento: OOT entre dias")
    experimento_oot(df)
    copiar_existentes()
    print("OK ->", FIG)


if __name__ == "__main__":
    main()
