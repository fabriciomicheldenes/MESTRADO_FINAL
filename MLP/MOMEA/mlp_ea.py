# -*- coding: utf-8 -*-
"""
mlp_ea.py - Classificação das condições de operação de uma bomba a partir dos
hits de emissão acústica (EA) gravados pelo MOMEA (ae_hit_dashboard.py, hits.csv).

Fluxo
  1. Lê todas as corridas em Dados/<VALV>/<NN_..HZ-...>/<data_hora>/hits.csv.
     Os rótulos saem do caminho: a pasta de 1º nível dá a abertura das válvulas
     (ex.: 100REC_50SUC) e a de 2º nível dá a frequência (ex.: 01_40HZ-...).
     Atenção: as subpastas de 100REC_50SUC estão nomeadas "REC100-25SUC";
     a pasta de 1º nível é a que vale.
  2. Harmoniza o limiar (descarta hits abaixo do maior limiar usado nas sessões).
  3. Agrega os hits em janelas de tempo (padrão 5 s) e calcula descritores por canal.
  4. Treina MLP, random forest e gradient boosting com três validações sem sorteio
     de hits:
       BLOCOS : cada corrida é dividida em K blocos de tempo consecutivos; deixa-se
                um bloco de fora por vez (pseudo-repetição), com zona de exclusão
                de 1 janela entre treino e teste.
       LORO   : deixa uma corrida (condição) inteira de fora; a frequência e a
                configuração da corrida aparecem no treino em outras combinações.
       LOSO   : deixa uma série de válvulas inteira de fora (sessão contínua no
                tempo); só para a frequência (a classe de válvula ficaria ausente).
     Alvos: frequência do motor (regressão; arredondada ao nível mais próximo para
     a leitura ordinal) e configuração das válvulas (7 classes, 8 corridas cada;
     % recalque e % sucção derivados dela).
  5. Importância dos descritores (permutação nos blocos de teste).
  6. Verificação não supervisionada: k-means sem rótulos, cruzado com os fatores
     (ARI, NMI, tabelas de contingência) e PCA.
  7. Grava métricas (CSV), figuras (PNG) e um relatório automático (Markdown)
     em resultados/.

Uso
  .venv\\Scripts\\python.exe mlp_ea.py                 # janela 5 s
  .venv\\Scripts\\python.exe mlp_ea.py --janela 2      # janela 2 s
  .venv\\Scripts\\python.exe mlp_ea.py --sem-harmonizar --saida resultados_sem_harm
"""
from __future__ import annotations

import argparse
import json
import os
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from joblib import Parallel, delayed  # noqa: E402
from sklearn.base import clone  # noqa: E402
from sklearn.cluster import KMeans  # noqa: E402
from sklearn.compose import TransformedTargetRegressor  # noqa: E402
from sklearn.decomposition import PCA  # noqa: E402
from sklearn.ensemble import (HistGradientBoostingClassifier,  # noqa: E402
                              HistGradientBoostingRegressor,
                              RandomForestClassifier, RandomForestRegressor)
from sklearn.impute import SimpleImputer  # noqa: E402
from sklearn.inspection import permutation_importance  # noqa: E402
from sklearn.metrics import (accuracy_score, adjusted_rand_score,  # noqa: E402
                             balanced_accuracy_score, confusion_matrix,
                             f1_score, mean_absolute_error,
                             normalized_mutual_info_score, r2_score)
from sklearn.neural_network import MLPClassifier, MLPRegressor  # noqa: E402
from sklearn.pipeline import make_pipeline  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

AQUI = Path(__file__).resolve().parent
FREQS = [20, 30, 40, 50, 60, 70, 80, 90]
CONFIGS = ["100-100", "100-75", "100-50", "100-25", "75-100", "50-100", "25-100"]
CANAIS = {0: "A", 1: "B"}
SEMENTE = 42

# Paleta (ordem categórica fixa) e rampa sequencial única
COR_MODELO = {"MLP": "#2a78d6", "RF": "#eb6834", "GB": "#1baf7a"}
CMAP_SEQ = "Blues"
TINTA = "#0b0b0b"
TINTA2 = "#52514e"
GRADE = "#e4e3df"

plt.rcParams.update({
    "figure.dpi": 110, "savefig.dpi": 150, "savefig.bbox": "tight",
    "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9,
    "axes.edgecolor": TINTA2, "axes.labelcolor": TINTA, "xtick.color": TINTA2,
    "ytick.color": TINTA2, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRADE, "grid.linewidth": 0.6,
    "axes.axisbelow": True, "legend.frameon": False,
})


# --------------------------------------------------------------------------- #
# 1. Leitura
# --------------------------------------------------------------------------- #
def ler_corridas(pasta_dados: Path) -> list[dict]:
    corridas = []
    for csv in sorted(pasta_dados.glob("*/*/*/hits.csv")):
        valv_dir, freq_dir, data_dir = csv.parts[-4], csv.parts[-3], csv.parts[-2]
        m_v = re.fullmatch(r"(\d+)REC_(\d+)SUC", valv_dir, re.I)
        m_f = re.search(r"_(\d+)HZ", freq_dir, re.I)
        if not (m_v and m_f):
            print(f"  [aviso] caminho fora do padrão, ignorado: {csv}")
            continue
        rec, suc, freq = int(m_v.group(1)), int(m_v.group(2)), int(m_f.group(1))
        m_sub = re.search(r"REC(\d+)-(\d+)SUC", freq_dir, re.I)
        if m_sub and (int(m_sub.group(1)), int(m_sub.group(2))) != (rec, suc):
            print(f"  [aviso] subpasta '{freq_dir}' diz {m_sub.group(1)}-{m_sub.group(2)}; "
                  f"vale a pasta '{valv_dir}' ({rec}-{suc})")
        cfg = json.loads((csv.parent / "config.json").read_text(encoding="utf-8"))
        ex = cfg.get("extrator", {})
        hits = pd.read_csv(csv)
        corridas.append(dict(
            arquivo=str(csv.relative_to(pasta_dados)), rec=rec, suc=suc, freq=freq,
            config=f"{rec}-{suc}", inicio=pd.Timestamp(cfg.get("inicio", data_dir)),
            thr_a=float(ex.get("thr_dbae_a", np.nan)), thr_b=float(ex.get("thr_dbae_b", np.nan)),
            gain_a=float(ex.get("gain_db_a", np.nan)), gain_b=float(ex.get("gain_db_b", np.nan)),
            hits=hits))
    if not corridas:
        raise SystemExit(f"Nenhum hits.csv encontrado em {pasta_dados}")
    corridas.sort(key=lambda c: c["inicio"])
    for i, c in enumerate(corridas):
        c["ordem"] = i          # ordem cronológica global
        c["run"] = f"{c['config']}@{c['freq']}Hz"
    return corridas


# --------------------------------------------------------------------------- #
# 2-3. Janelas e descritores
# --------------------------------------------------------------------------- #
def _pct(x, q):
    return float(np.percentile(x, q)) if len(x) else np.nan


def descritores_canal(h: pd.DataFrame, dur_janela: float, pref: str) -> dict:
    """Descritores de um canal numa janela. h já filtrado (canal, janela, limiar)."""
    f = {}
    n = len(h)
    f[f"{pref}taxa_hits"] = n / dur_janela
    f[f"{pref}frac_sat"] = h["sat"].mean() if n else np.nan
    f[f"{pref}frac_maxdur"] = h["maxdur"].mean() if n else np.nan
    # energia total por segundo (inclui hits truncados por MAXDUR: sinal contínuo)
    f[f"{pref}log_energia_s"] = np.log10(h["energy_eu"].sum() / dur_janela + 1.0)
    # frequência média de todos os hits (contagens / duração), em kHz
    fm_all = (h["counts"] / h["duration_us"].clip(lower=1) * 1e3).to_numpy()
    f[f"{pref}fm_kHz_med"] = _pct(fm_all, 50)
    # descritores de forma só com hits "normais" (sem MAXDUR, que trunca a duração)
    hn = h[h["maxdur"] == 0]
    amp = hn["amp_dbae"].to_numpy()
    f[f"{pref}amp_med"] = amp.mean() if len(amp) else np.nan
    f[f"{pref}amp_p10"] = _pct(amp, 10)
    f[f"{pref}amp_p50"] = _pct(amp, 50)
    f[f"{pref}amp_p90"] = _pct(amp, 90)
    le = np.log10(hn["energy_eu"].to_numpy() + 1.0)
    f[f"{pref}log_energia_p50"] = _pct(le, 50)
    f[f"{pref}log_energia_p90"] = _pct(le, 90)
    ld = np.log10(hn["duration_us"].to_numpy() + 1.0)
    f[f"{pref}log_dur_p50"] = _pct(ld, 50)
    f[f"{pref}log_dur_p90"] = _pct(ld, 90)
    lc = np.log10(hn["counts"].to_numpy() + 1.0)
    f[f"{pref}log_cont_p50"] = _pct(lc, 50)
    f[f"{pref}log_cont_p90"] = _pct(lc, 90)
    f[f"{pref}log_subida_p50"] = _pct(np.log10(hn["rise_us"].to_numpy() + 1.0), 50)
    fm = (hn["counts"] / hn["duration_us"].clip(lower=1) * 1e3).to_numpy()
    f[f"{pref}fm_kHz_normais"] = _pct(fm, 50)
    # RA = tempo de subida / amplitude (ms/V), amplitude em volts no sensor
    ra = (hn["rise_us"] / 1e3 / hn["peak_v"].clip(lower=1e-6)).to_numpy()
    f[f"{pref}log_RA_p50"] = _pct(np.log10(ra + 1e-6), 50)
    return f


def janelar(corridas: list[dict], dur_janela: float, n_blocos: int, harmonizar: bool,
            thr_comum: tuple[float, float]) -> pd.DataFrame:
    linhas = []
    for c in corridas:
        h = c["hits"]
        if harmonizar:
            thr = np.where(h["channel"] == 0, thr_comum[0], thr_comum[1])
            h = h[h["amp_dbae"] >= thr]
        t = h["rx_time_s"].to_numpy()
        n_jan = int(np.floor(c["hits"]["rx_time_s"].max() / dur_janela))
        idx = np.floor(t / dur_janela).astype(int)
        for k in range(n_jan):
            hk = h[idx == k]
            lin = dict(run=c["run"], ordem=c["ordem"], freq=c["freq"], rec=c["rec"],
                       suc=c["suc"], config=c["config"], janela=k,
                       t_ini=k * dur_janela, n_jan_run=n_jan,
                       bloco=min(int(k * n_blocos / n_jan), n_blocos - 1))
            for ch, nome in CANAIS.items():
                lin.update(descritores_canal(hk[hk["channel"] == ch], dur_janela, f"{nome}_"))
            linhas.append(lin)
    df = pd.DataFrame(linhas)
    # série de válvulas = sessão contínua; dia/limiar para checagem de confusão
    meta = {c["run"]: c for c in corridas}
    df["dia"] = df["run"].map(lambda r: str(meta[r]["inicio"].date()))
    df["limiar"] = df["run"].map(lambda r: f"{meta[r]['thr_a']:.0f}/{meta[r]['thr_b']:.0f}")
    return df


def colunas_descritores(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c[:2] in ("A_", "B_")]


# --------------------------------------------------------------------------- #
# 4. Modelos e validação
# --------------------------------------------------------------------------- #
def modelos(tarefa: str) -> dict:
    pre = lambda: [SimpleImputer(strategy="median"), StandardScaler()]  # noqa: E731
    if tarefa == "freq":
        mlp = MLPRegressor(hidden_layer_sizes=(64, 32), alpha=1e-3, learning_rate_init=1e-3,
                           early_stopping=True, validation_fraction=0.15, max_iter=2000,
                           n_iter_no_change=30, random_state=SEMENTE)
        return {
            "MLP": make_pipeline(*pre(), TransformedTargetRegressor(
                regressor=mlp, transformer=StandardScaler())),
            "RF": make_pipeline(*pre(), RandomForestRegressor(
                n_estimators=300, min_samples_leaf=2, max_features=0.5, n_jobs=1,
                random_state=SEMENTE)),
            "GB": make_pipeline(*pre(), HistGradientBoostingRegressor(
                max_iter=300, learning_rate=0.05, early_stopping=False, random_state=SEMENTE)),
        }
    mlp = MLPClassifier(hidden_layer_sizes=(64, 32), alpha=1e-3, learning_rate_init=1e-3,
                        early_stopping=True, validation_fraction=0.15, max_iter=2000,
                        n_iter_no_change=30, random_state=SEMENTE)
    return {
        "MLP": make_pipeline(*pre(), mlp),
        "RF": make_pipeline(*pre(), RandomForestClassifier(
            n_estimators=300, min_samples_leaf=2, max_features="sqrt", n_jobs=1,
            random_state=SEMENTE)),
        "GB": make_pipeline(*pre(), HistGradientBoostingClassifier(
            max_iter=200, learning_rate=0.05, early_stopping=False, random_state=SEMENTE)),
    }


def gerar_folds(df: pd.DataFrame, esquema: str, n_blocos: int):
    """Devolve lista de (nome_fold, idx_treino, idx_teste)."""
    folds = []
    if esquema == "BLOCOS":
        for b in range(n_blocos):
            te = df["bloco"] == b
            # zona de exclusão: janelas vizinhas ao bloco de teste na mesma corrida
            viz = np.zeros(len(df), bool)
            for _, g in df[te].groupby("run"):
                kmin, kmax = g["janela"].min(), g["janela"].max()
                viz |= (df["run"] == g["run"].iloc[0]).to_numpy() & \
                       df["janela"].isin([kmin - 1, kmax + 1]).to_numpy()
            tr = ~te.to_numpy() & ~viz
            folds.append((f"bloco{b + 1}", np.where(tr)[0], np.where(te)[0]))
    elif esquema == "LORO":
        for r in sorted(df["run"].unique(), key=lambda r: df.loc[df.run == r, "ordem"].iloc[0]):
            te = (df["run"] == r).to_numpy()
            folds.append((r, np.where(~te)[0], np.where(te)[0]))
    elif esquema == "LOSO":
        for cfg in CONFIGS:
            te = (df["config"] == cfg).to_numpy()
            if te.any():
                folds.append((f"serie {cfg}", np.where(~te)[0], np.where(te)[0]))
    return folds


def _ajusta_prediz(modelo, X, y, tr, te):
    m = clone(modelo).fit(X[tr], y[tr])
    return te, m.predict(X[te])


def validar(df, X, esquema, tarefa, n_blocos, n_jobs):
    y = df["freq"].to_numpy(float) if tarefa == "freq" else df["config"].to_numpy()
    folds = gerar_folds(df, esquema, n_blocos)
    saida = []
    for nome, modelo in modelos(tarefa).items():
        t0 = time.time()
        res = Parallel(n_jobs=n_jobs)(delayed(_ajusta_prediz)(modelo, X, y, tr, te)
                                      for _, tr, te in folds)
        pred = np.empty(len(df), dtype=object if tarefa != "freq" else float)
        fold_de = np.empty(len(df), dtype=object)
        for (fnome, _, _), (te, p) in zip(folds, res):
            pred[te] = p
            fold_de[te] = fnome
        d = df[["run", "ordem", "freq", "config", "rec", "suc", "janela", "bloco"]].copy()
        d["modelo"], d["esquema"], d["tarefa"], d["fold"] = nome, esquema, tarefa, fold_de
        d["pred"] = pred
        saida.append(d)
        print(f"    {esquema:6s} {tarefa:6s} {nome:3s}  {len(folds):2d} folds  {time.time() - t0:5.1f} s")
    return pd.concat(saida, ignore_index=True)


def arred_freq(p):
    p = np.asarray(p, float)
    return np.array(FREQS)[np.abs(p[:, None] - np.array(FREQS)[None, :]).argmin(1)]


def metricas(pred: pd.DataFrame) -> pd.DataFrame:
    """Métricas por (esquema, tarefa, modelo, fold) e totais; janela e corrida."""
    linhas = []
    for (esq, tar, mod), g in pred.groupby(["esquema", "tarefa", "modelo"]):
        for fold, gf in list(g.groupby("fold")) + [("TOTAL", g)]:
            lin = dict(esquema=esq, tarefa=tar, modelo=mod, fold=fold, n_janelas=len(gf))
            if tar == "freq":
                yt, yp = gf["freq"].to_numpy(float), gf["pred"].to_numpy(float)
                lin.update(MAE_Hz=mean_absolute_error(yt, yp),
                           RMSE_Hz=float(np.sqrt(np.mean((yt - yp) ** 2))),
                           R2=r2_score(yt, yp) if yt.std() > 0 else np.nan,
                           acc_nivel=accuracy_score(yt, arred_freq(yp)),
                           acc_10Hz=float(np.mean(np.abs(arred_freq(yp) - yt) <= 10)))
                # por corrida: mediana das janelas
                gr = gf.groupby("run").agg(yt=("freq", "first"), yp=("pred", "median"))
                lin.update(MAE_Hz_corrida=mean_absolute_error(gr.yt, gr.yp),
                           acc_nivel_corrida=accuracy_score(gr.yt, arred_freq(gr.yp)))
            else:
                yt, yp = gf["config"].astype(str), gf["pred"].astype(str)
                rec_p = yp.str.split("-").str[0].astype(int)
                suc_p = yp.str.split("-").str[1].astype(int)
                lin.update(acc=accuracy_score(yt, yp),
                           acc_bal=balanced_accuracy_score(yt, yp) if fold == "TOTAL" else np.nan,
                           F1_macro=f1_score(yt, yp, average="macro") if fold == "TOTAL" else np.nan,
                           acc_rec=float(np.mean(rec_p.to_numpy() == gf["rec"].to_numpy())),
                           acc_suc=float(np.mean(suc_p.to_numpy() == gf["suc"].to_numpy())))
                gr = gf.groupby("run").agg(yt=("config", "first"),
                                           yp=("pred", lambda s: s.astype(str).mode().iloc[0]))
                lin.update(acc_corrida=accuracy_score(gr.yt, gr.yp))
            linhas.append(lin)
    return pd.DataFrame(linhas)


# --------------------------------------------------------------------------- #
# 5. Importância
# --------------------------------------------------------------------------- #
def importancias(df, X, cols, n_blocos, n_jobs):
    linhas = []
    for tarefa in ("freq", "config"):
        y = df["freq"].to_numpy(float) if tarefa == "freq" else df["config"].to_numpy()
        score = "neg_mean_absolute_error" if tarefa == "freq" else "balanced_accuracy"
        for nome, modelo in modelos(tarefa).items():
            def um_fold(tr, te, modelo=modelo, y=y):
                m = clone(modelo).fit(X[tr], y[tr])
                r = permutation_importance(m, X[te], y[te], scoring=score, n_repeats=5,
                                           random_state=SEMENTE, n_jobs=1)
                return r.importances_mean
            res = Parallel(n_jobs=n_jobs)(delayed(um_fold)(tr, te)
                                          for _, tr, te in gerar_folds(df, "BLOCOS", n_blocos))
            imp = np.mean(res, axis=0)
            for c, v in zip(cols, imp):
                linhas.append(dict(tarefa=tarefa, modelo=nome, descritor=c, importancia=v))
    return pd.DataFrame(linhas)


# --------------------------------------------------------------------------- #
# 6. Não supervisionado
# --------------------------------------------------------------------------- #
def nao_supervisionado(df, X):
    Z = StandardScaler().fit_transform(SimpleImputer(strategy="median").fit_transform(X))
    fatores = {"freq": df["freq"].astype(str), "config": df["config"],
               "corrida": df["run"], "dia": df["dia"], "limiar": df["limiar"]}
    linhas, rotulos = [], {}
    for k in (2, 3, 7, 8, 14, 56):
        lab = KMeans(n_clusters=k, n_init=20, random_state=SEMENTE).fit_predict(Z)
        rotulos[k] = lab
        for fn, fv in fatores.items():
            linhas.append(dict(k=k, fator=fn, ARI=adjusted_rand_score(fv, lab),
                               NMI=normalized_mutual_info_score(fv, lab)))
    pca = PCA(n_components=2, random_state=SEMENTE).fit(Z)
    return pd.DataFrame(linhas), rotulos, pca.transform(Z), pca.explained_variance_ratio_


# --------------------------------------------------------------------------- #
# 7. Figuras
# --------------------------------------------------------------------------- #
def _matriz(ax, cm, rot_x, rot_y, titulo, normalizar=True):
    cmn = cm / cm.sum(1, keepdims=True).clip(min=1) if normalizar else cm
    ax.imshow(cmn, cmap=CMAP_SEQ, vmin=0, vmax=1 if normalizar else None, aspect="auto")
    ax.grid(False)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            if cm[i, j]:
                v = cmn[i, j]
                ax.text(j, i, f"{v:.2f}" if normalizar else f"{cm[i, j]}", ha="center",
                        va="center", fontsize=7, color="white" if v > 0.55 else TINTA)
    ax.set_xticks(range(len(rot_x)), rot_x, rotation=45, ha="right")
    ax.set_yticks(range(len(rot_y)), rot_y)
    ax.set_title(titulo)


def fig_freq_prev_real(pred, esquema, saida):
    g = pred[(pred.esquema == esquema) & (pred.tarefa == "freq")]
    fig, axs = plt.subplots(1, 3, figsize=(11, 3.6), sharey=True)
    for ax, mod in zip(axs, ["MLP", "RF", "GB"]):
        gm = g[g.modelo == mod]
        dados = [gm.loc[gm.freq == f, "pred"].astype(float) for f in FREQS]
        bp = ax.boxplot(dados, positions=FREQS, widths=5, showfliers=False, patch_artist=True)
        for b in bp["boxes"]:
            b.set(facecolor=COR_MODELO[mod], alpha=0.35, edgecolor=COR_MODELO[mod])
        for k in ("whiskers", "caps", "medians"):
            for l in bp[k]:
                l.set(color=COR_MODELO[mod], linewidth=1.2)
        # mediana por corrida (pontos)
        gr = gm.groupby("run").agg(f=("freq", "first"), p=("pred", "median"))
        ax.scatter(gr.f + np.random.default_rng(0).uniform(-1.5, 1.5, len(gr)), gr.p,
                   s=14, color=TINTA, zorder=3, linewidths=0, label="mediana da corrida")
        ax.plot([15, 95], [15, 95], color=TINTA2, lw=1, ls="--", label="ideal")
        mae = np.mean(np.abs(gm.pred.astype(float) - gm.freq))
        ax.set_title(f"{mod}  (MAE janela = {mae:.1f} Hz)")
        ax.set_xlabel("Frequência real (Hz)")
        ax.set_xticks(FREQS, FREQS)
        ax.set_xlim(14, 96)
    axs[0].set_ylabel("Frequência prevista (Hz)")
    axs[0].legend(loc="upper left", fontsize=7)
    fig.suptitle(f"Frequência do motor: previsto × real - validação {esquema}", fontsize=10)
    fig.savefig(saida / f"fig_freq_previsto_real_{esquema}.png")
    plt.close(fig)


def fig_confusoes(pred, esquema, saida):
    fig, axs = plt.subplots(2, 3, figsize=(13, 8))
    for j, mod in enumerate(["MLP", "RF", "GB"]):
        g = pred[(pred.esquema == esquema) & (pred.modelo == mod)]
        gf = g[g.tarefa == "freq"]
        cm = confusion_matrix(gf.freq, arred_freq(gf.pred), labels=FREQS)
        _matriz(axs[0, j], cm, FREQS, FREQS, f"{mod} - frequência (Hz, arredondada)")
        gc = g[g.tarefa == "config"]
        cm = confusion_matrix(gc.config.astype(str), gc.pred.astype(str), labels=CONFIGS)
        _matriz(axs[1, j], cm, CONFIGS, CONFIGS, f"{mod} - válvulas (rec-suc %)")
    for ax in axs[:, 0]:
        ax.set_ylabel("real")
    for ax in axs[1, :]:
        ax.set_xlabel("previsto")
    fig.suptitle(f"Matrizes de confusão por janela (linhas normalizadas) - validação {esquema}",
                 fontsize=10)
    fig.tight_layout()
    fig.savefig(saida / f"fig_confusao_{esquema}.png")
    plt.close(fig)


def fig_importancias(imp, saida, top=15):
    fig, axs = plt.subplots(1, 2, figsize=(12, 5.2))
    for ax, tar, tit in zip(axs, ["freq", "config"],
                            ["Frequência (queda do -MAE, Hz)", "Válvulas (queda da acurácia balanceada)"]):
        g = imp[imp.tarefa == tar].pivot(index="descritor", columns="modelo", values="importancia")
        g = g.loc[g.mean(axis=1).sort_values(ascending=False).index[:top]][::-1]
        y = np.arange(len(g))
        h = 0.26
        for i, mod in enumerate(["MLP", "RF", "GB"]):
            ax.barh(y + (i - 1) * h, g[mod], height=h - 0.03, color=COR_MODELO[mod], label=mod)
        ax.set_yticks(y, g.index)
        ax.set_title(tit)
        ax.axvline(0, color=TINTA2, lw=0.8)
        ax.grid(axis="y", visible=False)
    axs[0].legend(loc="lower right")
    fig.suptitle(f"Importância por permutação (média dos {imp.attrs.get('n_blocos', '')} blocos de teste), "
                 f"{top} principais descritores", fontsize=10)
    fig.tight_layout()
    fig.savefig(saida / "fig_importancia_descritores.png")
    plt.close(fig)


def fig_nao_sup(df, ns, rotulos, Zp, var, saida):
    fig, axs = plt.subplots(1, 3, figsize=(14, 4.4))
    sc = axs[0].scatter(Zp[:, 0], Zp[:, 1], c=df.freq, cmap=CMAP_SEQ, s=5, vmin=0, vmax=95)
    fig.colorbar(sc, ax=axs[0], label="frequência (Hz)")
    axs[0].set_title("PCA dos descritores, cor = frequência")
    cores = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6"]
    for cfg, cor in zip(CONFIGS, cores):
        m = (df.config == cfg).to_numpy()
        axs[1].scatter(Zp[m, 0], Zp[m, 1], s=4, color=cor, label=cfg, alpha=0.7, linewidths=0)
    axs[1].legend(fontsize=7, markerscale=3, title="rec-suc", title_fontsize=7)
    axs[1].set_title("PCA, cor = válvulas (pontos sobrepostos)")
    for ax in axs[:2]:
        ax.set_xlabel(f"PC1 ({var[0]:.0%})")
        ax.set_ylabel(f"PC2 ({var[1]:.0%})")
    # contingência k=8 × frequência
    tab = pd.crosstab(rotulos[8], df.freq)
    _matriz(axs[2], tab.to_numpy().T, [f"c{i}" for i in tab.index], FREQS,
            "k-means k=8 × frequência (fração por linha)")
    axs[2].set_xlabel("cluster")
    axs[2].set_ylabel("frequência real (Hz)")
    fig.tight_layout()
    fig.savefig(saida / "fig_nao_supervisionado.png")
    plt.close(fig)


def fig_descritores(df, cols, saida):
    sel = [c for c in ["A_frac_maxdur", "A_log_energia_s", "A_fm_kHz_med",
                       "B_log_dur_p50", "B_frac_sat", "B_log_energia_s"] if c in cols]
    cores = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6"]
    fig, axs = plt.subplots(2, 3, figsize=(13, 6.5))
    for ax, c in zip(axs.flat, sel):
        g = df.groupby(["config", "freq"])[c].median().unstack(0)
        for cfg, cor in zip(CONFIGS, cores):
            if cfg in g:
                ax.plot(g.index, g[cfg], marker="o", ms=3.5, lw=1.5, color=cor, label=cfg)
        ax.set_title(c)
        ax.set_xticks(FREQS, FREQS)
        ax.set_xlabel("frequência (Hz)")
    h, l = axs[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=7, fontsize=8, title="válvulas rec-suc (%)",
               title_fontsize=8, bbox_to_anchor=(0.5, -0.01))
    fig.suptitle("Mediana por corrida de descritores selecionados", fontsize=10)
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.savefig(saida / "fig_descritores_por_condicao.png")
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Relatório automático
# --------------------------------------------------------------------------- #
def relatorio(corridas, df, met, ns, imp, args, saida, harm_info):
    tot = met[met.fold == "TOTAL"].copy()
    L = [f"# Relatório automático - mlp_ea.py\n",
         f"Gerado em {pd.Timestamp.now():%Y-%m-%d %H:%M}. Janela = {args.janela:g} s, "
         f"{args.blocos} blocos por corrida, harmonização de limiar: {harm_info}.\n",
         f"- Corridas: {len(corridas)} (1 por condição); janelas: {len(df)}; "
         f"descritores: {len(colunas_descritores(df))}.",
         f"- Hits com ROVL=1: {np.mean(np.concatenate([c['hits'].rovl for c in corridas])):.1%}; "
         f"SAT=1: {np.mean(np.concatenate([c['hits'].sat for c in corridas])):.1%}; "
         f"MAXDUR=1: {np.mean(np.concatenate([c['hits'].maxdur for c in corridas])):.1%}.\n",
         "## Frequência do motor (regressão; nível = arredondado a 20..90 Hz)\n"]
    f = tot[tot.tarefa == "freq"][["esquema", "modelo", "MAE_Hz", "RMSE_Hz", "R2", "acc_nivel",
                                   "acc_10Hz", "MAE_Hz_corrida", "acc_nivel_corrida"]]
    L.append(f.round(3).to_markdown(index=False))
    L.append("\n## Configuração das válvulas (7 classes)\n")
    c = tot[tot.tarefa == "config"][["esquema", "modelo", "acc", "acc_bal", "F1_macro",
                                     "acc_rec", "acc_suc", "acc_corrida"]]
    L.append(c.round(3).to_markdown(index=False))
    L.append("\nReferência de acaso: acc (7 classes equilibradas) = 0.143; acc_rec e acc_suc "
             "chutando sempre 100 % = 0.571 (4 das 7 configurações têm 100 %).\n")
    L.append("\n## Métricas por fold (BLOCOS)\n")
    b = met[(met.esquema == "BLOCOS") & (met.fold != "TOTAL")]
    L.append(b.pivot_table(index="fold", columns=["tarefa", "modelo"],
                           values=["MAE_Hz", "acc"]).dropna(axis=1, how="all").round(3).to_markdown())
    L.append("\n## Não supervisionado (k-means sobre os descritores padronizados)\n")
    L.append(ns.pivot_table(index="k", columns="fator", values="ARI").round(3).to_markdown())
    L.append("\n(ARI; 0 = acaso, 1 = partição idêntica.)\n")
    L.append("\n## 10 descritores mais importantes (média dos 3 modelos)\n")
    for tar in ("freq", "config"):
        g = imp[imp.tarefa == tar].groupby("descritor").importancia.mean().sort_values(ascending=False)
        L.append(f"- {tar}: " + ", ".join(f"{k} ({v:.3g})" for k, v in g.head(10).items()))
    (saida / "relatorio_auto.md").write_text("\n".join(L) + "\n", encoding="utf-8")


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dados", default=str(AQUI / "Dados"))
    ap.add_argument("--saida", default=str(AQUI / "resultados"))
    ap.add_argument("--janela", type=float, default=5.0, help="duração da janela (s)")
    ap.add_argument("--blocos", type=int, default=5, help="blocos de tempo por corrida (BLOCOS)")
    ap.add_argument("--sem-harmonizar", action="store_true",
                    help="não descartar hits abaixo do maior limiar das sessões")
    ap.add_argument("--esquemas", default="BLOCOS,LORO,LOSO")
    ap.add_argument("--jobs", type=int, default=-1)
    args = ap.parse_args()

    saida = Path(args.saida)
    saida.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    print("1) Lendo corridas...")
    corridas = ler_corridas(Path(args.dados))
    print(f"   {len(corridas)} corridas, {sum(len(c['hits']) for c in corridas)} hits")
    thr_comum = (max(c["thr_a"] for c in corridas), max(c["thr_b"] for c in corridas))
    harm = not args.sem_harmonizar
    harm_info = (f"sim (A >= {thr_comum[0]:.2f} dBAE, B >= {thr_comum[1]:.2f} dBAE)"
                 if harm else "não")
    pd.DataFrame([{k: v for k, v in c.items() if k != "hits"} | {"n_hits": len(c["hits"])}
                  for c in corridas]).to_csv(saida / "corridas.csv", index=False)

    print(f"2-3) Janelas de {args.janela:g} s, harmonização: {harm_info}")
    df = janelar(corridas, args.janela, args.blocos, harm, thr_comum)
    cols = colunas_descritores(df)
    df.to_csv(saida / "janelas_descritores.csv", index=False)
    print(f"   {len(df)} janelas × {len(cols)} descritores; "
          f"faltantes: {df[cols].isna().mean().mean():.1%}")
    # descritores constantes não ajudam: remove
    cols = [c for c in cols if df[c].nunique(dropna=True) > 1]
    X = df[cols].to_numpy(float)

    print("4) Validação")
    preds = []
    for esq in args.esquemas.split(","):
        for tar in ("freq", "config"):
            if esq == "LOSO" and tar == "config":
                continue
            preds.append(validar(df, X, esq, tar, args.blocos, args.jobs))
    pred = pd.concat(preds, ignore_index=True)
    pred.to_csv(saida / "predicoes.csv", index=False)
    met = metricas(pred)
    met.to_csv(saida / "metricas.csv", index=False)

    print("5) Importância por permutação (BLOCOS)")
    imp = importancias(df, X, cols, args.blocos, args.jobs)
    imp.attrs["n_blocos"] = args.blocos
    imp.to_csv(saida / "importancias.csv", index=False)

    print("6) k-means / PCA")
    ns, rotulos, Zp, var = nao_supervisionado(df, X)
    ns.to_csv(saida / "nao_supervisionado_ARI_NMI.csv", index=False)
    for k in (7, 8):
        for fat in ("freq", "config"):
            pd.crosstab(rotulos[k], df[fat]).to_csv(saida / f"contingencia_k{k}_{fat}.csv")

    print("7) Figuras e relatório")
    for esq in pred.esquema.unique():
        fig_freq_prev_real(pred, esq, saida)
        if esq != "LOSO":
            fig_confusoes(pred, esq, saida)
    fig_importancias(imp, saida)
    fig_nao_sup(df, ns, rotulos, Zp, var, saida)
    fig_descritores(df, cols, saida)
    relatorio(corridas, df, met, ns, imp, args, saida, harm_info)

    tot = met[met.fold == "TOTAL"]
    with pd.option_context("display.width", 200, "display.max_columns", 30):
        print(tot.drop(columns=["fold"]).round(3).to_string(index=False))
    print(f"Concluído em {time.time() - t0:.0f} s. Saída: {saida}")


if __name__ == "__main__":
    main()
