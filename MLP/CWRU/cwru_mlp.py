# -*- coding: utf-8 -*-
"""
cwru_mlp.py - Diagnóstico de rolamentos (base CWRU, 48 kHz, drive end) com MLP, usando
descritores clássicos, de envelope e de hits no estilo MOMEA, sob validações com graus
crescentes de independência. Estudo complementar ao de EA na bomba (../MLP/mlp_ea.py).

Uso (a partir desta pasta):
    ..\\MLP\\.venv\\Scripts\\python.exe -I cwru_mlp.py

Dados: Dados/<rótulo>__<número>.mat, baixados de engineering.case.edu/bearingdatacenter
(48k Drive End Bearing Fault Data e Normal Baseline Data). Subconjunto: normal, pista
interna (IR), esfera (B) e pista externa centrada às 6h (OR@6), diâmetros 0,007/0,014/0,021",
cargas 0-3 HP. Lista em Dados/lista_arquivos.json.

Fluxo
  1. Lê o sinal do acelerômetro do lado do acionamento (DE) de cada gravação, usando a
     variável cujo número coincide com o do arquivo (alguns .mat trazem sinais de outras
     gravações; o 174.mat só contém X173).
  2. Detecta hits com a lógica do extrator do MOMEA (limiar fixo sobre |x - média|, fim do hit
     após DDT amostras sem cruzamento, bloqueio HLT re-armável, duração máxima), com os tempos
     escalados para 48 kHz.
  3. Divide cada gravação em janelas de 0,25 s e calcula três grupos de descritores:
     clássicos (RMS, curtose, fator de crista...), de envelope (amplitude do espectro do
     envelope nas frequências de defeito) e de hits (taxa, amplitude, energia, duração,
     contagens e a ordem do intervalo entre hits em relação à rotação).
  4. Treina MLP (mesmos hiperparâmetros do estudo da bomba) e floresta aleatória com três
     validações: janelas sorteadas (vaza a gravação), carga deixada de fora (LOLO) e diâmetro
     do defeito deixado de fora (LOSO).
  5. Avalia a MLP quantizada em int8 (mesma aritmética da implementação embarcada) no LOLO.
  6. Grava resultados/ (CSV, figuras, resumo.json).
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.io as sio
from scipy.signal import butter, hilbert, sosfiltfilt
from scipy.stats import kurtosis, skew
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from sklearn.base import clone  # noqa: E402
from sklearn.ensemble import RandomForestClassifier  # noqa: E402
from sklearn.impute import SimpleImputer  # noqa: E402
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix  # noqa: E402
from sklearn.model_selection import StratifiedKFold  # noqa: E402
from sklearn.neural_network import MLPClassifier  # noqa: E402
from sklearn.pipeline import make_pipeline  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

AQUI = Path(__file__).resolve().parent
DADOS = AQUI / "Dados"
SAIDA = AQUI / "resultados"
SEMENTE = 42
FS = 48_000.0
JANELA_S = 0.25
N_JAN = int(JANELA_S * FS)

# extrator de hits (lógica do MOMEA, tempos escalados para 48 kHz)
LIMIAR_G = 0.30            # ~1,4x o percentil 99,9 de |x| da máquina sã
K_FLUT = 1.5               # limiar flutuante: K x RMS da janela anterior (mínimo LIMIAR_G);
                           # escolhido por varredura (1,5-3; DDT 0,5-1 ms) pela concordância da
                           # ordem do intervalo entre hits com BPFO/BPFI (usa os rótulos: ver relatório)
DDT = int(1e-3 * FS)       # 48 amostras: fim do hit após 1 ms sem cruzamento
HLT = int(1e-3 * FS)       # 48 amostras: bloqueio após o fim, re-armável
MAXDUR = int(20e-3 * FS)   # 960 amostras

# rolamento 6205-2RS JEM SKF (drive end): múltiplos da frequência de rotação
ORDENS = {"BPFI": 5.4152, "BPFO": 3.5848, "BSF": 2.3570}
RPM_NOMINAL = {0: 1797, 1: 1772, 2: 1750, 3: 1730}
CLASSES = ["N", "IR", "B", "OR"]
COR = {"N": "#2a78d6", "IR": "#eb6834", "B": "#1baf7a", "OR": "#eda100"}
TINTA, TINTA2, GRADE = "#0b0b0b", "#52514e", "#e4e3df"
plt.rcParams.update({
    "figure.dpi": 110, "savefig.dpi": 200, "savefig.bbox": "tight", "font.size": 9,
    "axes.titlesize": 10, "axes.labelsize": 9, "axes.edgecolor": TINTA2,
    "axes.labelcolor": TINTA, "xtick.color": TINTA2, "ytick.color": TINTA2,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
    "grid.color": GRADE, "grid.linewidth": 0.6, "axes.axisbelow": True, "legend.frameon": False,
})


# --------------------------------------------------------------------------- #
# 1. Leitura
# --------------------------------------------------------------------------- #
def ler_gravacoes() -> list[dict]:
    grav = []
    for f in sorted(DADOS.glob("*.mat")):
        m_nome = re.fullmatch(r"(Normal|IR|B|OR)(\d{3})?(?:at6)?_(\d)__(\d+)\.mat", f.name)
        if not m_nome:
            print(f"  [aviso] nome fora do padrão: {f.name}")
            continue
        tipo, diam, carga, num = m_nome.group(1), m_nome.group(2), int(m_nome.group(3)), int(m_nome.group(4))
        m = sio.loadmat(f)
        k = f"X{num:03d}_DE_time"
        if k not in m:
            cand = [c for c in m if c.endswith("_DE_time")]
            if len(cand) != 1:
                raise SystemExit(f"{f.name}: variável DE ambígua {cand}")
            print(f"  [aviso] {f.name}: sem {k}; usando {cand[0]}")
            k = cand[0]
        x = np.ravel(m[k]).astype(np.float64)
        krpm = k.replace("_DE_time", "RPM")
        rpm = float(np.ravel(m[krpm])[0]) if krpm in m else float(RPM_NOMINAL[carga])
        grav.append(dict(arquivo=f.name, variavel=k, classe="N" if tipo == "Normal" else tipo,
                         diam=(int(diam) if diam else 0), carga=carga, rpm=rpm,
                         rpm_gravada=krpm in m, x=x))
    return grav


# --------------------------------------------------------------------------- #
# 2. Extrator de hits (lógica do MOMEA)
# --------------------------------------------------------------------------- #
def limiar_flutuante(x: np.ndarray) -> np.ndarray:
    """Limiar por amostra: K_FLUT x RMS da janela anterior (causal), no mínimo LIMIAR_G.
    Na lógica, equivale a uma média móvel de x^2 atualizada a cada janela."""
    u = x - x.mean()
    n_w = int(np.ceil(len(x) / N_JAN))
    rms = np.array([np.sqrt(np.mean(u[k * N_JAN:(k + 1) * N_JAN] ** 2)) for k in range(n_w)])
    ant = np.r_[rms[0], rms[:-1]]          # a primeira janela usa o próprio RMS
    thr_w = np.maximum(LIMIAR_G, K_FLUT * ant)
    return np.repeat(thr_w, N_JAN)[:len(x)]


def detectar_hits(x: np.ndarray, limiar=LIMIAR_G) -> pd.DataFrame:
    """Hits sobre |x - média|. limiar: escalar (fixo) ou vetor por amostra (flutuante).
    Retorna início, fim (último cruzamento), pico, subida, duração, contagens (cruzamentos
    ascendentes do limiar positivo), energia e indicadores."""
    u = x - x.mean()
    ativo = np.abs(u) > limiar
    if not ativo.any():
        return pd.DataFrame(columns=["ini", "fim", "pico", "subida", "dur", "cont", "energia",
                                     "maxdur", "rovl"])
    d = np.diff(ativo.astype(np.int8))
    ini_r = np.flatnonzero(d == 1) + 1
    fim_r = np.flatnonzero(d == -1)          # última amostra ativa de cada trecho
    if ativo[0]:
        ini_r = np.r_[0, ini_r]
    if ativo[-1]:
        fim_r = np.r_[fim_r, len(x) - 1]
    pos = np.flatnonzero(np.diff((u > limiar).astype(np.int8)) == 1) + 1   # cruzamentos +
    hits = []
    i, n_r = 0, len(ini_r)
    bloqueio_ate, rovl_pend = -1, False
    while i < n_r:
        s, e = ini_r[i], fim_r[i]
        if s <= bloqueio_ate:                 # cruzamento no HLT: re-arma o bloqueio
            bloqueio_ate = e + HLT
            rovl_pend = True
            i += 1
            continue
        ini, ult, j, md = s, e, i + 1, False
        while j < n_r and ini_r[j] - ult <= DDT:   # continua enquanto houver cruzamento no DDT
            ult = fim_r[j]
            j += 1
        if ult - ini + 1 > MAXDUR:            # trunca na duração máxima
            ult, md = ini + MAXDUR - 1, True
            while j > i + 1 and ini_r[j - 1] > ult:
                j -= 1
        seg = u[ini:ult + 1]
        k_pico = int(np.argmax(np.abs(seg)))
        cont = int(np.searchsorted(pos, ult, side="right") - np.searchsorted(pos, ini, side="left"))
        hits.append((ini, ult, float(np.abs(seg[k_pico])), k_pico, ult - ini + 1, max(cont, 1),
                     float(np.sum(seg ** 2) / FS), md, rovl_pend))
        rovl_pend = False
        fim_hit = ult + DDT                   # o hit termina DDT amostras após o último cruzamento
        bloqueio_ate = fim_hit + HLT
        i = j
        while i < n_r and ini_r[i] <= fim_hit:   # trechos dentro do DDT final já pertencem ao hit
            i += 1
    return pd.DataFrame(hits, columns=["ini", "fim", "pico", "subida", "dur", "cont", "energia",
                                       "maxdur", "rovl"])


# --------------------------------------------------------------------------- #
# 3. Descritores por janela
# --------------------------------------------------------------------------- #
SOS_ENV = butter(4, [1_000, 10_000], btype="bandpass", fs=FS, output="sos")


def desc_classicos(w: np.ndarray) -> dict:
    u = w - w.mean()
    rms = np.sqrt(np.mean(u ** 2))
    pico = np.max(np.abs(u))
    media_abs = np.mean(np.abs(u))
    return dict(c_rms=rms, c_pico=pico, c_crista=pico / rms, c_curtose=kurtosis(u, fisher=False),
                c_assimetria=skew(u), c_forma=rms / media_abs, c_impulso=pico / media_abs,
                c_folga=pico / np.mean(np.sqrt(np.abs(u))) ** 2)


def desc_envelope(w: np.ndarray, fr: float) -> dict:
    env = np.abs(hilbert(sosfiltfilt(SOS_ENV, w - w.mean())))
    env = env - env.mean()
    esp = np.abs(np.fft.rfft(env * np.hanning(len(env))))
    f = np.fft.rfftfreq(len(env), 1 / FS)
    ref = np.median(esp[(f > 10) & (f < 1000)]) + 1e-12
    out = dict(e_curtose_env=kurtosis(env, fisher=False))
    df_ = f[1] - f[0]
    for nome, o in ORDENS.items():
        for h in (1, 2):
            f0 = h * o * fr * (2 if nome == "BSF" else 1)   # esfera: impactos nas duas pistas
            k = (np.abs(f - f0) <= max(2.0, 0.01 * f0) + df_)
            out[f"e_{nome}_{h}x_dB"] = 20 * np.log10(esp[k].max() / ref)
    return out


def desc_hits(h: pd.DataFrame, fr: float, pref: str = "h_", limiar: float = LIMIAR_G) -> dict:
    """Mesmos descritores do estudo da bomba (amplitude, energia, duração, contagens,
    subida, frequência média) + ordem do intervalo entre hits. Sem hits: valores de 'ausência'."""
    n = len(h)
    thr_db = 20 * np.log10(limiar / 1e-3)
    v = dict(taxa=n / JANELA_S, frac_maxdur=(h["maxdur"].mean() if n else 0.0),
             frac_rovl=(h["rovl"].mean() if n else 0.0))
    if n == 0:
        v.update(amp_p10=thr_db, amp_p50=thr_db, amp_p90=thr_db, log_e_p50=0.0, log_e_p90=0.0,
                 log_dur_p50=0.0, log_cont_p50=0.0, log_sub_p50=0.0, fm_kHz=0.0, ordem_ihi=0.0,
                 cv_ihi=0.0)
        return {pref + k: val for k, val in v.items()}
    amp = 20 * np.log10(h["pico"].to_numpy() / 1e-3)          # dB re 1 mg
    dur_us = h["dur"].to_numpy() / FS * 1e6
    le = np.log10(h["energia"].to_numpy() * 1e6 + 1)
    v.update(amp_p10=np.percentile(amp, 10), amp_p50=np.percentile(amp, 50),
             amp_p90=np.percentile(amp, 90), log_e_p50=np.percentile(le, 50),
             log_e_p90=np.percentile(le, 90), log_dur_p50=np.percentile(np.log10(dur_us + 1), 50),
             log_cont_p50=np.percentile(np.log10(h["cont"].to_numpy() + 1), 50),
             log_sub_p50=np.percentile(np.log10(h["subida"].to_numpy() / FS * 1e6 + 1), 50),
             fm_kHz=np.median(h["cont"].to_numpy() / dur_us * 1e3))
    if n >= 3:
        ihi = np.diff(h["ini"].to_numpy()) / FS
        v.update(ordem_ihi=(1.0 / np.median(ihi)) / fr, cv_ihi=np.std(ihi) / np.mean(ihi))
    else:
        v.update(ordem_ihi=0.0, cv_ihi=0.0)
    return {pref + k: val for k, val in v.items()}


def janelar(grav: list[dict]) -> tuple[pd.DataFrame, pd.DataFrame]:
    linhas, inv = [], []
    for g in grav:
        x, fr = g["x"], g["rpm"] / 60.0
        thr_fl = limiar_flutuante(x)
        h = detectar_hits(x, thr_fl)             # limiar flutuante (principal)
        hf = detectar_hits(x, LIMIAR_G)          # limiar fixo, como o MOMEA hoje
        n_w = len(x) // N_JAN
        # saturação: várias amostras repetidas exatamente no extremo do registro
        n_max = int(np.sum(x >= x.max() - 1e-9))
        n_min = int(np.sum(x <= x.min() + 1e-9))
        sat = (n_max if n_max > 3 else 0) + (n_min if n_min > 3 else 0)
        inv.append(dict(arquivo=g["arquivo"], variavel=g["variavel"], classe=g["classe"],
                        diam=g["diam"], carga=g["carga"], rpm=g["rpm"], rpm_gravada=g["rpm_gravada"],
                        duracao_s=len(x) / FS, janelas=n_w, hits=len(h), hits_fixo=len(hf),
                        hits_por_s=len(h) / (len(x) / FS),
                        frac_maxdur_fixo=float(hf["maxdur"].mean()) if len(hf) else 0.0,
                        frac_maxdur=float(h["maxdur"].mean()) if len(h) else 0.0,
                        amostras_no_extremo=sat,
                        extremo_g=float(np.max(np.abs(x)))))
        for k in range(n_w):
            a, b = k * N_JAN, (k + 1) * N_JAN
            w = x[a:b]
            hk = h[(h["ini"] >= a) & (h["ini"] < b)]
            hfk = hf[(hf["ini"] >= a) & (hf["ini"] < b)]
            lin = dict(gravacao=g["arquivo"], classe=g["classe"], diam=g["diam"],
                       carga=g["carga"], janela=k, rotulo10=(g["classe"] if g["classe"] == "N"
                                                              else f"{g['classe']}{g['diam']:03d}"))
            lin.update(desc_classicos(w))
            lin.update(desc_envelope(w, fr))
            lin.update(desc_hits(hk, fr, "h_", float(thr_fl[a])))
            lin.update(desc_hits(hfk, fr, "hf_", LIMIAR_G))
            lin["limiar_flut_g"] = float(thr_fl[a])
            linhas.append(lin)
    return pd.DataFrame(linhas), pd.DataFrame(inv)


# --------------------------------------------------------------------------- #
# 4. Modelos e validação
# --------------------------------------------------------------------------- #
def modelos() -> dict:
    pre = lambda: [SimpleImputer(strategy="median"), StandardScaler()]  # noqa: E731
    return {
        "MLP": make_pipeline(*pre(), MLPClassifier(
            hidden_layer_sizes=(64, 32), alpha=1e-3, learning_rate_init=1e-3, early_stopping=True,
            validation_fraction=0.15, max_iter=2000, n_iter_no_change=30, random_state=SEMENTE)),
        "RF": make_pipeline(*pre(), RandomForestClassifier(
            n_estimators=300, min_samples_leaf=2, max_features="sqrt", n_jobs=1,
            random_state=SEMENTE)),
    }


def folds(df: pd.DataFrame, esquema: str, alvo: str):
    if esquema == "sorteada":
        skf = StratifiedKFold(5, shuffle=True, random_state=SEMENTE)
        for k, (tr, te) in enumerate(skf.split(df, df[alvo])):
            yield f"k{k}", tr, te
    elif esquema == "LOLO":                      # carga deixada de fora
        for c in sorted(df["carga"].unique()):
            te = (df["carga"] == c).to_numpy()
            yield f"carga{c}", np.flatnonzero(~te), np.flatnonzero(te)
    elif esquema == "LOSO":                      # diâmetro deixado de fora (normal: uma carga)
        for i, d in enumerate((7, 14, 21)):
            te = ((df["diam"] == d) | ((df["classe"] == "N") & (df["carga"] == i))).to_numpy()
            yield f"d{d:03d}", np.flatnonzero(~te), np.flatnonzero(te)


GRUPOS = {"classicos": "c_", "envelope": "e_", "hits_fixo": "hf_", "hits": "h_"}


def colunas(df, grupo):
    if grupo == "todos":
        return [c for c in df.columns if c[:2] in ("c_", "e_", "h_")]
    return [c for c in df.columns if c.startswith(GRUPOS[grupo])]


def avaliar(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    met, pred = [], []
    tarefas = [("4 classes", "classe", ("sorteada", "LOLO", "LOSO")),
               ("10 classes", "rotulo10", ("sorteada", "LOLO"))]
    for nome_t, alvo, esquemas in tarefas:
        for esq in esquemas:
            for grupo in ("classicos", "envelope", "hits_fixo", "hits", "todos"):
                cols = colunas(df, grupo)
                X, y = df[cols].to_numpy(float), df[alvo].to_numpy()
                for nome_m, mod in modelos().items():
                    yp = np.empty(len(df), dtype=object)
                    for fold, tr, te in folds(df, esq, alvo):
                        m = clone(mod).fit(X[tr], y[tr])
                        yp[te] = m.predict(X[te])
                    ok = yp != None  # noqa: E711
                    d = df.loc[ok, ["gravacao", alvo]].assign(pred=yp[ok])
                    voto = d.groupby("gravacao").agg(real=(alvo, "first"),
                                                     pred=("pred", lambda s: s.mode().iloc[0]))
                    met.append(dict(tarefa=nome_t, esquema=esq, grupo=grupo, modelo=nome_m,
                                    acc=accuracy_score(y[ok], yp[ok]),
                                    acc_bal=balanced_accuracy_score(y[ok], yp[ok]),
                                    acc_gravacao=accuracy_score(voto["real"], voto["pred"])))
                    if nome_t == "4 classes":
                        pred.append(pd.DataFrame(dict(esquema=esq, grupo=grupo, modelo=nome_m,
                                                      real=y, pred=yp)))
                    print(f"  {nome_t:10s} {esq:8s} {grupo:9s} {nome_m}: acc={met[-1]['acc']:.3f} "
                          f"(por gravação {met[-1]['acc_gravacao']:.3f})")
    return pd.DataFrame(met), pd.concat(pred, ignore_index=True)


# --------------------------------------------------------------------------- #
# 5. MLP int8 (mesma aritmética da implementação embarcada)
# --------------------------------------------------------------------------- #
def avaliar_int8(df: pd.DataFrame, grupo: str) -> pd.DataFrame:
    sys.path.insert(0, str(AQUI.parent / "MLP" / "embarcado"))
    import exportar_modelo as em  # quantizar, quant_entrada, int8_ref
    cols = colunas(df, grupo)
    X, y = df[cols].to_numpy(float), df["classe"].to_numpy()
    out = []
    for fold, tr, te in folds(df, "LOLO", "classe"):
        pipe = clone(modelos()["MLP"]).fit(X[tr], y[tr])
        imp, sc, mlp = pipe.steps[0][1], pipe.steps[1][1], pipe.steps[2][1]
        med, mu, sd = imp.statistics_, sc.mean_, sc.scale_
        W = [w.astype(np.float64) for w in mlp.coefs_]
        b = [v.astype(np.float64) for v in mlp.intercepts_]
        Xtr_std = (np.where(np.isnan(X[tr]), med, X[tr]) - mu) / sd
        q = em.quantizar(Xtr_std, W, b)            # calibração só no treino do fold
        xq = em.quant_entrada(X[te], med, mu, sd, q["s_in"])
        acc_q = em.int8_ref(xq, q)
        y_f = pipe.predict(X[te])
        y_q = mlp.classes_[acc_q.argmax(1)]
        out.append(dict(fold=fold, n=len(te), acc_float=accuracy_score(y[te], y_f),
                        acc_int8=accuracy_score(y[te], y_q), concordancia=float(np.mean(y_f == y_q)),
                        n_entradas=len(cols), macs=len(cols) * 64 + 64 * 32 + 32 * 4))
    return pd.DataFrame(out)


# --------------------------------------------------------------------------- #
# 6. Figuras
# --------------------------------------------------------------------------- #
def fig_sinais(grav, saida):
    sel = {("N", 0), ("IR", 7), ("B", 7), ("OR", 7)}
    gs = [g for g in grav if g["carga"] == 0 and (g["classe"], g["diam"]) in sel]
    gs.sort(key=lambda g: CLASSES.index(g["classe"]))
    fig, axs = plt.subplots(len(gs), 1, figsize=(7.2, 6.4), sharex=True)
    t0, dur = 0.20, 0.06
    for ax, g in zip(axs, gs):
        a, b = int(t0 * FS), int((t0 + dur) * FS)
        u = g["x"] - g["x"].mean()
        t = np.arange(a, b) / FS * 1e3
        ax.plot(t, u[a:b], color=COR[g["classe"]], lw=0.7)
        thr = limiar_flutuante(g["x"])
        h = detectar_hits(g["x"], thr)
        hf = detectar_hits(g["x"], LIMIAR_G)
        for s in (+1, -1):
            ax.plot(t, s * thr[a:b], color=TINTA2, lw=0.9, ls="--")
            ax.axhline(s * LIMIAR_G, color=TINTA2, lw=0.8, ls=":")
        hh = h[(h["fim"] >= a) & (h["ini"] < b)]
        n_f = int(np.sum((hf["fim"] >= a) & (hf["ini"] < b)))
        for _, r in hh.iterrows():
            ax.axvspan(max(r["ini"], a) / FS * 1e3, min(r["fim"] + 1, b) / FS * 1e3,
                       color=GRADE, alpha=0.9, lw=0)
        nome = {"N": "Normal", "IR": "Pista interna 0,007\"", "B": "Esfera 0,007\"",
                "OR": "Pista externa 0,007\""}[g["classe"]]
        ax.set_title(f"{nome}, carga 0: {len(hh)} hits com limiar flutuante (sombreados), "
                     f"{n_f} com limiar fixo", loc="left", fontsize=8.5)
        ax.set_ylabel("g")
    axs[-1].set_xlabel("Tempo (ms)")
    fig.tight_layout()
    fig.savefig(saida / "fig_sinais_hits.png")
    plt.close(fig)


def fig_ordem(df, saida):
    fig, axs = plt.subplots(2, 1, figsize=(7.0, 5.0), sharex=True)
    bins = np.arange(0, 10.01, 0.1)
    for ax, pref, tit in ((axs[0], "hf_", "(a) Limiar fixo de 0,3 g (como o MOMEA hoje)"),
                          (axs[1], "h_", "(b) Limiar flutuante: 1,5 × RMS da janela anterior (mínimo 0,3 g)")):
        for c in ("IR", "B", "OR"):
            v = df.loc[(df["classe"] == c) & (df[pref + "ordem_ihi"] > 0), pref + "ordem_ihi"]
            ax.hist(v, bins=bins, histtype="step", lw=2, color=COR[c], label=f"{c} ({len(v)} janelas)")
        for nome, o, mult, dx in (("BPFO", ORDENS["BPFO"], 1, -0.08), ("2×BSF", ORDENS["BSF"], 2, -0.08),
                                  ("BPFI", ORDENS["BPFI"], 1, 0.08)):
            ax.axvline(o * mult, color=TINTA2, ls="--", lw=0.9)
            ax.text(o * mult + dx, 0.97, nome, transform=ax.get_xaxis_transform(), fontsize=8,
                    color=TINTA2, va="top", ha="right" if dx < 0 else "left")
        ax.set_title(tit, loc="left", fontsize=9)
        ax.set_ylabel("Janelas de 0,25 s")
        ax.legend(loc="upper right", fontsize=8)
    axs[1].set_xlabel("Ordem do intervalo entre hits: (1 / mediana do intervalo) / frequência de rotação")
    fig.tight_layout()
    fig.savefig(saida / "fig_ordem_ihi.png")
    plt.close(fig)


def fig_acuracia(met, saida):
    m = met[(met["tarefa"] == "4 classes") & (met["modelo"] == "MLP")]
    grupos = ["classicos", "envelope", "hits_fixo", "hits", "todos"]
    nomes_g = {"classicos": "Clássicos", "envelope": "Envelope", "hits_fixo": "Hits\nlimiar fixo",
               "hits": "Hits\nlimiar flutuante", "todos": "Clássicos + envelope\n+ hits flutuante"}
    esquemas = ["sorteada", "LOLO", "LOSO"]
    nomes_e = {"sorteada": "Janelas sorteadas", "LOLO": "Carga deixada de fora",
               "LOSO": "Diâmetro deixado de fora"}
    cores = ["#2a78d6", "#eb6834", "#1baf7a"]
    fig, ax = plt.subplots(figsize=(7.4, 3.5))
    x = np.arange(len(grupos))
    lw = 0.26
    for i, (e, cor) in enumerate(zip(esquemas, cores)):
        v = [m[(m["esquema"] == e) & (m["grupo"] == g)]["acc"].iloc[0] * 100 for g in grupos]
        ax.bar(x + (i - 1) * lw, v, width=lw - 0.03, color=cor, label=nomes_e[e])
        for xi, vi in zip(x + (i - 1) * lw, v):
            ax.text(xi, vi + 1, f"{vi:.0f}", ha="center", fontsize=7.5, color=TINTA)
    ax.axhline(25, color=TINTA2, lw=0.8, ls=":")
    ax.text(-0.45, 26.5, "acaso (25 %)", fontsize=7.5, color=TINTA2, ha="left")
    ax.set_xticks(x, [nomes_g[g] for g in grupos])
    ax.set_ylim(0, 112)
    ax.set_ylabel("Acurácia por janela (%)")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=3)
    ax.grid(axis="x", visible=False)
    fig.tight_layout()
    fig.savefig(saida / "fig_acuracia_validacoes.png")
    plt.close(fig)


def fig_confusao(pred, saida):
    fig, axs = plt.subplots(1, 3, figsize=(7.6, 2.8))
    for ax, esq in zip(axs, ("sorteada", "LOLO", "LOSO")):
        p = pred[(pred["esquema"] == esq) & (pred["grupo"] == "hits") & (pred["modelo"] == "MLP")]
        p = p[p["pred"].notna()]
        cm = confusion_matrix(p["real"], p["pred"], labels=CLASSES, normalize="true") * 100
        ax.imshow(cm, cmap="Blues", vmin=0, vmax=100)
        for i in range(4):
            for j in range(4):
                ax.text(j, i, f"{cm[i, j]:.0f}", ha="center", va="center", fontsize=8,
                        color="white" if cm[i, j] > 60 else TINTA)
        ax.set_xticks(range(4), CLASSES)
        ax.set_yticks(range(4), CLASSES)
        ax.set_title({"sorteada": "Sorteada", "LOLO": "Carga de fora", "LOSO": "Diâmetro de fora"}[esq])
        ax.set_xlabel("Previsto")
        ax.grid(False)
    axs[0].set_ylabel("Real")
    fig.tight_layout()
    fig.savefig(saida / "fig_confusao_hits.png")
    plt.close(fig)


# --------------------------------------------------------------------------- #
def main():
    SAIDA.mkdir(exist_ok=True)
    print("1) Leitura")
    grav = ler_gravacoes()
    print(f"   {len(grav)} gravações")
    print("2-3) Hits e descritores por janela")
    df, inv = janelar(grav)
    df.to_csv(SAIDA / "janelas.csv", index=False)
    inv.to_csv(SAIDA / "inventario.csv", index=False)
    print(f"   {len(df)} janelas; por classe: {df['classe'].value_counts().to_dict()}")
    print("4) Validação")
    met, pred = avaliar(df)
    met.to_csv(SAIDA / "metricas.csv", index=False)
    print("5) MLP int8 (LOLO)")
    q8 = pd.concat([avaliar_int8(df, g).assign(grupo=g) for g in ("hits", "todos")],
                   ignore_index=True)
    q8.to_csv(SAIDA / "int8_lolo.csv", index=False)
    print(q8.to_string(index=False))
    print("6) Figuras")
    fig_sinais(grav, SAIDA)
    fig_ordem(df, SAIDA)
    fig_acuracia(met, SAIDA)
    fig_confusao(pred, SAIDA)
    resumo = dict(
        gravacoes=len(grav), janelas=len(df), janelas_por_classe=df["classe"].value_counts().to_dict(),
        limiar_g=LIMIAR_G, ddt=DDT, hlt=HLT, maxdur=MAXDUR, janela_s=JANELA_S,
        gravacoes_com_saturacao=inv.loc[inv["amostras_no_extremo"] > 0, "arquivo"].tolist(),
        ordem_ihi_mediana={c: float(df.loc[(df["classe"] == c) & (df["h_ordem_ihi"] > 0),
                                         "h_ordem_ihi"].median()) for c in ("IR", "B", "OR")},
        int8=q8.groupby("grupo")[["acc_float", "acc_int8", "concordancia"]].mean().to_dict(),
    )
    (SAIDA / "resumo.json").write_text(json.dumps(resumo, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(resumo, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
