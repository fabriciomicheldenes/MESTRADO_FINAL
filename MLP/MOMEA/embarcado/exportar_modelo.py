"""
Exporta as MLPs (frequência e configuração de válvulas) para C embarcado.

Uso (a partir de MLP/):
    .venv\\Scripts\\python.exe embarcado\\exportar_modelo.py

Passos:
  1. Lê resultados/janelas_descritores.csv (36 descritores por janela de 5 s).
  2. Treina os modelos finais (mesmos hiperparâmetros de mlp_ea.py) com TODAS as janelas.
     Isso é o modelo "de implantação"; o desempenho fora da amostra é o do estudo
     principal (BLOCOS/LORO/LOSO), não o deste script.
  3. Gera:
       gerado/mlp_pesos_f32.h   pesos float32 (imputação + padronização dobradas na 1a camada)
       gerado/mlp_pesos_q8.h    pesos int8 / bias int32 / requantização inteira (bit-exata)
       gerado/vetores_teste.bin todas as janelas: entrada bruta + saídas esperadas (float e int8)
       gerado/vetores_tb.h      subconjunto para o testbench HLS
       gerado/resumo_export.txt concordância sklearn x float x int8
  O modelo int8 é emulado aqui em numpy com aritmética inteira idêntica à do C, de modo que o
  C e o HLS devem reproduzir as saídas int8 exatamente.
"""
from __future__ import annotations

import struct
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import TransformedTargetRegressor
from sklearn.impute import SimpleImputer
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

SEMENTE = 42
AQUI = Path(__file__).resolve().parent
RAIZ = AQUI.parent
SAIDA = AQUI / "gerado"
N_TB = 256          # vetores no testbench HLS
CLIP_IN = 8.0       # entradas padronizadas saturadas em +-8 desvios antes de quantizar
SHIFT = 31          # requantização: y = (acc * M + 2^(S-1)) >> S, com M < 2^31


# --------------------------------------------------------------------------- #
def treinar(X: np.ndarray, df: pd.DataFrame):
    pre = lambda: [SimpleImputer(strategy="median"), StandardScaler()]  # noqa: E731
    reg = make_pipeline(*pre(), TransformedTargetRegressor(
        regressor=MLPRegressor(hidden_layer_sizes=(64, 32), alpha=1e-3, learning_rate_init=1e-3,
                               early_stopping=True, validation_fraction=0.15, max_iter=2000,
                               n_iter_no_change=30, random_state=SEMENTE),
        transformer=StandardScaler()))
    cls = make_pipeline(*pre(), MLPClassifier(
        hidden_layer_sizes=(64, 32), alpha=1e-3, learning_rate_init=1e-3, early_stopping=True,
        validation_fraction=0.15, max_iter=2000, n_iter_no_change=30, random_state=SEMENTE))
    reg.fit(X, df["freq"].to_numpy(float))
    cls.fit(X, df["config"].astype(str).to_numpy())
    return reg, cls


def extrair(pipe):
    """Retorna (medianas, média, escala, [W], [b], (y_media, y_escala) ou None, classes)."""
    imp, sc, est = pipe.steps[0][1], pipe.steps[1][1], pipe.steps[2][1]
    y_tr = None
    if isinstance(est, TransformedTargetRegressor):
        y_tr = (float(est.transformer_.mean_[0]), float(est.transformer_.scale_[0]))
        est = est.regressor_
    assert est.activation == "relu"
    assert len(imp.statistics_) == len(sc.mean_), "imputador descartou colunas vazias"
    W = [w.astype(np.float64) for w in est.coefs_]
    b = [v.astype(np.float64) for v in est.intercepts_]
    classes = list(getattr(est, "classes_", []))
    return imp.statistics_.astype(np.float64), sc.mean_, sc.scale_, W, b, y_tr, classes


# --------------------------------------------------------------------------- #
# Referência float (mesma ordem de operações do C)
# --------------------------------------------------------------------------- #
def float_ref(Xraw, med, W, b):
    x = np.where(np.isnan(Xraw), med, Xraw)
    h = x
    for k, (w, bb) in enumerate(zip(W, b)):
        h = h @ w + bb
        if k < len(W) - 1:
            h = np.maximum(h, 0.0)
    return h


def dobrar_escala(med, mu, sd, W, b, y_tr):
    """Dobra (x-mu)/sd na 1a camada e a desnormalização do alvo na última (versão float)."""
    W = [w.copy() for w in W]
    b = [v.copy() for v in b]
    b[0] = b[0] - (mu / sd) @ W[0]
    W[0] = W[0] / sd[:, None]
    if y_tr is not None:
        W[-1] = W[-1] * y_tr[1]
        b[-1] = b[-1] * y_tr[1] + y_tr[0]
    return W, b


# --------------------------------------------------------------------------- #
# Quantização int8 simétrica por camada, emulação inteira bit-exata
# --------------------------------------------------------------------------- #
def multiplicador(real: float) -> int:
    m = int(round(real * (1 << SHIFT)))
    assert 0 < m < (1 << 31), f"multiplicador fora da faixa: {real}"
    return m


def requant(acc: np.ndarray, M: int) -> np.ndarray:
    """ReLU + (acc*M + 2^(S-1)) >> S, saturado em [0,127]. Igual ao C."""
    acc = np.maximum(acc.astype(np.int64), 0)
    y = (acc * M + (1 << (SHIFT - 1))) >> SHIFT
    return np.minimum(y, 127).astype(np.int64)


def quantizar(Xcal_std, W, b):
    """Escalas: entrada fixa (CLIP_IN/127); pesos por camada; ativações pelo máximo na calibração."""
    s_in = CLIP_IN / 127.0
    sw = [float(np.abs(w).max()) / 127.0 for w in W]
    Wq = [np.clip(np.round(w / s), -127, 127).astype(np.int64) for w, s in zip(W, sw)]
    # escalas de ativação a partir da rede float na calibração
    h = np.clip(Xcal_std, -CLIP_IN, CLIP_IN)
    s_act = []
    for k in range(len(W) - 1):
        h = np.maximum(h @ W[k] + b[k], 0.0)
        s_act.append(float(np.percentile(h.max(axis=1), 99.9)) / 127.0)
    s_prev = [s_in] + s_act
    bq = [np.round(bb / (sp * s)).astype(np.int64) for bb, sp, s in zip(b, s_prev, sw)]
    M = [multiplicador(s_prev[k] * sw[k] / s_act[k]) for k in range(len(W) - 1)]
    s_out = s_prev[-1] * sw[-1]  # escala do acumulador da última camada
    return dict(s_in=s_in, sw=sw, Wq=Wq, bq=bq, M=M, s_out=s_out, s_act=s_act)


def quant_entrada(Xraw, med, mu, sd, s_in):
    """Emula em float32 exatamente as operações do C (mlp_quantiza_entrada)."""
    f = np.float32
    x = Xraw.astype(f)
    x = np.where(np.isnan(x), med.astype(f), x)
    z = (x - mu.astype(f)) * (1.0 / sd).astype(f)
    z = np.clip(z, f(-CLIP_IN), f(CLIP_IN))
    v = np.rint(z * f(1.0 / s_in))           # rintf: arredonda metade para par
    return np.clip(v, -127, 127).astype(np.int64)


def int8_ref(xq, q):
    h = xq
    n = len(q["Wq"])
    for k in range(n):
        acc = h @ q["Wq"][k] + q["bq"][k]
        h = requant(acc, q["M"][k]) if k < n - 1 else acc
    return h  # acumulador int32 da última camada


# --------------------------------------------------------------------------- #
# Escrita de cabeçalhos C
# --------------------------------------------------------------------------- #
def arr_c(tipo, nome, a, fmt, por_linha=12):
    a = np.asarray(a).ravel()
    linhas = []
    for i in range(0, len(a), por_linha):
        linhas.append("    " + ", ".join(fmt(v) for v in a[i:i + por_linha]))
    return f"static const {tipo} {nome}[{len(a)}] = {{\n" + ",\n".join(linhas) + "\n};\n"


def ff(v) -> str:
    """Literal float32 do C que reproduz exatamente np.float32(v)."""
    s = f"{float(np.float32(v)):.9g}"
    if not any(c in s for c in ".en"):
        s += ".0"
    return s + "f"



fi = lambda v: f"{int(v)}"  # noqa: E731


def header_f32(path, nomes, mods):
    out = ["/* Gerado por exportar_modelo.py - NÃO EDITAR. MLP 36-64-32-k, float32.",
           " * Imputação (mediana) aplicada antes; padronização dobrada na 1a camada;",
           " * desnormalização do alvo (regressor) dobrada na última camada.",
           " * Pesos em ordem [entrada][saída] (linha = entrada). */",
           "#ifndef MLP_PESOS_F32_H", "#define MLP_PESOS_F32_H", ""]
    out.append(f"#define MLP_N_IN {len(nomes)}")
    out.append("/* Descritores na ordem esperada:")
    out += [f" *  {i:2d} {n}" for i, n in enumerate(nomes)]
    out.append(" */")
    for tag, m in mods.items():
        out.append(f"\n/* ---- modelo {tag} ---- */")
        out.append(arr_c("float", f"{tag}_MEDIANA", m["med"], ff))
        for k, (w, bb) in enumerate(zip(m["Wf"], m["bf"])):
            out.append(arr_c("float", f"{tag}_W{k}", w, ff))
            out.append(arr_c("float", f"{tag}_B{k}", bb, ff))
    path.write_text("\n".join(out) + "\n#endif\n", encoding="utf-8")


def header_q8(path, mods):
    out = ["/* Gerado por exportar_modelo.py - NÃO EDITAR. MLP 36-64-32-k, int8.",
           " * Entrada: z = clip((x-media)/escala, +-CLIP) ; xq = round(z / S_IN) em int8.",
           " * Camadas ocultas: acc = sum(xq*Wq) + Bq (int32); y = min(127, (max(acc,0)*M + 2^30) >> 31).",
           " * Saída: acumulador int32; valor real = acc * S_OUT (regressor: em Hz, ja desnormalizado",
           " * por Y_MEDIA/Y_ESCALA abaixo). Pesos em ordem [entrada][saída]. */",
           "#ifndef MLP_PESOS_Q8_H", "#define MLP_PESOS_Q8_H", "#include <stdint.h>", "",
           f"#define MLP_Q_SHIFT {SHIFT}", f"#define MLP_Q_CLIP {CLIP_IN:.1f}f"]
    m0 = mods["FREQ"]
    out += [f"#define MLP_N_IN {len(m0['mu'])}", f"#define MLP_N_H1 {m0['q']['Wq'][0].shape[1]}",
            f"#define MLP_N_H2 {m0['q']['Wq'][1].shape[1]}",
            f"#define CFG_N_OUT {mods['CFG']['q']['Wq'][2].shape[1]}", "",
            "/* ---- pré-processamento comum aos dois modelos ---- */",
            arr_c("float", "Q_MEDIANA", m0["med"], ff), arr_c("float", "Q_MEDIA", m0["mu"], ff),
            arr_c("float", "Q_INV_ESCALA", 1.0 / m0["sd"], ff),
            f"#define Q_INV_S_IN {ff(1.0 / m0['q']['s_in'])}"]
    for tag, m in mods.items():
        q = m["q"]
        out.append(f"\n/* ---- modelo {tag} ---- */")
        out.append(f"#define {tag}_Q_S_OUT {ff(q['s_out'])}")
        if m["y_tr"] is not None:
            out.append(f"#define {tag}_Y_MEDIA {ff(m['y_tr'][0])}")
            out.append(f"#define {tag}_Y_ESCALA {ff(m['y_tr'][1])}")
        for k, (w, bb) in enumerate(zip(q["Wq"], q["bq"])):
            out.append(arr_c("int8_t", f"{tag}_QW{k}", w, fi, 20))
            out.append(arr_c("int32_t", f"{tag}_QB{k}", bb, fi))
        for k, M in enumerate(q["M"]):
            out.append(f"#define {tag}_QM{k} {M}  /* {M / (1 << SHIFT):.6g} */")
    path.write_text("\n".join(out) + "\n#endif\n", encoding="utf-8")


# --------------------------------------------------------------------------- #
def main():
    SAIDA.mkdir(exist_ok=True)
    df = pd.read_csv(RAIZ / "resultados" / "janelas_descritores.csv")
    nomes = [c for c in df.columns if c[:2] in ("A_", "B_")]
    X = df[nomes].to_numpy(np.float64)
    print(f"{len(df)} janelas x {len(nomes)} descritores; NaN em {np.isnan(X).any(1).sum()} janelas")

    reg, cls = treinar(X, df)
    for a, b_ in zip(extrair(reg)[:3], extrair(cls)[:3]):
        assert np.array_equal(a, b_), "pré-processamento diferente entre os modelos"
    mods = {}
    for tag, pipe in (("FREQ", reg), ("CFG", cls)):
        med, mu, sd, W, b, y_tr, classes = extrair(pipe)
        Wf, bf = dobrar_escala(med, mu, sd, W, b, y_tr)
        Xstd = (np.where(np.isnan(X), med, X) - mu) / sd
        # para quantizar, a desnormalização do alvo do regressor fica fora (escala de saída)
        q = quantizar(Xstd, W, b)
        mods[tag] = dict(med=med, mu=mu, sd=sd, Wf=Wf, bf=bf, y_tr=y_tr, classes=classes, q=q)

    # ---------------- referências ----------------
    sk_freq = reg.predict(X)
    sk_cfg = cls.predict(X)
    m = mods["FREQ"]
    f32_freq = float_ref(X, m["med"], m["Wf"], m["bf"])[:, 0]
    xq_f = quant_entrada(X, m["med"], m["mu"], m["sd"], m["q"]["s_in"])
    acc_f = int8_ref(xq_f, m["q"])[:, 0]
    q8_freq = acc_f * m["q"]["s_out"] * m["y_tr"][1] + m["y_tr"][0]

    m = mods["CFG"]
    cl = np.array(m["classes"])
    f32_cfg_log = float_ref(X, m["med"], m["Wf"], m["bf"])
    xq_c = quant_entrada(X, m["med"], m["mu"], m["sd"], m["q"]["s_in"])
    assert np.array_equal(xq_c, xq_f)
    acc_c = int8_ref(xq_c, m["q"])
    f32_cfg = f32_cfg_log.argmax(1)
    q8_cfg = acc_c.argmax(1)  # empate: menor índice, igual ao C

    assert np.abs(acc_f).max() < 2**31 and np.abs(acc_c).max() < 2**31

    # ---------------- cabeçalhos ----------------
    header_f32(SAIDA / "mlp_pesos_f32.h", nomes, mods)
    header_q8(SAIDA / "mlp_pesos_q8.h", mods)
    with open(SAIDA / "mlp_classes.h", "w", encoding="utf-8") as fh:
        fh.write("/* Gerado por exportar_modelo.py. Classes do modelo CFG (config. recalque-sucção). */\n")
        fh.write("#ifndef MLP_CLASSES_H\n#define MLP_CLASSES_H\n")
        fh.write(f"#define CFG_N_CLASSES {len(cl)}\n")
        fh.write("static const char *const CFG_CLASSES[] = {" +
                 ", ".join(f'"{c}"' for c in cl) + "};\n#endif\n")

    # vetores de teste completos (binário little-endian):
    # cabeçalho: n, n_in, n_cls ; por janela: x[36] f32, freq_f32 f32, cfg_f32 i32,
    #            acc_freq i32, cfg_q8 i32, acc_cfg[n_cls] i32
    n, nin, ncl = X.shape[0], X.shape[1], len(cl)
    with open(SAIDA / "vetores_teste.bin", "wb") as fh:
        fh.write(struct.pack("<3i", n, nin, ncl))
        for i in range(n):
            fh.write(struct.pack(f"<{nin}f", *X[i].astype(np.float32)))
            fh.write(struct.pack("<fiii", f32_freq[i], int(f32_cfg[i]), int(acc_f[i]), int(q8_cfg[i])))
            fh.write(struct.pack(f"<{ncl}i", *acc_c[i].astype(np.int64)))

    # subconjunto para o testbench HLS: entradas já quantizadas + saídas inteiras esperadas
    rng = np.random.default_rng(SEMENTE)
    sel = np.sort(rng.choice(n, size=min(N_TB, n), replace=False))
    with open(SAIDA / "vetores_tb.h", "w", encoding="utf-8") as fh:
        fh.write("/* Gerado por exportar_modelo.py. Vetores do testbench HLS (entradas int8 já\n"
                 " * quantizadas por modelo; saídas = acumuladores int32 esperados). */\n")
        fh.write("#ifndef VETORES_TB_H\n#define VETORES_TB_H\n#include <stdint.h>\n")
        fh.write(f"#define TB_N {len(sel)}\n")
        fh.write(arr_c("int8_t", "TB_XQ", xq_f[sel], fi, 36))
        fh.write(arr_c("int32_t", "TB_ACC_FREQ", acc_f[sel], fi))
        fh.write(arr_c("int32_t", "TB_ACC_CFG", acc_c[sel], fi, 7))
        fh.write("#endif\n")

    # vetores brutos para o teste na MicroBlaze (NaN = janela sem hits normais)
    sel_mb = sel[:16]
    fn = lambda v: "NAN" if np.isnan(v) else ff(v)  # noqa: E731
    with open(SAIDA / "vetores_mb.h", "w", encoding="utf-8") as fh:
        fh.write("/* Gerado por exportar_modelo.py. 16 janelas brutas + saídas esperadas. */\n")
        fh.write("#ifndef VETORES_MB_H\n#define VETORES_MB_H\n#include <math.h>\n#include <stdint.h>\n")
        fh.write(f"#define MB_N {len(sel_mb)}\n")
        fh.write(arr_c("float", "MB_X", X[sel_mb], fn, 6))
        fh.write(arr_c("int32_t", "MB_ACC_FREQ", acc_f[sel_mb], fi))
        fh.write(arr_c("int32_t", "MB_CLASSE_Q8", q8_cfg[sel_mb], fi))
        fh.write(arr_c("int32_t", "MB_CLASSE_F32", f32_cfg[sel_mb], fi))
        fh.write("#endif\n")

    # ---------------- resumo ----------------
    y_freq = df["freq"].to_numpy(float)
    y_cfg = df["config"].astype(str).to_numpy()
    lin = [
        f"Janelas: {n}; descritores: {nin}; classes CFG: {[str(c) for c in cl]}",
        f"Parâmetros por rede: FREQ {sum(w.size + bb.size for w, bb in zip(mods['FREQ']['Wf'], mods['FREQ']['bf']))}, "
        f"CFG {sum(w.size + bb.size for w, bb in zip(mods['CFG']['Wf'], mods['CFG']['bf']))}",
        "",
        "FREQ (regressor, Hz)",
        f"  |sklearn - float32 C-ref| máx = {np.abs(sk_freq - f32_freq).max():.2e} Hz",
        f"  |sklearn - int8|          máx = {np.abs(sk_freq - q8_freq).max():.3f} Hz, "
        f"média = {np.abs(sk_freq - q8_freq).mean():.3f} Hz",
        f"  nível arredondado (10 Hz) igual sklearn x int8: "
        f"{np.mean(np.round(sk_freq / 10) == np.round(q8_freq / 10)) * 100:.2f} %",
        f"  MAE treino (só referência, não é desempenho): sklearn {np.abs(sk_freq - y_freq).mean():.2f} Hz, "
        f"int8 {np.abs(q8_freq - y_freq).mean():.2f} Hz",
        "",
        "CFG (classificador, 7 classes)",
        f"  classe igual sklearn x float32: {np.mean(cl[f32_cfg] == sk_cfg) * 100:.2f} %",
        f"  classe igual sklearn x int8:    {np.mean(cl[q8_cfg] == sk_cfg) * 100:.2f} %",
        f"  acurácia treino (só referência): sklearn {np.mean(sk_cfg == y_cfg) * 100:.1f} %, "
        f"int8 {np.mean(cl[q8_cfg] == y_cfg) * 100:.1f} %",
        "",
        "Quantização",
    ]
    for tag in ("FREQ", "CFG"):
        q = mods[tag]["q"]
        lin.append(f"  {tag}: S_IN={q['s_in']:.5f}  s_w={['%.4g' % s for s in q['sw']]}  "
                   f"s_act={['%.4g' % s for s in q['s_act']]}  M={q['M']}")
    txt = "\n".join(lin)
    (SAIDA / "resumo_export.txt").write_text(txt + "\n", encoding="utf-8")
    print(txt)


if __name__ == "__main__":
    main()
