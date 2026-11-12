"""
Exporta os hits gravados e os descritores esperados para validar o cálculo em C
(src/ea_descritores.c) contra resultados/janelas_descritores.csv.

Uso (a partir de MLP/):
    .venv\\Scripts\\python.exe embarcado\\exportar_hits.py

Gera:
  gerado/descritores_cfg.h  limiares harmonizados, duração da janela e ordem dos descritores
  gerado/hits_teste.bin     todas as corridas: calibração, hits (campos brutos do registro e
                            unidades físicas do hits.csv) e os 36 descritores esperados por janela

Também confere as fórmulas de conversão do registro bruto para unidades físicas
(as mesmas que o firmware terá de aplicar), comparando com as colunas do hits.csv.
"""
from __future__ import annotations

import json
import struct
import sys
from pathlib import Path

import numpy as np
import pandas as pd

AQUI = Path(__file__).resolve().parent
RAIZ = AQUI.parent
sys.path.insert(0, str(RAIZ))
import mlp_ea  # noqa: E402

SAIDA = AQUI / "gerado"
DUR_JANELA = 5.0
FS = 10e6          # amostragem do ADC (amostras/s)
EU = 1e-14         # unidade de energia (V^2 s)

ORDEM_CANAL = ["taxa_hits", "frac_sat", "frac_maxdur", "log_energia_s", "fm_kHz_med", "amp_med",
               "amp_p10", "amp_p50", "amp_p90", "log_energia_p50", "log_energia_p90",
               "log_dur_p50", "log_dur_p90", "log_cont_p50", "log_cont_p90", "log_subida_p50",
               "fm_kHz_normais", "log_RA_p50"]

# struct do hit no binário (little-endian)
FMT_HIT = "<dBBBBIIIIQfffff"


def conv_fisico(h: pd.DataFrame, vpc: np.ndarray, g_db: np.ndarray) -> dict:
    """Mesmas fórmulas de ea_hit_de_registro (src/ea_descritores.c)."""
    g = 10.0 ** (g_db / 20.0)
    peak_v = h["peak"].to_numpy(float) * vpc
    return dict(peak_v=peak_v,
                amp_dbae=20.0 * np.log10(peak_v / g / 1e-6),
                energy_eu=h["energy"].to_numpy(float) * vpc ** 2 / FS / g ** 2 / EU,
                rise_us=h["rise"].to_numpy(float) / FS * 1e6,
                duration_us=h["duration"].to_numpy(float) / FS * 1e6)


def main():
    SAIDA.mkdir(exist_ok=True)
    corridas = mlp_ea.ler_corridas(RAIZ / "Dados")
    thr = (max(c["thr_a"] for c in corridas), max(c["thr_b"] for c in corridas))
    df = pd.read_csv(RAIZ / "resultados" / "janelas_descritores.csv")
    nomes = [f"{p}_{n}" for p in ("A", "B") for n in ORDEM_CANAL]
    assert nomes == [c for c in df.columns if c[:2] in ("A_", "B_")], "ordem dos descritores mudou"
    # a ordem das janelas no CSV deve ser a das corridas (cronológica) e k crescente
    runs_csv = list(dict.fromkeys(df["run"]))
    assert runs_csv == [c["run"] for c in corridas], "ordem das corridas difere do CSV"

    err = {k: 0.0 for k in ("peak_v", "amp_dbae", "energy_eu", "rise_us", "duration_us")}
    n_jan_tot = 0
    mb = []   # 4 janelas (de corridas distintas) para o benchmark na MicroBlaze
    sel_mb = {0: 3, len(corridas) // 3: 5, 2 * len(corridas) // 3: 7, len(corridas) - 1: 2}
    with open(SAIDA / "hits_teste.bin", "wb") as fh:
        fh.write(b"EAH1")
        fh.write(struct.pack("<iifff", len(corridas), len(df), thr[0], thr[1], DUR_JANELA))
        for c in corridas:
            cfg = json.loads((RAIZ / "Dados" / Path(c["arquivo"]).parent / "config.json")
                             .read_text(encoding="utf-8"))
            vpc_ab = [cfg["calibracao"][n]["volts_per_code"] for n in ("A", "B")]
            g_ab = [cfg["extrator"]["gain_db_a"], cfg["extrator"]["gain_db_b"]]
            h = c["hits"].sort_values("rx_time_s", kind="stable")
            ch = h["channel"].to_numpy(int)
            fis = conv_fisico(h, np.where(ch == 0, vpc_ab[0], vpc_ab[1]),
                              np.where(ch == 0, g_ab[0], g_ab[1]))
            for k, v in fis.items():
                ref = h[k].to_numpy(float)
                ok = np.abs(ref) > 1e-9
                if ok.any():
                    err[k] = max(err[k], float(np.max(np.abs(v[ok] - ref[ok]) / np.abs(ref[ok]))))
            n_jan = int(np.floor(c["hits"]["rx_time_s"].max() / DUR_JANELA))
            dj = df[df["run"] == c["run"]]
            assert len(dj) == n_jan and (dj["janela"].to_numpy() == np.arange(n_jan)).all()
            n_jan_tot += n_jan
            fh.write(struct.pack("<ii4f", len(h), n_jan, *vpc_ab, *g_ab))
            cols = h[["rx_time_s", "channel", "sat", "maxdur", "rovl", "counts", "peak", "rise",
                      "duration", "energy", "peak_v", "amp_dbae", "energy_eu", "rise_us",
                      "duration_us"]].itertuples(index=False)
            for r in cols:
                fh.write(struct.pack(FMT_HIT, r[0], r[1], r[2], r[3], r[4], int(r[5]), int(r[6]),
                                     int(r[7]), int(r[8]), int(r[9]), *map(float, r[10:])))
            fh.write(dj[nomes].to_numpy(np.float32).tobytes())
            ic = corridas.index(c)
            if ic in sel_mb:
                k = sel_mb[ic]
                hk = h[np.floor(h["rx_time_s"].to_numpy() / DUR_JANELA) == k]
                mb.append(dict(run=c["run"], k=k, vpc=vpc_ab, g=g_ab, hits=hk,
                               x=dj[nomes].to_numpy(np.float32)[k]))
    assert n_jan_tot == len(df)

    # cabeçalho para a MicroBlaze: hits brutos de 4 janelas + descritores esperados
    def fl(v):
        if np.isnan(v):
            return "NAN"
        s = f"{float(np.float32(v)):.9g}"
        return (s if any(c in s for c in ".en") else s + ".0") + "f"
    with open(SAIDA / "hits_mb.h", "w", encoding="utf-8") as fh:
        fh.write("/* Gerado por exportar_hits.py. Hits brutos de 4 janelas e descritores esperados\n"
                 " * (resultados/janelas_descritores.csv) para o benchmark na MicroBlaze. */\n")
        fh.write("#ifndef HITS_MB_H\n#define HITS_MB_H\n#include <math.h>\n#include <stdint.h>\n")
        fh.write("typedef struct { uint8_t canal, sat, maxdur; uint32_t pico, contagens, subida, "
                 "duracao; uint64_t energia; } hit_mb_t;\n")
        fh.write(f"#define HMB_N_JAN {len(mb)}\n")
        fh.write("static const uint32_t HMB_INICIO[] = {" +
                 ", ".join(str(sum(len(m['hits']) for m in mb[:i])) for i in range(len(mb) + 1)) + "};\n")
        fh.write("static const float HMB_CAL[][4] = {  /* vpc A, vpc B, ganho A, ganho B */\n" +
                 ",\n".join("    {" + ", ".join(fl(v) for v in (*m["vpc"], *m["g"])) + "}"
                            for m in mb) + "\n};\n")
        fh.write("static const hit_mb_t HMB_HITS[] = {\n")
        linhas = []
        for m in mb:
            linhas.append(f"    /* {m['run']}, janela {m['k']} */")
            for r in m["hits"][["channel", "sat", "maxdur", "peak", "counts", "rise", "duration",
                                "energy"]].itertuples(index=False):
                linhas.append("    {" + ", ".join(str(int(v)) for v in r[:7]) + f", {int(r[7])}ULL}},")
        fh.write("\n".join(linhas) + "\n};\n")
        fh.write("static const float HMB_X_ESPERADO[][36] = {\n" +
                 ",\n".join("    {" + ", ".join(fl(v) for v in m["x"]) + "}" for m in mb) + "\n};\n")
        fh.write("#endif\n")

    with open(SAIDA / "descritores_cfg.h", "w", encoding="utf-8") as fh:
        fh.write("/* Gerado por exportar_hits.py. Parâmetros do cálculo de descritores\n"
                 " * usados no treino (mlp_ea.py, harmonização de limiar ligada). */\n")
        fh.write("#ifndef DESCRITORES_CFG_H\n#define DESCRITORES_CFG_H\n")
        fh.write(f"#define EA_JANELA_S {DUR_JANELA:.1f}f\n")
        fh.write(f"#define EA_LIMIAR_A_DBAE {thr[0]:.4f}f  /* hits abaixo são descartados */\n")
        fh.write(f"#define EA_LIMIAR_B_DBAE {thr[1]:.4f}f\n")
        fh.write("/* Ordem dos 18 descritores por canal (canal A: 0-17, canal B: 18-35):\n")
        for i, n in enumerate(ORDEM_CANAL):
            fh.write(f" *  {i:2d} {n}\n")
        fh.write(" */\n#endif\n")

    print(f"{len(corridas)} corridas, {len(df)} janelas; limiares harmonizados A={thr[0]:.2f}, "
          f"B={thr[1]:.2f} dBAE")
    print("Conversão registro -> unidades físicas, erro relativo máx. contra o hits.csv:")
    for k, v in err.items():
        print(f"  {k:12s} {v:.2e}")


if __name__ == "__main__":
    main()
