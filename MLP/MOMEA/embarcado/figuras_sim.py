"""
Figuras do artigo de simulação da MLP embarcada.

Uso (a partir de MLP/):
    .venv\\Scripts\\python.exe embarcado\\figuras_sim.py

Lê os artefatos de verificação gerados em embarcado/ e grava PNGs em
ARTIGO_SIM_EMBARCADO-10.11.2026/figuras/, mais um resumo numérico (resumo_figuras.json)
usado para conferir os números citados no texto.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

AQUI = Path(__file__).resolve().parent
RAIZ = AQUI.parent
GER = AQUI / "gerado"
SAIDA = RAIZ / "ARTIGO_SIM_EMBARCADO-10.11.2026" / "figuras"

# paleta categórica (ordem fixa, validada) e tintas
AZUL, LARANJA, AQUA, AMARELO = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
TINTA, TINTA2, GRADE = "#0b0b0b", "#52514e", "#e4e3df"

plt.rcParams.update({
    "figure.dpi": 110, "savefig.dpi": 200, "savefig.bbox": "tight",
    "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9,
    "axes.edgecolor": TINTA2, "axes.labelcolor": TINTA, "xtick.color": TINTA2,
    "ytick.color": TINTA2, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRADE, "grid.linewidth": 0.6,
    "axes.axisbelow": True, "legend.frameon": False,
})

NOMES = ["taxa_hits", "frac_sat", "frac_maxdur", "log_energia_s", "fm_kHz_med", "amp_med",
         "amp_p10", "amp_p50", "amp_p90", "log_energia_p50", "log_energia_p90", "log_dur_p50",
         "log_dur_p90", "log_cont_p50", "log_cont_p90", "log_subida_p50", "fm_kHz_normais",
         "log_RA_p50"]


def const_q8(nome: str) -> float:
    txt = (GER / "mlp_pesos_q8.h").read_text(encoding="utf-8")
    return float(re.search(rf"#define {nome} ([-0-9.e]+)f", txt).group(1))


def ler_vetores():
    raw = (GER / "vetores_teste.bin").read_bytes()
    n, nin, ncl = np.frombuffer(raw[:12], "<i4")
    reg = np.dtype([("x", "<f4", nin), ("freq_f32", "<f4"), ("cls_f32", "<i4"),
                    ("acc_f", "<i4"), ("cls_q8", "<i4"), ("acc_c", "<i4", ncl)])
    return np.frombuffer(raw[12:], reg, count=n)


def ler_c_descr(arq: Path, n: int):
    reg = np.dtype([("x", "<f4", 36), ("cls", "<i4"), ("hz", "<f4")])
    return np.frombuffer(arq.read_bytes(), reg, count=n)


def salvar(fig, nome):
    SAIDA.mkdir(parents=True, exist_ok=True)
    fig.savefig(SAIDA / nome)
    plt.close(fig)


# --------------------------------------------------------------------------- #
def fig_quantizacao(v, df, resumo):
    hz_q8 = v["acc_f"] * const_q8("FREQ_Q_S_OUT") * const_q8("FREQ_Y_ESCALA") + const_q8("FREQ_Y_MEDIA")
    hz_f = v["freq_f32"].astype(float)
    d = hz_q8 - hz_f
    y = df["freq"].to_numpy(float)
    fig, ax = plt.subplots(1, 2, figsize=(7.2, 3.0))
    ax[0].scatter(hz_f, hz_q8, s=6, color=AZUL, alpha=0.35, linewidths=0)
    lim = [min(hz_f.min(), hz_q8.min()) - 3, max(hz_f.max(), hz_q8.max()) + 3]
    ax[0].plot(lim, lim, color=TINTA2, lw=1, ls="--")
    ax[0].set(xlim=lim, ylim=lim, xlabel="Frequência estimada, float32 (Hz)",
              ylabel="Frequência estimada, int8 (Hz)", title="(a) int8 × float32 por janela")
    bins = np.arange(-12.5, 12.6, 0.5)
    ax[1].hist(d, bins=bins, color=AZUL, edgecolor="white", linewidth=0.6)
    ax[1].set(xlabel="Diferença int8 − float32 (Hz)", ylabel="Janelas",
              title="(b) Distribuição da diferença")
    ax[1].axvline(0, color=TINTA2, lw=0.8)
    ax[1].text(0.97, 0.95, f"média {d.mean():+.2f} Hz\n|Δ| médio {np.abs(d).mean():.2f} Hz\n"
               f"|Δ| máx {np.abs(d).max():.1f} Hz", transform=ax[1].transAxes, ha="right",
               va="top", color=TINTA, fontsize=8)
    fig.tight_layout()
    salvar(fig, "fig_quantizacao.png")
    resumo["quant"] = dict(
        dif_media=float(d.mean()), dif_abs_media=float(np.abs(d).mean()),
        dif_abs_max=float(np.abs(d).max()), dif_p95=float(np.percentile(np.abs(d), 95)),
        nivel_igual=float(np.mean(np.round(hz_q8 / 10) == np.round(hz_f / 10))),
        classe_igual=float(np.mean(v["cls_q8"] == v["cls_f32"])),
        mae_f32_treino=float(np.abs(hz_f - y).mean()), mae_q8_treino=float(np.abs(hz_q8 - y).mean()))


def fig_descritores(df, resumo):
    ref = df[[f"{c}_{n}" for c in ("A", "B") for n in NOMES]].to_numpy(np.float32)
    out = {}
    for modo in ("exato", "hist"):
        c = ler_c_descr(GER / f"c_descr_{modo}.bin", len(df))
        x = c["x"].astype(np.float64)
        r = ref.astype(np.float64)
        ok = ~np.isnan(r) & ~np.isnan(x)
        e = np.where(ok, np.abs(x - r) / (1.0 + np.abs(r)), np.nan)
        e2 = np.concatenate([e[:, :18], e[:, 18:]], axis=0)   # junta canais A e B
        out[modo] = dict(p99=np.nanpercentile(e2, 99, axis=0), mx=np.nanmax(e2, axis=0),
                         nan_dif=int(np.sum(np.isnan(r) != np.isnan(x))), cls=c["cls"], hz=c["hz"])
    fig, ax = plt.subplots(figsize=(7.0, 4.6))
    ypos = np.arange(len(NOMES))
    piso = 1e-9
    for k, (modo, cor, rot) in enumerate((("exato", AZUL, "Percentis exatos"),
                                          ("hist", LARANJA, "Percentis por histograma"))):
        mx = np.maximum(out[modo]["mx"], piso)
        ax.scatter(mx, ypos + (k - 0.5) * 0.25, s=28, color=cor, label=rot, zorder=3,
                   edgecolors="white", linewidths=0.8)
    ax.axvline(1e-3, color=TINTA2, lw=1, ls="--")
    ax.text(1.25e-3, 0.0, "tolerância 10⁻³", color=TINTA2, fontsize=8, va="center")
    ax.set_xscale("log")
    ax.set_yticks(ypos, NOMES)
    ax.set_ylim(len(NOMES) + 0.2, -0.8)
    ax.set_xlim(piso / 2, 1)
    ax.set_xlabel("Erro normalizado máximo |C − Python| / (1 + |Python|), canais A e B")
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2)
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    salvar(fig, "fig_erro_descritores.png")
    return out


def concordancia(v, df, out, resumo):
    hz_py = v["acc_f"] * const_q8("FREQ_Q_S_OUT") * const_q8("FREQ_Y_ESCALA") + const_q8("FREQ_Y_MEDIA")
    for modo in ("exato", "hist"):
        o = out[modo]
        resumo[f"cadeia_{modo}"] = dict(
            classe_igual=float(np.mean(o["cls"] == v["cls_q8"])),
            nivel_igual=float(np.mean(np.round(o["hz"] / 10) == np.round(hz_py / 10))),
            hz_dif_max=float(np.max(np.abs(o["hz"] - hz_py))),
            erro_norm_max=float(np.nanmax(o["mx"])), nan_divergentes=o["nan_dif"],
            acuracia_cfg_treino=float(np.mean(
                np.array(sorted(df["config"].astype(str).unique()))[o["cls"]] == df["config"].astype(str))))


# --------------------------------------------------------------------------- #
def figs_ciclos(resumo):
    arqs = sorted((AQUI / "microblaze" / "sim" / "resultados").glob("ciclos_mb*.csv"))
    if not arqs:
        print("ciclos_mb*.csv ausente: figuras de ciclos não geradas")
        return
    c = (pd.concat([pd.read_csv(a) for a in arqs])
         .drop_duplicates(subset=["config", "programa", "etapa", "indice"], keep="last"))
    cfgs = [("soft", "Sem FPU"), ("fpu", "Com FPU")]
    cad = c[c.programa.str.startswith("cadeia")]
    res = {}
    for cfg, _ in cfgs:
        for prog in ("cadeia_exato", "cadeia_hist"):
            g = cad[(cad.config == cfg) & (cad.programa == prog)]
            if g.empty:
                continue
            hits = g[g.etapa == "hits_total"]
            res[(cfg, prog)] = dict(
                ciclos_hit=float(hits.ciclos.sum() / hits.hits.sum()),
                fechar=float(g[g.etapa == "fechar_janela"].ciclos.mean()),
                mlp=float(g[g.etapa == "mlp_int8_com_quantizacao"].ciclos.mean()),
                hits_jan=float(hits.hits.mean()), confere=bool(g.confere_pc.all()))
    m = c[c.programa == "mlp"]
    mlp = {}
    for cfg, _ in cfgs:
        g = m[m.config == cfg]
        if g.empty:
            continue
        mlp[cfg] = dict(float32=float(g[g.etapa == "mlp_float32_2redes"].ciclos.mean()),
                        quant=float(g[g.etapa == "quantizacao_entrada"].ciclos.mean()),
                        int8=float(g[g.etapa == "nucleo_int8_2redes"].ciclos.mean()),
                        confere=bool(g.confere_pc.all()))
    resumo["mb_cadeia"] = {f"{a}/{b}": v for (a, b), v in res.items()}
    resumo["mb_mlp"] = mlp

    # (1) composição dos ciclos por janela (hits médios da janela de teste)
    if res:
        fig, ax = plt.subplots(figsize=(7.0, 2.9))
        rot, hits_c, fech_c, mlp_c = [], [], [], []
        for cfg, nome_cfg in cfgs:
            for prog, nome_p in (("cadeia_exato", "exato"), ("cadeia_hist", "histograma")):
                if (cfg, prog) not in res:
                    continue
                r = res[(cfg, prog)]
                rot.append(f"{nome_cfg}\n{nome_p}")
                hits_c.append(r["ciclos_hit"] * r["hits_jan"] / 1e3)
                fech_c.append(r["fechar"] / 1e3)
                mlp_c.append(r["mlp"] / 1e3)
        y = np.arange(len(rot))
        esq = np.zeros(len(rot))
        for vals, cor, nome in ((hits_c, AZUL, "Processar os hits da janela"),
                                (fech_c, LARANJA, "Fechar a janela (percentis)"),
                                (mlp_c, AQUA, "MLP int8 (2 redes)")):
            ax.barh(y, vals, left=esq, color=cor, height=0.6, label=nome, edgecolor="white",
                    linewidth=2)
            esq += np.array(vals)
        for yi, tot in zip(y, esq):
            txt = (f"{tot / 1000:.2f} milhões".replace(".", ",") if tot >= 1000
                   else f"{tot:.0f} mil")
            ax.text(tot * 1.01, yi, txt, va="center", color=TINTA, fontsize=8)
        ax.set_yticks(y, rot)
        ax.invert_yaxis()
        ax.set_xlabel(f"Milhares de ciclos por janela de 5 s (~{np.mean([r['hits_jan'] for r in res.values()]):.0f} hits)")
        ax.set_xlim(0, esq.max() * 1.18)
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.28), ncol=3)
        ax.grid(axis="y", visible=False)
        fig.tight_layout()
        salvar(fig, "fig_ciclos_janela.png")

        # (2) carga da CPU em função da taxa de hits (modo exato e histograma)
        fig, ax = plt.subplots(figsize=(6.4, 3.4))
        taxa = np.logspace(0, np.log10(5000), 200)  # hits/s somando os 2 canais
        estilos = {("soft", "cadeia_exato"): (AZUL, "-"), ("soft", "cadeia_hist"): (AZUL, ":"),
                   ("fpu", "cadeia_exato"): (LARANJA, "-"), ("fpu", "cadeia_hist"): (LARANJA, ":")}
        carga = {}
        for (cfg, prog), r in res.items():
            cor, ls = estilos[(cfg, prog)]
            u = (taxa * r["ciclos_hit"] + (r["fechar"] + r["mlp"]) / 5.0) / 100e6 * 100
            nome = f"{dict(cfgs)[cfg]}, {'exato' if prog == 'cadeia_exato' else 'histograma'}"
            ax.plot(taxa, u, color=cor, ls=ls, lw=2, label=nome)
            carga[f"{cfg}/{prog}"] = {str(t): float((t * r["ciclos_hit"] + (r["fechar"] + r["mlp"]) / 5.0) / 1e6)
                                      for t in (16, 666, 2000)}
        for t, txt in ((16, "taxa dos ensaios\n(~16 hits/s)"), (666, "333 hits/s\npor canal")):
            ax.axvline(t, color=TINTA2, lw=0.8, ls="--")
            ax.text(t * 1.08, 0.97, txt, transform=ax.get_xaxis_transform(), fontsize=7.5,
                    color=TINTA2, va="top")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("Taxa de hits (2 canais somados, hits/s)")
        ax.set_ylabel("Ocupação da CPU a 100 MHz (%)")
        ax.legend(loc="lower right", fontsize=8)
        fig.tight_layout()
        salvar(fig, "fig_carga_cpu.png")
        resumo["carga_pct"] = carga

    # (3) latência da inferência (2 redes) por plataforma
    if mlp:
        itens = []
        for cfg, nome in cfgs:
            if cfg in mlp:
                rot = nome[0].lower() + nome[1:]          # "sem FPU" / "com FPU"
                itens.append((f"MicroBlaze {rot}, float32", mlp[cfg]["float32"] / 100.0))
                itens.append((f"MicroBlaze {rot}, int8", (mlp[cfg]["int8"] + mlp[cfg]["quant"]) / 100.0))
        itens.append(("IP HLS int8 (100 MHz)", 11391 / 100.0))
        fig, ax = plt.subplots(figsize=(6.4, 2.6))
        y = np.arange(len(itens))
        cores = [AZUL if "float" in n else (AQUA if "HLS" in n else LARANJA) for n, _ in itens]
        ax.barh(y, [v for _, v in itens], color=cores, height=0.6)
        for yi, (_, v) in zip(y, itens):
            txt = (f"{v / 1000:.2f} ms".replace(".", ",") if v >= 1000 else f"{v:.0f} µs")
            ax.text(v * 1.08, yi, txt, va="center", fontsize=8, color=TINTA)
        ax.set_yticks(y, [n for n, _ in itens])
        ax.invert_yaxis()
        ax.set_xscale("log")
        ax.set_xlim(right=max(v for _, v in itens) * 6)
        ax.set_xlabel("Latência de uma inferência (FREQ + CFG), µs, escala log")
        ax.grid(axis="y", visible=False)
        fig.tight_layout()
        salvar(fig, "fig_latencia_mlp.png")


# --------------------------------------------------------------------------- #
def fi(n) -> str:
    """inteiro com milhar em \\, (LaTeX, pt-BR)"""
    s = f"{int(round(n)):,}".replace(",", "\\,")
    return s


def fd(x, d=1) -> str:
    return f"{x:.{d}f}".replace(".", "{,}")


def tamanhos_elf() -> dict:
    import subprocess
    mbsize = Path(r"E:\Xilinx\Vitis\2022.2\gnu\microblaze\nt\bin\mb-size.exe")
    out = {}
    for e in (AQUI / "microblaze" / "sim" / "elf").glob("cadeia_*.elf"):
        r = subprocess.run([str(mbsize), str(e)], capture_output=True, text=True)
        t, d = r.stdout.splitlines()[-1].split()[:2]
        out[e.stem] = (int(t) + int(d)) / 1024
    return out


def escrever_numeros(resumo):
    arq = SAIDA.parent / "numeros_mb.tex"
    nomes = ["mbMlpIntSoftUs", "mbMlpIntFpuUs", "mbHitSoft", "mbHitFpu", "mbCargaEnsaioMax",
             "mbMlpFloatSoft", "mbMlpFloatFpu", "mbQuantSoft", "mbQuantFpu", "mbMlpIntSoft",
             "mbMlpIntFpu", "mbHitHistSoft", "mbHitHistFpu", "mbFecharSoft", "mbFecharFpu",
             "mbFecharHistSoft", "mbFecharHistFpu", "mbParagrafoMlp", "mbParagrafoCarga",
             "mbTextoMin", "mbTextoMax", "mbTextoExatoSoft", "mbTextoExatoFpu", "mbTextoHistSoft",
             "mbTextoHistFpu", "mbParagrafoRota", "mbFatorHls"]
    val = {n: "\\completar{simulação pendente}" for n in nomes}
    tam = tamanhos_elf()
    if tam:
        val.update(mbTextoMin=fd(min(tam.values())), mbTextoMax=fd(max(tam.values())),
                   mbTextoExatoSoft=fd(tam.get("cadeia_exato_soft", np.nan)),
                   mbTextoExatoFpu=fd(tam.get("cadeia_exato_fpu", np.nan)),
                   mbTextoHistSoft=fd(tam.get("cadeia_hist_soft", np.nan)),
                   mbTextoHistFpu=fd(tam.get("cadeia_hist_fpu", np.nan)))
    m = resumo.get("mb_mlp", {})
    c = resumo.get("mb_cadeia", {})
    if "soft" in m and "fpu" in m:
        s, f = m["soft"], m["fpu"]
        int_s, int_f = s["int8"] + s["quant"], f["int8"] + f["quant"]
        val.update(mbMlpFloatSoft=fi(s["float32"]), mbMlpFloatFpu=fi(f["float32"]),
                   mbQuantSoft=fi(s["quant"]), mbQuantFpu=fi(f["quant"]),
                   mbMlpIntSoft=fi(s["int8"]), mbMlpIntFpu=fi(f["int8"]),
                   mbMlpIntSoftUs=fi(int_s / 100), mbMlpIntFpuUs=fi(int_f / 100),
                   mbFatorHls=fd(int_s / 11391, 0))
        val["mbParagrafoMlp"] = (
            f"Sem FPU, a inferência em float32 custou {fi(s['float32'])} ciclos "
            f"({fd(s['float32'] / 1e5, 1)}\\,ms), {fd(s['float32'] / int_s, 0)} vezes a inferência int8, "
            f"porque cada operação de ponto flutuante é emulada em software. Com FPU, o float32 caiu para "
            f"{fi(f['float32'])} ciclos, {fd(f['float32'] / int_f, 1)} vezes o int8. O núcleo int8 não usa "
            + (f"ponto flutuante e custou exatamente o mesmo nas duas configurações ({fi(s['int8'])} ciclos"
               if s["int8"] == f["int8"] else
               f"ponto flutuante e custou praticamente o mesmo nas duas configurações ({fi(s['int8'])} e "
               f"{fi(f['int8'])} ciclos")
            + f", ou {fd(s['int8'] / 8960, 1)} ciclos por multiplicação-acumulação dos 8\\,960 pesos); "
            f"a diferença entre elas está na quantização da entrada, com algumas dezenas de operações "
            f"de ponto flutuante, que custou {fi(s['quant'])} ciclos sem FPU e {fi(f['quant'])} com FPU.")
        val["mbParagrafoRota"] = (
            f"Na MicroBlaze sem FPU, a inferência int8 leva {fi(int_s / 100)}\\,\\(\\mu\\)s, "
            f"{fd(int_s / 1e8 / 5 * 100, 3)}\\,\\% de uma janela, e mesmo o float32 emulado "
            f"({fd(s['float32'] / 1e5, 1)}\\,ms) seria aceitável; a rota em software não exige "
            f"nenhum recurso adicional de lógica.")
    if all(k in c for k in ("soft/cadeia_exato", "fpu/cadeia_exato", "soft/cadeia_hist", "fpu/cadeia_hist")):
        se, fe = c["soft/cadeia_exato"], c["fpu/cadeia_exato"]
        sh, fh = c["soft/cadeia_hist"], c["fpu/cadeia_hist"]
        val.update(mbHitSoft=fi(se["ciclos_hit"]), mbHitFpu=fi(fe["ciclos_hit"]),
                   mbHitHistSoft=fi(sh["ciclos_hit"]), mbHitHistFpu=fi(fh["ciclos_hit"]),
                   mbFecharSoft=fi(se["fechar"]), mbFecharFpu=fi(fe["fechar"]),
                   mbFecharHistSoft=fi(sh["fechar"]), mbFecharHistFpu=fi(fh["fechar"]))
        carga = resumo.get("carga_pct", {})
        u = {k: v for k, v in carga.items()}
        ens = max(u[k]["16"] for k in u)
        # limite superior "redondo" para frases do tipo "menos de X %"
        val["mbCargaEnsaioMax"] = str(int(np.ceil(ens))) if ens >= 0.5 else fd(np.ceil(ens * 10) / 10, 1)
        val["mbParagrafoCarga"] = (
            f"O custo por \\textit{{hit}} é dominado pelos logaritmos, emulados em software sem FPU: "
            f"{fi(se['ciclos_hit'])} ciclos sem FPU e {fi(fe['ciclos_hit'])} com FPU, no modo exato. "
            f"O histograma custa um pouco mais por \\textit{{hit}} ({fi(sh['ciclos_hit'])} e "
            f"{fi(fh['ciclos_hit'])} ciclos), porque calcula o índice da classe, mas fecha a janela "
            f"mais depressa ({fi(sh['fechar'])} contra {fi(se['fechar'])} ciclos sem FPU), porque não "
            f"ordena: sem FPU, cada comparação da ordenação é uma chamada à biblioteca de ponto flutuante. "
            f"Na taxa dos ensaios, cerca de 16 \\textit{{hits}}/s nos dois canais, a cadeia completa ocupa "
            f"{fd(u['soft/cadeia_exato']['16'], 2)}\\,\\% da CPU sem FPU e {fd(u['fpu/cadeia_exato']['16'], 2)}\\,\\% com FPU. "
            f"A 333 \\textit{{hits}}/s por canal (666 no total), a ocupação chega a "
            f"{fd(u['soft/cadeia_exato']['666'], 1)}\\,\\% sem FPU e {fd(u['fpu/cadeia_exato']['666'], 1)}\\,\\% com FPU; "
            f"nesse regime, a FPU passa a ser recomendável, ou o cálculo dos logaritmos por tabela.")
    linhas = ["% Gerado por embarcado/figuras_sim.py a partir de ciclos_mb.csv e dos ELFs. NÃO EDITAR."]
    for n in nomes:
        linhas.append(f"\\newcommand{{\\{n}}}{{{val[n]}}}")
    arq.write_text("\n".join(linhas) + "\n", encoding="utf-8")


def main():
    df = pd.read_csv(RAIZ / "resultados" / "janelas_descritores.csv")
    v = ler_vetores()
    resumo = {}
    fig_quantizacao(v, df, resumo)
    out = fig_descritores(df, resumo)
    concordancia(v, df, out, resumo)
    figs_ciclos(resumo)
    escrever_numeros(resumo)
    SAIDA.parent.mkdir(parents=True, exist_ok=True)
    (SAIDA.parent / "resumo_figuras.json").write_text(json.dumps(resumo, indent=2, ensure_ascii=False),
                                                      encoding="utf-8")
    print(json.dumps(resumo, indent=1, ensure_ascii=False)[:3000])


if __name__ == "__main__":
    main()
