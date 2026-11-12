"""
Mede ciclos na MicroBlaze por simulação RTL (Vivado/xsim 2022.2), sem placa.

Uso (a partir desta pasta; os ELFs já compilados em elf/):
    ..\\..\\..\\.venv\\Scripts\\python.exe rodar_sim_mb.py [--trabalho I:/Momea/2026/mbsim]

Para cada configuração da MicroBlaze (sem FPU / com FPU estendida):
  1. Vivado (batch): cria o sistema (sim_mb.tcl) e gera a imagem .mem de cada ELF;
  2. compila e elabora o testbench uma vez (compile.bat, elaborate.bat);
  3. para cada ELF: instala o .mem, simula (simulate.bat) e lê marcas.txt.
Grava resultados/marcas_<cfg>_<elf>.txt e resultados/ciclos_mb.csv, e confere os valores
das marcas de resultado com a execução do mesmo programa no PC (ref_host_*.txt).
O diretório de trabalho fica fora do projeto porque o Vivado gera caminhos longos.
"""
from __future__ import annotations

import argparse
import csv
import os
import re
import shutil
import subprocess
import time
from pathlib import Path

AQUI = Path(__file__).resolve().parent
VIVADO_BIN = Path(r"E:\Xilinx\Vivado\2022.2\bin")
CONFIGS = {"soft": 0, "fpu": 2}
ELFS = ["mlp", "cadeia_hist", "cadeia_exato"]


def rodar(cmd: list, cwd: Path, log: Path, env=None) -> int:
    with open(log, "w", encoding="utf-8", errors="replace") as f:
        p = subprocess.run([str(c) for c in cmd], cwd=cwd, stdout=f, stderr=subprocess.STDOUT,
                           stdin=subprocess.DEVNULL, env=env)
    return p.returncode


def n_hits_por_janela() -> list[int]:
    txt = (AQUI.parents[1] / "gerado" / "hits_mb.h").read_text(encoding="utf-8")
    ini = [int(v) for v in re.search(r"HMB_INICIO\[\] = \{([^}]*)\}", txt).group(1).split(",")]
    return [b - a for a, b in zip(ini, ini[1:])]


def ler_marcas(p: Path) -> list[tuple[int, int]]:
    out = []
    for lin in p.read_text().splitlines():
        c, v = lin.split()
        out.append((int(c), int(v, 16)))
    return out


def analisar(nome: str, marcas: list[tuple[int, int]], nh: list[int]) -> list[dict]:
    t = {v: c for c, v in marcas}
    linhas = []
    if nome == "mlp":
        j = 0
        while (0x11000000 | j) in t:
            c11, c12 = t[0x11000000 | j], t[0x12000000 | j]
            c13, c14 = t[0x13000000 | j], t[0x14000000 | j]
            linhas.append(dict(etapa="mlp_float32_2redes", indice=j, ciclos=c12 - c11))
            linhas.append(dict(etapa="quantizacao_entrada", indice=j, ciclos=c13 - c12))
            linhas.append(dict(etapa="nucleo_int8_2redes", indice=j, ciclos=c14 - c13))
            j += 1
    else:
        w = 0
        while (0x01000000 | w) in t:
            c1, c2 = t[0x01000000 | w], t[0x02000000 | w]
            c3, c4 = t[0x03000000 | w], t[0x04000000 | w]
            linhas.append(dict(etapa="hits_total", indice=w, ciclos=c2 - c1, hits=nh[w]))
            linhas.append(dict(etapa="hit_medio", indice=w, ciclos=round((c2 - c1) / nh[w], 1),
                               hits=nh[w]))
            linhas.append(dict(etapa="fechar_janela", indice=w, ciclos=c3 - c2))
            linhas.append(dict(etapa="mlp_int8_com_quantizacao", indice=w, ciclos=c4 - c3))
            w += 1
    return linhas


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trabalho", default="I:/Momea/2026/mbsim")
    ap.add_argument("--configs", default="soft,fpu")
    ap.add_argument("--elfs", default=",".join(ELFS),
                    help="programas a simular; cada chamada usa seu próprio projeto, o que "
                         "permite rodar vários em paralelo")
    args = ap.parse_args()
    elfs_sel = args.elfs.split(",")
    sufixo = "" if elfs_sel == ELFS else "_" + "_".join(elfs_sel)
    trab = Path(args.trabalho)
    trab.mkdir(parents=True, exist_ok=True)
    res = AQUI / "resultados"
    res.mkdir(exist_ok=True)
    env = dict(os.environ)
    env["PATH"] = str(VIVADO_BIN) + os.pathsep + env["PATH"]
    nh = n_hits_por_janela()
    tabela = []

    for cfg in args.configs.split(","):
        proj = trab / f"{cfg}{sufixo}"
        elfs = [AQUI / "elf" / f"{e}_{cfg}.elf" for e in elfs_sel]
        t0 = time.time()
        if proj.exists():
            shutil.rmtree(proj)
        cmd = [VIVADO_BIN / "vivado.bat", "-mode", "batch", "-nojournal", "-log",
               trab / f"viv_{cfg}{sufixo}.log", "-source", AQUI / "sim_mb.tcl", "-tclargs",
               proj.as_posix(), CONFIGS[cfg]] + [e.as_posix() for e in elfs]
        rc = rodar(cmd, trab, trab / f"viv_{cfg}{sufixo}_saida.txt", env)
        print(f"[{cfg}] Vivado (projeto + imagens): rc={rc}, {time.time() - t0:.0f} s")
        if rc:
            raise SystemExit(f"falhou; ver {trab / f'viv_{cfg}{sufixo}_saida.txt'}")
        xs = proj / "mbsim.sim" / "sim_1" / "behav" / "xsim"
        for e in elfs_sel:   # a imagem tem de vir do ELF, não do bootloop padrão da MicroBlaze
            if f"{e}_{cfg}.elf" not in (proj / f"{e}_{cfg}.mem").read_text(errors="replace")[:2000]:
                raise SystemExit(f"imagem {e}_{cfg}.mem não veio do ELF")
        nome_mem = (proj / "nome_mem.txt").read_text().strip()
        tcl = xs / "tb_mb.tcl"
        txt = tcl.read_text()
        if "quit" not in txt:
            tcl.write_text(txt.rstrip() + "\nquit\n")
        for etapa in ("compile", "elaborate"):
            t0 = time.time()
            rc = rodar([xs / f"{etapa}.bat"], xs, xs / f"{etapa}_saida.txt", env)
            print(f"[{cfg}] {etapa}: rc={rc}, {time.time() - t0:.0f} s")
            if rc:
                raise SystemExit(f"falhou; ver {xs / f'{etapa}_saida.txt'}")
        for e in elfs_sel:
            nome = f"{e}_{cfg}"
            shutil.copyfile(proj / f"{nome}.mem", xs / nome_mem)
            (xs / "marcas.txt").unlink(missing_ok=True)
            t0 = time.time()
            rc = rodar([xs / "simulate.bat"], xs, xs / f"simulate_{nome}.txt", env)
            dt = time.time() - t0
            marcas_arq = xs / "marcas.txt"
            if not marcas_arq.exists():
                raise SystemExit(f"sem marcas.txt para {nome} (rc={rc}); ver simulate_{nome}.txt")
            shutil.copyfile(marcas_arq, res / f"marcas_{nome}.txt")
            marcas = ler_marcas(marcas_arq)
            ult = marcas[-1]
            # confere valores de resultado com a execução no PC
            curto = e.replace("cadeia_", "")
            arq_ref = AQUI / f"ref_host_{curto}_{cfg}.txt"   # referência própria da config
            if not arq_ref.exists():
                arq_ref = AQUI / f"ref_host_{curto}.txt"
            ref = arq_ref.read_text().split()
            sim_vals = [f"{v:08x}" for _, v in marcas
                        if (v >> 24) not in (0x01, 0x02, 0x03, 0x04, 0x11, 0x12, 0x13, 0x14)]
            igual = sim_vals == ref
            print(f"[{cfg}] {e}: {ult[0]} ciclos simulados em {dt:.0f} s; marca final "
                  f"{ult[1]:08x}; resultados iguais ao PC: {igual}")
            for lin in analisar(e, marcas, nh):
                tabela.append(dict(config=cfg, programa=e, **lin, confere_pc=igual,
                                   ciclos_totais=ult[0]))

    # um CSV por conjunto de configurações (permite rodar soft e fpu em paralelo)
    saida_csv = res / f"ciclos_mb_{args.configs.replace(',', '_')}{sufixo}.csv"
    with open(saida_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["config", "programa", "etapa", "indice", "ciclos",
                                          "hits", "confere_pc", "ciclos_totais"])
        w.writeheader()
        w.writerows(tabela)
    print(f"gravado {saida_csv}")


if __name__ == "__main__":
    main()
