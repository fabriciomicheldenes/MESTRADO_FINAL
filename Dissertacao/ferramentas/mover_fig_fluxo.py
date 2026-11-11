"""Corrige a posição da figura do fluxo de desenvolvimento: move o parágrafo e a figura,
inseridos após um título comentado, para logo depois do título ativo da subseção."""
from pathlib import Path

f = Path(__file__).resolve().parent.parent / "capitulos" / "FundamentacaoTeorica" / "03-FPGA.tex"
t = f.read_text(encoding="utf-8")
ini = t.find("A Figura~\\ref{fig:FluxoDesenvolvimentoFPGA} resume o fluxo")
fim = t.find("\\end{figure}", ini) + len("\\end{figure}")
bloco = t[ini:fim]
t = t[:ini] + t[fim:]
alvo = "\n\\subsection*{Fluxo de Desenvolvimento com FPGA}\n"
k = t.find(alvo)
if k < 0:
    raise SystemExit("título ativo não encontrado")
k += len(alvo)
t = t[:k] + "\n" + bloco + "\n" + t[k:]
f.write_text(t, encoding="utf-8")
print("figura movida para depois do título ativo")
