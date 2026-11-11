"""Ajustes de estrutura na Fundamentação Teórica (executar uma única vez).
- rótulos nas seções de AM, MLP e FPGA;
- move a subseção 'MLP em FPGA para diagnóstico de falhas' para o fim da seção de FPGA
  (anotação da banca: os trabalhos com MLP em FPGA vêm depois da apresentação dos FPGAs);
- remove da seção de hits importada a subseção que repete a cadeia de medição de EA."""
from pathlib import Path

D = Path(__file__).resolve().parent.parent / "capitulos"
am = D / "FundamentacaoTeorica" / "02-AprendizadoMaquina.tex"
fp = D / "FundamentacaoTeorica" / "03-FPGA.tex"
hi = D / "importado" / "v2_hits_fundamentos.tex"

t = am.read_text(encoding="utf-8")
if "\\label{sec:AM}" not in t:
    t = t.replace("\\section{Aprendizado de Máquina}", "\\section{Aprendizado de Máquina}\n\\label{sec:AM}", 1)
    t = t.replace("\\section{Redes Neurais MLP}", "\\section{Redes Neurais MLP}\n\\label{sec:MLP}", 1)
marca = "\\subsection{MLP em FPGA para diagnóstico de falhas}"
bloco = ""
if marca in t:
    i = t.index(marca)
    bloco = t[i:]
    t = t[:i].rstrip() + "\n"
am.write_text(t, encoding="utf-8")

f = fp.read_text(encoding="utf-8")
if "\\label{sec:FPGA}" not in f:
    f = f.replace("\\section{Field-Programmable Gate Arrays (FPGAs)}",
                  "\\section{\\textit{Field-Programmable Gate Arrays} (FPGAs)}\n\\label{sec:FPGA}", 1)
if bloco and marca not in f:
    f = f.rstrip() + "\n\n% Movida de 02-AprendizadoMaquina.tex (anotação da banca de qualificação)\n" + bloco
fp.write_text(f, encoding="utf-8")

h = hi.read_text(encoding="utf-8")
a = h.find("\\subsection{Fundamentos da técnica e cadeia de medição}")
b = h.find("\\subsection{Definição operacional do \\textit{hit}}")
if a >= 0 and b > a:
    h = h[:a] + h[b:]
hi.write_text(h, encoding="utf-8")
print("ok", len(bloco))
