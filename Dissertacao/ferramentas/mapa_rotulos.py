"""Mostra quais rótulos definidos nos capítulos 1-3 são referenciados nos capítulos 4-5,
para saber o que precisa ser preservado ao enxugar os capítulos iniciais."""
import re
from pathlib import Path

D = Path(__file__).resolve().parent.parent / "capitulos"
INICIAIS = ["introducao.tex", "FundamentacaoTeorica/01-EmissaoAcustica.tex", "importado/v2_hits_fundamentos.tex",
            "FundamentacaoTeorica/02-AprendizadoMaquina.tex", "FundamentacaoTeorica/03-FPGA.tex",
            "MateriaisMetodos.tex", "importado/v2_decisoes.tex", "importado/v2_implementacao.tex",
            "importado/v2_delineamento.tex", "importado/mlp_metodos.tex", "importado/sim_metodos.tex",
            "importado/cwru_introducao.tex", "importado/cwru_metodos.tex"]
FINAIS = ["Resultados.tex", "Conclusoes.tex", "importado/v2_resultados.tex", "importado/mlp_resultados.tex",
          "importado/sim_resultados.tex", "importado/cwru_resultados.tex"]


def ativo(t):
    return "\n".join(l for l in t.split("\n") if not l.lstrip().startswith("%"))


defs = {}
for f in INICIAIS:
    for m in re.finditer(r"\\label\{([^}]+)\}", ativo((D / f).read_text(encoding="utf-8"))):
        defs[m.group(1)] = f
usos = {}
for f in FINAIS:
    for m in re.finditer(r"\\(?:ref|eqref|pageref)\{([^}]+)\}", ativo((D / f).read_text(encoding="utf-8"))):
        if m.group(1) in defs:
            usos.setdefault(m.group(1), set()).add(f)
for k in sorted(usos, key=lambda k: defs[k]):
    print(f"{defs[k]:40s} {k:45s} <- {', '.join(sorted(usos[k]))}")
