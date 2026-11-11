"""Mapeia rótulos de seção dos artigos originais (definidos nos arquivos principais, que não
foram importados) para as seções equivalentes da dissertação, e remove do Main.tex o
ambiente otherlanguage* que quebra o resumo em inglês do abnTeX2."""
import re
from pathlib import Path

D = Path(__file__).resolve().parent.parent
MAPA = {
    "v2:sec:introducao": "chap:introducao",
    "v2:sec:emissao_acustica": "sec:fund_hits",
    "v2:sec:decisoes_projeto": "sec:mm_decisoes",
    "v2:sec:implementacao_sistema": "sec:mm_implementacao",
    "v2:sec:delineamento_experimental": "sec:mm_delineamento",
    "v2:sec:resultados_discussao": "sec:res_extrator",
    "v2:sec:conclusoes": "chap:conclusoes",
}
for f in (D / "capitulos" / "importado").glob("*.tex"):
    t = f.read_text(encoding="utf-8")
    t2 = t
    for a, b in MAPA.items():
        t2 = t2.replace("{" + a + "}", "{" + b + "}")
    if t2 != t:
        f.write_text(t2, encoding="utf-8")
        print("rótulos ajustados em", f.name)

m = D / "Main.tex"
t = m.read_text(encoding="utf-8")
t2 = re.sub(r"[ \t]*\\(?:begin|end)\{otherlanguage\*\}(?:\{english\})?[ \t]*\n", "", t)
if t2 != t:
    m.write_text(t2, encoding="utf-8")
    print("otherlanguage* removido do Main.tex")
