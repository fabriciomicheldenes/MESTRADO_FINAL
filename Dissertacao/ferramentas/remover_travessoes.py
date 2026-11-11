"""Remove travessões (—) do texto ativo (anotação da banca): pares 'a — b — c' viram
'a (b) c'; travessão isolado vira vírgula. Linhas comentadas não são alteradas."""
import re
from pathlib import Path

D = Path(__file__).resolve().parent.parent / "capitulos"
arqs = list((D / "FundamentacaoTeorica").glob("*.tex")) + list((D / "importado").glob("*.tex")) + list(D.glob("*.tex"))
total = 0
for f in arqs:
    linhas = f.read_text(encoding="utf-8").split("\n")
    mud = 0
    for i, l in enumerate(linhas):
        if l.lstrip().startswith("%") or "—" not in l:
            continue
        novo = re.sub(r"\s*—\s*([^—]+?)\s*—\s*", lambda m: f" ({m.group(1).strip()}) ", l)
        novo = re.sub(r"\s*—\s*", ", ", novo)
        novo = novo.replace(" ,", ",").replace("( ", "(").replace(" )", ")")
        if novo != l:
            linhas[i] = novo
            mud += 1
    if mud:
        f.write_text("\n".join(linhas), encoding="utf-8")
        total += mud
        print(f"{f.name}: {mud} linhas")
print("total:", total)
