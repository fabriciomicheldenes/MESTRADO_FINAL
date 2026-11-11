"""Remove as marcas \\novo{...} e o ambiente novobloco (destaques de revisão do relatório)
dos arquivos importados, mantendo o conteúdo. Uso: python -I ferramentas/limpar_novo.py"""
import re
from pathlib import Path

PASTA = Path(__file__).resolve().parent.parent / "capitulos" / "importado"
MARCA = "\\novo{"


def tira_novo(t: str) -> str:
    out, i = [], 0
    while True:
        j = t.find(MARCA, i)
        if j < 0:
            out.append(t[i:])
            return "".join(out)
        out.append(t[i:j])
        k, p = j + len(MARCA), 1
        while k < len(t) and p:
            c = t[k]
            if c == "\\":
                k += 2
                continue
            p += {"{": 1, "}": -1}.get(c, 0)
            k += 1
        out.append(t[j + len(MARCA):k - 1])
        i = k


for f in sorted(PASTA.glob("*.tex")):
    t = f.read_text(encoding="utf-8")
    n0 = t.count(MARCA)
    while MARCA in t:
        t = tira_novo(t)
    t = re.sub(r"\\(?:begin|end)\{novobloco\}[ \t]*\n?", "", t)
    f.write_text(t, encoding="utf-8")
    if n0:
        print(f"{f.name}: {n0} marcas removidas")
