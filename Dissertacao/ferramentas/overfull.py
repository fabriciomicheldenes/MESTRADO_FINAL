"""Lista os estouros de margem (Overfull \\hbox) do Main.log com o arquivo-fonte e a linha."""
import re
from pathlib import Path

log = (Path(__file__).resolve().parent.parent / "Main.log").read_text(encoding="latin-1")
pilha, res = [], []
padrao = re.compile(r"\((\./[^\s()]+\.tex)|\)|Overfull \\hbox \(([0-9.]+)pt too wide\) (?:in paragraph|detected) at lines? (\d+)")
for tok in padrao.finditer(log):
    if tok.group(1):
        pilha.append(tok.group(1))
    elif tok.group(0) == ")":
        if pilha:
            pilha.pop()
    elif tok.group(2) and float(tok.group(2)) > 4:
        res.append((float(tok.group(2)), pilha[-1] if pilha else "?", int(tok.group(3))))
for r in sorted(res, reverse=True):
    print(f"{r[0]:7.1f} pt  {r[1]}:{r[2]}")
