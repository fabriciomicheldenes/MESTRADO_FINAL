"""Remove do Main.tex o resumo da qualificação (bloco que começa por 'Este trabalho investiga')."""
from pathlib import Path

p = Path(__file__).resolve().parent.parent / "Main.tex"
t = p.read_text(encoding="utf-8")
ini = t.find("\\begin{resumo}\n\tEste trabalho investiga")
if ini < 0:
    raise SystemExit("bloco antigo não encontrado")
fim = t.find("\\end{resumo}", ini) + len("\\end{resumo}")
t = t[:ini] + t[fim:]
p.write_text(t, encoding="utf-8")
print("resumo antigo removido")
