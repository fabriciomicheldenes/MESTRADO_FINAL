"""
Importa o texto dos relatórios do projeto para seções da dissertação (gera capitulos/importado/).

Uso (a partir de Dissertacao/):
    ..\\..\\MLP\\.venv\\Scripts\\python.exe -I ferramentas\\importar_relatorios.py

Fontes (em ../../):
  ARTIGO_V2-10.11.2026/*.tex                     extrator de hits em FPGA (versão limpa do relatório)
  MLP/artigo/MLP_EA_bomba.tex                    classificação das condições da bomba
  MLP/ARTIGO_SIM_EMBARCADO-10.11.2026/*.tex      inferência embarcada verificada por simulação
  CWRU/RELATORIO_CWRU-10.11.2026/*.tex           validação externa na base CWRU

Transformações:
  * corpo do documento sem preâmbulo, resumo e bibliografia;
  * artigos de arquivo único: \\section -> \\subsection, \\subsection -> \\subsubsection;
  * rótulos e referências prefixados por grupo (v2:, mlp:, sim:, cwru:);
  * chaves de citação mapeadas para as do .bib da dissertação; citações cruzadas entre os
    relatórios viram referências a seções;
  * caminhos das figuras -> figs/<grupo>/ (as figuras são copiadas).
O texto gerado é revisado à mão depois; este script só faz a conversão mecânica.
"""
from __future__ import annotations

import re
import shutil
from pathlib import Path

AQUI = Path(__file__).resolve().parent.parent          # Dissertacao/
RAIZ = AQUI.parent.parent                              # Relatorio_Artigo/
SAIDA = AQUI / "capitulos" / "importado"

MAPA_CITE = {"neill1997": "Neill1997Detection", "alfayez2005": "Alfayez2005ndt"}
CRUZADAS = {
    "mlpbomba2026": r"Seção~\ref{sec:mm_bomba}",
    "simembarcado2026": r"Seção~\ref{sec:mm_embarcado}",
}


def corpo(tex: str) -> str:
    if r"\begin{document}" in tex:
        tex = tex.split(r"\begin{document}", 1)[1]
    tex = re.sub(r"\\maketitle", "", tex)
    tex = re.sub(r"\\begin\{abstract\}.*?\\end\{abstract\}", "", tex, flags=re.S)
    tex = re.split(r"\\begin\{thebibliography\}", tex)[0]
    tex = tex.replace(r"\end{document}", "")
    return tex


def rebaixar(tex: str) -> str:
    tex = re.sub(r"\\subsection(\*?)\{", r"\\subsubsectionTMP\1{", tex)
    tex = re.sub(r"\\section(\*?)\{", r"\\subsection\1{", tex)
    return tex.replace(r"\subsubsectionTMP", r"\subsubsection")


def prefixar(tex: str, g: str) -> str:
    tex = re.sub(r"\\label\{([^}]+)\}", lambda m: rf"\label{{{g}:{m.group(1)}}}", tex)
    tex = re.sub(r"\\(ref|eqref|pageref)\{([^}]+)\}", lambda m: rf"\{m.group(1)}{{{g}:{m.group(2)}}}", tex)
    # \ref{a} e~\ref{b} já cobertos; listas \ref{a,b} não são usadas
    return tex


def citacoes(tex: str) -> str:
    def troca(m):
        cmd, opt, chaves = m.group(1), m.group(2) or "", m.group(3)
        ks = [k.strip() for k in chaves.split(",")]
        cruz = [CRUZADAS[k] for k in ks if k in CRUZADAS]
        resto = [MAPA_CITE.get(k, k) for k in ks if k not in CRUZADAS]
        partes = []
        if resto:
            partes.append(rf"\{cmd}{opt}{{{','.join(resto)}}}")
        if cruz:
            partes.append("(" + "; ".join(cruz) + ")")
        return " ".join(partes)
    return re.sub(r"\\(cite[a-z]*)(\[[^\]]*\])?\{([^}]+)\}", troca, tex)


def figuras(tex: str, pasta: str) -> str:
    def troca(m):
        opt, arq = m.group(1) or "", m.group(2)
        nome = Path(arq).name
        return rf"\includegraphics{opt}{{figs/{pasta}/{nome}}}"
    return re.sub(r"\\includegraphics(\[[^\]]*\])?\{([^}]+)\}", troca, tex)


def separar(tex: str, marcadores: list[tuple[str, str]]) -> dict:
    """Divide pelo título de seção (após rebaixar): cada marcador inicia um bloco nomeado."""
    pos = []
    for nome, padrao in marcadores:
        m = re.search(padrao, tex)
        if not m:
            raise SystemExit(f"marcador não encontrado: {padrao}")
        pos.append((m.start(), nome))
    pos.sort()
    blocos = {}
    for i, (ini, nome) in enumerate(pos):
        fim = pos[i + 1][0] if i + 1 < len(pos) else len(tex)
        blocos[nome] = blocos.get(nome, "") + tex[ini:fim]
    return blocos


def gravar(nome: str, texto: str, origem: str):
    SAIDA.mkdir(parents=True, exist_ok=True)
    cab = (f"% Gerado por ferramentas/importar_relatorios.py a partir de {origem}.\n"
           f"% Revisado à mão na dissertação: edite este arquivo, não a fonte.\n")
    (SAIDA / f"{nome}.tex").write_text(cab + texto.strip() + "\n", encoding="utf-8")


def copiar_figs(origem: Path, pasta: str, padrao="*.png"):
    dest = AQUI / "figs" / pasta
    dest.mkdir(parents=True, exist_ok=True)
    for f in origem.glob(padrao):
        shutil.copy2(f, dest / f.name)


def main():
    # ---------------- ARTIGO_V2 (extrator) ----------------
    v2 = RAIZ / "ARTIGO_V2-10.11.2026"
    for arq, nome in (("2-Emissao_acustica_deteccao2.tex", "v2_hits_fundamentos"),
                      ("3-Decisoes_tecnologicas_metodologicas2.tex", "v2_decisoes"),
                      ("4-Implementacao_sistema2.tex", "v2_implementacao"),
                      ("5-Delineamento_experimental2.tex", "v2_delineamento"),
                      ("6-Resultados_discussao2.tex", "v2_resultados"),
                      ("7-Conclusoes2.tex", "v2_conclusoes")):
        t = (v2 / arq).read_text(encoding="utf-8")
        t = citacoes(prefixar(t, "v2"))
        gravar(nome, t, f"ARTIGO_V2-10.11.2026/{arq}")

    # ---------------- MLP da bomba ----------------
    f = RAIZ / "MLP" / "artigo" / "MLP_EA_bomba.tex"
    t = figuras(citacoes(prefixar(rebaixar(corpo(f.read_text(encoding="utf-8"))), "mlp")), "mlp")
    b = separar(t, [("intro", r"\\subsection\{Introdução\}"),
                    ("mm", r"\\subsection\{Dados e delineamento\}"),
                    ("res", r"\\subsection\{Resultados\}"),
                    ("concl", r"\\subsection\{Conclusões\}")])
    gravar("mlp_introducao", b["intro"], "MLP/artigo/MLP_EA_bomba.tex")
    gravar("mlp_metodos", b["mm"], "MLP/artigo/MLP_EA_bomba.tex")
    gravar("mlp_resultados", b["res"], "MLP/artigo/MLP_EA_bomba.tex")
    gravar("mlp_conclusoes", b["concl"], "MLP/artigo/MLP_EA_bomba.tex")
    copiar_figs(RAIZ / "MLP" / "artigo" / "figuras", "mlp")

    # ---------------- Inferência embarcada ----------------
    d = RAIZ / "MLP" / "ARTIGO_SIM_EMBARCADO-10.11.2026"
    t = corpo((d / "ARTIGO_SIM_EMBARCADO.tex").read_text(encoding="utf-8"))
    t = figuras(citacoes(prefixar(rebaixar(t), "sim")), "embarcado")
    b = separar(t, [("intro", r"\\subsection\{Introdução\}"),
                    ("mm", r"\\subsection\{Fundamentos\}"),
                    ("res", r"\\subsection\{Resultados\}"),
                    ("concl", r"\\subsection\{Conclusões\}")])
    for k in ("intro", "mm", "res", "concl"):
        gravar(f"sim_{ {'intro': 'introducao', 'mm': 'metodos', 'res': 'resultados', 'concl': 'conclusoes'}[k] }",
               b[k], "MLP/ARTIGO_SIM_EMBARCADO-10.11.2026/ARTIGO_SIM_EMBARCADO.tex")
    copiar_figs(d / "figuras", "embarcado")
    shutil.copy2(d / "numeros_mb.tex", AQUI / "numeros_mb.tex")

    # ---------------- CWRU ----------------
    d = RAIZ / "CWRU" / "RELATORIO_CWRU-10.11.2026"
    t = corpo((d / "RELATORIO_CWRU.tex").read_text(encoding="utf-8"))
    t = figuras(citacoes(prefixar(rebaixar(t), "cwru")), "cwru")
    b = separar(t, [("intro", r"\\subsection\{Objetivo e enquadramento\}"),
                    ("mm", r"\\subsection\{Base de dados\}"),
                    ("res", r"\\subsection\{Resultados\}"),
                    ("concl", r"\\subsection\{Conclusões\}")])
    for k, n in (("intro", "introducao"), ("mm", "metodos"), ("res", "resultados"), ("concl", "conclusoes")):
        gravar(f"cwru_{n}", b[k], "CWRU/RELATORIO_CWRU-10.11.2026/RELATORIO_CWRU.tex")
    copiar_figs(RAIZ / "CWRU" / "resultados", "cwru")
    print("ok:", sorted(p.name for p in SAIDA.glob("*.tex")))


if __name__ == "__main__":
    main()
