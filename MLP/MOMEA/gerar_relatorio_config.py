# -*- coding: utf-8 -*-
"""
gerar_relatorio_config.py - Gera o relatório de configuração e uso (.docx) de mlp_ea.py:
como rodar, opções, e como os dados foram separados em treino, validação e teste em cada
esquema. Rodar depois de mlp_ea.py e figuras_artigo.py.

Uso:  .venv\\Scripts\\python.exe gerar_relatorio_config.py
Saída: Relatorio_Configuracao_Uso_MLP.docx
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from docx import Document
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Pt, RGBColor

import mlp_ea as M

AQUI = Path(__file__).resolve().parent
FIG = AQUI / "artigo" / "figuras"
RES = AQUI / "resultados"
SAIDA = AQUI / "Relatorio_Configuracao_Uso_MLP.docx"
AZUL_CAB = "D9E2F3"
FONTE = "Arial"


# --------------------------------------------------------------------------- #
# Utilidades de formatação
# --------------------------------------------------------------------------- #
def _fonte_run(run, tam=None, negrito=None, italico=None, mono=False, cor=None):
    nome = "Consolas" if mono else FONTE
    run.font.name = nome
    rpr = run._element.get_or_add_rPr()
    rf = rpr.find(qn("w:rFonts"))
    if rf is None:
        rf = OxmlElement("w:rFonts")
        rpr.append(rf)
    for a in ("w:ascii", "w:hAnsi", "w:cs", "w:eastAsia"):
        rf.set(qn(a), nome)
    if tam:
        run.font.size = Pt(tam)
    if negrito is not None:
        run.bold = negrito
    if italico is not None:
        run.italic = italico
    if cor:
        run.font.color.rgb = RGBColor.from_string(cor)


def _sombrear(cell, cor):
    tcpr = cell._element.get_or_add_tcPr()
    shd = OxmlElement("w:shd")
    shd.set(qn("w:val"), "clear")
    shd.set(qn("w:color"), "auto")
    shd.set(qn("w:fill"), cor)
    tcpr.append(shd)


def _repetir_cabecalho(row):
    trpr = row._tr.get_or_add_trPr()
    el = OxmlElement("w:tblHeader")
    el.set(qn("w:val"), "true")
    trpr.append(el)


class Doc:
    def __init__(self):
        self.d = Document()
        s = self.d.sections[0]
        s.page_width, s.page_height = Cm(21.0), Cm(29.7)
        s.left_margin = s.right_margin = Cm(2.5)
        s.top_margin = s.bottom_margin = Cm(2.5)
        st = self.d.styles["Normal"]
        st.font.name = FONTE
        st.font.size = Pt(11)
        st.element.rPr.rFonts.set(qn("w:eastAsia"), FONTE)
        st.paragraph_format.space_after = Pt(6)
        st.paragraph_format.line_spacing = 1.15
        for nome, tam in (("Title", 18), ("Heading 1", 14), ("Heading 2", 12), ("Heading 3", 11)):
            h = self.d.styles[nome]
            h.font.name = FONTE
            h.font.size = Pt(tam)
            h.font.bold = True
            h.font.color.rgb = RGBColor(0x1F, 0x3A, 0x5F)
            rpr = h.element.get_or_add_rPr()
            rf = rpr.find(qn("w:rFonts"))
            if rf is None:
                rf = OxmlElement("w:rFonts")
                rpr.append(rf)
            for a in ("w:ascii", "w:hAnsi", "w:cs", "w:eastAsia"):
                rf.set(qn(a), FONTE)
            for a in ("w:asciiTheme", "w:hAnsiTheme", "w:cstheme", "w:eastAsiaTheme"):
                if rf.get(qn(a)) is not None:
                    del rf.attrib[qn(a)]
            h.paragraph_format.space_before = Pt(12 if nome != "Heading 3" else 8)
            h.paragraph_format.space_after = Pt(6)
            h.paragraph_format.keep_with_next = True
        self.nfig = 0
        self.ntab = 0
        # rodapé com número de página
        p = s.footer.paragraphs[0]
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        r = p.add_run()
        for tipo, txt in (("begin", None), (None, "PAGE"), ("end", None)):
            if tipo:
                el = OxmlElement("w:fldChar")
                el.set(qn("w:fldCharType"), tipo)
            else:
                el = OxmlElement("w:instrText")
                el.set(qn("xml:space"), "preserve")
                el.text = txt
            r._element.append(el)
        _fonte_run(r, 9)

    def titulo(self, txt, sub=None):
        p = self.d.add_paragraph(style="Title")
        r = p.add_run(txt)
        _fonte_run(r, 18, True, cor="1F3A5F")
        if sub:
            p = self.d.add_paragraph()
            r = p.add_run(sub)
            _fonte_run(r, 10, italico=True, cor="52514E")

    def h(self, txt, nivel=1):
        self.d.add_heading(txt, level=nivel)

    def p(self, *partes, alinhar=True):
        """partes: str ou (str, 'b'|'i'|'c'), 'c' = código inline."""
        par = self.d.add_paragraph()
        if alinhar:
            par.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
        for x in partes:
            txt, fmt = (x, "") if isinstance(x, str) else x
            r = par.add_run(txt)
            _fonte_run(r, 10 if "c" in fmt else None, negrito="b" in fmt or None,
                       italico="i" in fmt or None, mono="c" in fmt)
        return par

    def lista(self, itens, numerada=False):
        for it in itens:
            par = self.d.add_paragraph(style="List Number" if numerada else "List Bullet")
            partes = it if isinstance(it, list) else [it]
            for x in partes:
                txt, fmt = (x, "") if isinstance(x, str) else x
                r = par.add_run(txt)
                _fonte_run(r, 10 if "c" in fmt else None, negrito="b" in fmt or None,
                           italico="i" in fmt or None, mono="c" in fmt)

    def codigo(self, txt):
        for linha in txt.strip("\n").split("\n"):
            par = self.d.add_paragraph()
            par.paragraph_format.space_after = Pt(0)
            par.paragraph_format.left_indent = Cm(0.5)
            pPr = par._element.get_or_add_pPr()
            shd = OxmlElement("w:shd")
            shd.set(qn("w:val"), "clear")
            shd.set(qn("w:color"), "auto")
            shd.set(qn("w:fill"), "F2F2F2")
            pPr.append(shd)
            r = par.add_run(linha if linha else " ")
            _fonte_run(r, 9, mono=True)
        self.d.add_paragraph().paragraph_format.space_after = Pt(2)

    def tabela(self, cab, linhas, larguras=None, legenda=None, tam=9):
        if legenda:
            self.ntab += 1
            par = self.d.add_paragraph()
            par.paragraph_format.keep_with_next = True
            r = par.add_run(f"Tabela {self.ntab} – ")
            _fonte_run(r, 9, True)
            r = par.add_run(legenda)
            _fonte_run(r, 9)
        t = self.d.add_table(rows=1, cols=len(cab))
        t.style = "Table Grid"
        t.alignment = WD_TABLE_ALIGNMENT.CENTER
        for j, c in enumerate(cab):
            cell = t.rows[0].cells[j]
            cell.text = ""
            r = cell.paragraphs[0].add_run(c)
            _fonte_run(r, tam, True)
            _sombrear(cell, AZUL_CAB)
        _repetir_cabecalho(t.rows[0])
        for lin in linhas:
            cells = t.add_row().cells
            for j, v in enumerate(lin):
                cells[j].text = ""
                partes = v if isinstance(v, list) else [v]
                par = cells[j].paragraphs[0]
                for x in partes:
                    txt, fmt = (str(x), "") if not isinstance(x, tuple) else x
                    r = par.add_run(txt)
                    _fonte_run(r, tam, negrito="b" in fmt or None, mono="c" in fmt)
        if larguras:
            for row in t.rows:
                for j, w in enumerate(larguras):
                    row.cells[j].width = Cm(w)
        self.d.add_paragraph().paragraph_format.space_after = Pt(2)
        return t

    def figura(self, nome, legenda, largura=16.0):
        par = self.d.add_paragraph()
        par.alignment = WD_ALIGN_PARAGRAPH.CENTER
        par.paragraph_format.keep_with_next = True
        par.add_run().add_picture(str(FIG / nome), width=Cm(largura))
        self.nfig += 1
        par = self.d.add_paragraph()
        par.alignment = WD_ALIGN_PARAGRAPH.CENTER
        r = par.add_run(f"Figura {self.nfig} – ")
        _fonte_run(r, 9, True)
        r = par.add_run(legenda)
        _fonte_run(r, 9)

    def salvar(self, caminho):
        self.d.save(caminho)


# --------------------------------------------------------------------------- #
def numeros():
    df = pd.read_csv(RES / "janelas_descritores.csv")
    N = len(df)
    folds = {}
    for esq in ("BLOCOS", "LORO", "LOSO"):
        fs = M.gerar_folds(df, esq, 5)
        tr = np.array([len(a) for _, a, _ in fs])
        te = np.array([len(b) for _, _, b in fs])
        folds[esq] = dict(n=len(fs), tr=(tr.min(), tr.max(), tr.mean() / N),
                          te=(te.min(), te.max(), te.mean() / N), ex=(N - tr - te))
    vi = pd.read_csv(FIG / "validacao_interna_vs_teste_resumo.csv")
    oot = pd.read_csv(FIG / "oot_por_dia.csv")
    met = pd.read_csv(RES / "metricas.csv")
    return df, N, folds, vi, oot, met


def f2(x, d=2):
    return f"{x:.{d}f}".replace(".", ",")


def pct(x, d=1):
    return f"{100 * x:.{d}f} %".replace(".", ",")


def main():
    df, N, folds, vi, oot, met = numeros()
    tot = met[met.fold == "TOTAL"].set_index(["esquema", "tarefa", "modelo"])
    D = Doc()
    D.titulo("Relatório de configuração e uso – classificação das condições da bomba por EA (mlp_ea.py)",
             "MOMEA – MLP, random forest e gradient boosting sobre hits agregados em janelas. "
             "Pasta: I:\\Momea\\2026\\Artigo\\Relatorio_Artigo\\MLP. Versão de 10/11/2026.")

    # 1 ------------------------------------------------------------------
    D.h("1. Objetivo")
    D.p("Este documento descreve como configurar e executar o script ", ("mlp_ea.py", "c"),
        " e, principalmente, como os dados foram separados em treino, validação e teste em cada "
        "esquema de avaliação. O texto de referência sobre conjuntos de dados (treino de 70–80 %, "
        "validação de ~15 % e teste de 15–30 %, com um quarto conjunto opcional) é usado como guia. "
        "Para cada um desses conjuntos, o documento explica o que foi feito aqui e por que a divisão "
        "aleatória não se aplica a estes dados. Os resultados e sua interpretação estão no artigo ",
        ("artigo/MLP_EA_bomba.pdf", "c"), " e no resumo ", ("RESUMO_RESULTADOS.md", "c"), ".")
    D.tabela(["Arquivo", "Função"], [
        [("mlp_ea.py", "c"), "Pipeline completo: leitura, janelas, descritores, validação (BLOCOS, LORO, LOSO), "
                             "importâncias, k-means/PCA, figuras e relatório automático."],
        [("figuras_artigo.py", "c"), "Figuras complementares, diagramas da divisão e dois experimentos curtos "
                                     "(validação interna × teste; holdout fora do tempo entre dias)."],
        [("gerar_relatorio_config.py", "c"), "Gera este relatório (.docx)."],
        [("artigo/MLP_EA_bomba.tex", "c"), "Artigo técnico em LaTeX (pdflatex), com figuras em artigo/figuras/."],
        [("requirements.txt", "c"), "Versões das bibliotecas do ambiente virtual .venv."],
        [("Dados/", "c"), "56 corridas: Dados/<VALV>/<NN_xxHZ-...>/<data_hora>/hits.csv, config.json, summary.json."],
    ], larguras=[5.2, 10.8], legenda="Arquivos da pasta MLP.")

    # 2 ------------------------------------------------------------------
    D.h("2. Ambiente")
    D.p("O Python 3.15 em ", ("C:\\Python315", "c"), " não tinha bibliotecas científicas. Por isso, foi "
        "criado um ambiente virtual em ", ("MLP\\.venv", "c"), " com numpy, pandas, scikit-learn 1.9.1, "
        "matplotlib, tabulate e python-docx. Para recriá-lo:")
    D.codigo(r"""
cd I:\Momea\2026\Artigo\Relatorio_Artigo\MLP
C:\Python315\python.exe -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
.venv\Scripts\python.exe -m pip install python-docx
""")

    # 3 ------------------------------------------------------------------
    D.h("3. Como rodar")
    D.p("Execução completa, com janela de 5 s e os três esquemas (cerca de 5 min num PC de 8 núcleos):")
    D.codigo(r"""
.venv\Scripts\python.exe mlp_ea.py
.venv\Scripts\python.exe figuras_artigo.py          (figuras extras, ~2 min)
.venv\Scripts\python.exe gerar_relatorio_config.py  (este documento)
cd artigo
E:\texlive\2022\bin\win32\pdflatex.exe MLP_EA_bomba.tex   (rodar 2 vezes)
""")
    D.p("Variantes de sensibilidade usadas no artigo:")
    D.codigo(r"""
.venv\Scripts\python.exe mlp_ea.py --sem-harmonizar --esquemas LORO,LOSO --saida resultados_sensib\sem_harmonizar
.venv\Scripts\python.exe mlp_ea.py --janela 2 --esquemas LORO,LOSO --saida resultados_sensib\janela_2s
""")
    D.tabela(["Opção", "Padrão", "Descrição"], [
        [("--dados", "c"), ("MLP\\Dados", "c"), "Pasta raiz das corridas. Os rótulos são lidos do caminho: a pasta de "
                                                 "1º nível dá as válvulas (100REC_50SUC) e a de 2º nível dá a "
                                                 "frequência (01_40HZ-...)."],
        [("--saida", "c"), ("MLP\\resultados", "c"), "Pasta de saída (CSV, PNG, relatorio_auto.md)."],
        [("--janela", "c"), "5.0", "Duração da janela de agregação, em segundos."],
        [("--blocos", "c"), "5", "Número de trechos de tempo por corrida no esquema BLOCOS."],
        [("--sem-harmonizar", "c"), "desligado", "Não descarta os hits abaixo do maior limiar das sessões "
                                                 "(65,1/55,0 dBAE). O corte remove só 0,001 % dos hits."],
        [("--esquemas", "c"), ("BLOCOS,LORO,LOSO", "c"), "Esquemas de validação a rodar. O LOSO roda só para a frequência."],
        [("--jobs", "c"), "-1", "Processos paralelos (folds rodados em paralelo com joblib; -1 = todos os núcleos)."],
    ], larguras=[3.4, 3.6, 9.0], legenda="Opções de linha de comando de mlp_ea.py.")
    D.p("Os parâmetros dos modelos (Seção 8), a semente (", ("SEMENTE = 42", "c"), ") e a zona de "
        "exclusão (1 janela) estão no código, e não na linha de comando. Isso é deliberado: são "
        "decisões congeladas, que não devem ser ajustadas olhando o teste.")

    # 4 ------------------------------------------------------------------
    D.h("4. Fluxo de processamento e onde cada coisa é ajustada")
    D.tabela(["Etapa", "O que faz", "Ajustada com"], [
        ["1. Leitura", "56 hits.csv + config.json; rótulos do caminho; ordem cronológica pelo campo 'inicio'.", "—"],
        ["2. Limiar", "Descarta hits com amplitude abaixo de 65,1 dBAE (A) ou 55,0 dBAE (B).", "Regra fixa (sem ajuste)"],
        ["3. Janelas", "Janelas consecutivas de 5 s sem sobreposição; 18 descritores por canal (36 no total).",
         "Regra fixa; cada janela usa só os próprios hits"],
        ["4a. Imputação", "Mediana de cada descritor (6,7 % de faltantes).", ("Só o treino do fold", "b")],
        ["4b. Padronização", "Média 0 e desvio 1 por descritor (StandardScaler).", ("Só o treino do fold", "b")],
        ["4c. Alvo (regressão)", "Frequência padronizada para a MLP (TransformedTargetRegressor).", ("Só o treino do fold", "b")],
        ["4d. Modelo", "MLP, RF ou GB com hiperparâmetros fixos.", ("Só o treino do fold", "b")],
        ["4e. Parada antecipada (MLP)", "Escolhe a época pela validação interna.", "15 % das janelas do treino (Seção 7)"],
        ["5. Importância", "Permutação nos blocos de teste do BLOCOS.", "Modelo do fold; avaliado no teste"],
        ["6. k-means / PCA", "Agrupamento sem rótulos, cruzado com os fatores (ARI, NMI).",
         "Todas as janelas (não supervisionado; não entra nas métricas de classificação)"],
    ], larguras=[3.6, 7.4, 5.0], legenda="Etapas do pipeline e conjunto usado para ajustar cada uma.")
    D.p("Imputação, padronização e modelo estão num único ", ("Pipeline", "c"), " do scikit-learn, "
        "clonado e reajustado em cada fold (", ("clone(modelo).fit(X[treino], y[treino])", "c"), "). "
        "Assim, mediana, média e desvio do teste nunca entram no ajuste.")

    # 5 ------------------------------------------------------------------
    D.h("5. Os três conjuntos de dados e por que não usar a divisão aleatória 70/15/15")
    D.p("O texto de referência descreve três conjuntos:")
    D.lista([
        "o treino (70–80 %), em que o modelo aprende;",
        "a validação (~15 %), usada para ajustar hiperparâmetros e decidir a parada antecipada;",
        "o teste (15–30 %), inédito e usado só no fim.",
    ])
    D.p("Ele recomenda independência entre treino e teste, estratificação e, quando há linha do tempo, "
        "treinar com o passado e testar com o futuro. A Tabela 4 mostra como cada papel foi cumprido aqui.")
    D.tabela(["Conjunto", "Texto de referência", "Como foi feito neste estudo"], [
        [("Treino", "b"), "70–80 % dos dados; o modelo aprende os padrões.",
         f"BLOCOS: {pct(folds['BLOCOS']['tr'][2])} das janelas (4 de 5 trechos de cada corrida). "
         f"LORO: {pct(folds['LORO']['tr'][2])} (55 corridas). LOSO: {pct(folds['LOSO']['tr'][2])} (6 séries)."],
        [("Validação", "b"), "~15 %; ajuste de hiperparâmetros e parada antecipada.",
         "Hiperparâmetros fixos, sem busca, portanto sem validação para ajuste. Só a MLP tem validação "
         "interna: 15 % das janelas de treino, sorteadas, para a parada antecipada (Seção 7)."],
        [("Teste", "b"), "15–30 %; inédito; avaliação final.",
         f"BLOCOS: {pct(folds['BLOCOS']['te'][2])} (1 trecho de cada corrida). LORO: {pct(folds['LORO']['te'][2], 2)} "
         f"(1 corrida). LOSO: {pct(folds['LOSO']['te'][2])} (1 série). Cada janela é testada uma vez; "
         "as métricas juntam todos os folds."],
        [("Independência", "b"), "Treino e teste sem sobreposição.",
         "Unidade de independência = corrida. LORO e LOSO garantem corridas inéditas; BLOCOS só garante "
         "trechos de tempo inéditos, com 1 janela de exclusão."],
        [("Estratificação", "b"), "Mesma proporção de classes nos conjuntos.",
         "BLOCOS: todas as corridas estão em todos os folds (estratificado por construção). LORO/LOSO: o "
         "teste é uma corrida ou série, sem estratificação possível. Validação interna da MLP de "
         "válvulas: estratificada por classe; a da regressão, não."],
        [("Linha do tempo", "b"), "Treinar no passado, testar no futuro.",
         "Não respeitada nos esquemas principais (o treino inclui corridas posteriores ao teste). O teste "
         "fora do tempo entre dias (Seção 11) é o único que segue essa regra."],
    ], larguras=[2.6, 4.6, 8.8], legenda="Os três conjuntos do texto de referência e sua realização neste estudo.")
    D.h("5.1 Por que não sortear janelas", 2)
    D.p("A divisão aleatória supõe que as amostras são independentes e vêm da mesma distribuição. "
        "Aqui, nenhuma das duas hipóteses vale:")
    D.lista([
        [("Correlação dentro da corrida. ", "b"),
         f"Cada corrida tem cerca de 63 janelas de 5 s ({N} no total) medidas na mesma condição, com o mesmo "
         "acoplamento, a mesma temperatura e o mesmo limiar. Janelas vizinhas são quase réplicas. Num "
         "sorteio, cada janela de teste tem vizinhas no treino, e o modelo é avaliado em reconhecer a corrida, "
         "não a condição (Figura 1)."],
        [("Deriva e ordem fixa. ", "b"),
         "Cada série de válvulas foi ensaiada em bloco contínuo, num dia e horário próprios, e com limiar "
         "próprio. Qualquer deriva lenta (aquecimento, acoplamento, ganho) fica associada à série e, "
         "portanto, à classe das válvulas."],
        [("Poucas unidades independentes. ", "b"),
         "Há 56 corridas, uma por condição. Para a pergunta \"o modelo reconhece uma condição?\", o tamanho "
         "amostral efetivo é 56, e não 3 540."],
    ])
    blo = tot.loc[("BLOCOS", "config", "RF"), "acc"]
    lor = tot.loc[("LORO", "config", "RF"), "acc"]
    D.p("O efeito é medido diretamente. Para as válvulas, a random forest acerta ", (pct(blo, 0), "b"),
        " quando a corrida aparece no treino (BLOCOS) e ", (pct(lor, 0), "b"),
        " quando a corrida inteira fica de fora (LORO; acaso = 14 %). Um sorteio de janelas daria um número "
        "ainda mais próximo do primeiro, porque nem a exclusão entre blocos existiria.")
    D.figura("fig_divisao_aleatoria.png",
             "Divisão aleatória 70/15/15 por janela (não usada). Cada célula é uma corrida, e as faixas "
             "verticais são as janelas de 5 s em ordem de tempo. Treino, validação e teste ficam intercalados "
             "dentro da mesma corrida.")

    # 6 ------------------------------------------------------------------
    D.h("6. Esquemas de divisão por fold")
    D.p("Os três esquemas estão implementados em ", ("gerar_folds()", "c"), " (mlp_ea.py). Os números "
        "abaixo foram calculados sobre as 3 540 janelas da execução principal.")
    b, l, s = folds["BLOCOS"], folds["LORO"], folds["LOSO"]
    D.tabela(["", "BLOCOS", "LORO", "LOSO"], [
        [("Pergunta", "b"), "Reconhece outro trecho de tempo de uma corrida já vista?",
         "Reconhece uma condição (corrida) nunca vista?", "Reconhece a frequência numa sessão nunca vista?"],
        [("Folds", "b"), f"{b['n']} (um por trecho de ~60 s)", f"{l['n']} (um por corrida)", f"{s['n']} (um por série de válvulas)"],
        [("Teste", "b"), f"k-ésimo trecho de todas as 56 corridas: {b['te'][0]}–{b['te'][1]} janelas ({pct(b['te'][2])})",
         f"1 corrida inteira: {l['te'][0]}–{l['te'][1]} janelas ({pct(l['te'][2], 2)})",
         f"1 série inteira (8 corridas): {s['te'][0]}–{s['te'][1]} janelas ({pct(s['te'][2])})"],
        [("Treino", "b"), f"Outros 4 trechos de todas as corridas, menos a exclusão: {b['tr'][0]}–{b['tr'][1]} janelas",
         f"55 corridas: {l['tr'][0]}–{l['tr'][1]} janelas", f"6 séries (48 corridas): {s['tr'][0]}–{s['tr'][1]} janelas"],
        [("Exclusão", "b"), f"1 janela de cada lado do trecho de teste, na mesma corrida ({b['ex'].min()}–{b['ex'].max()} janelas)",
         "Nenhuma", "Nenhuma"],
        [("Validação interna (MLP)", "b"), "15 % do treino, sorteado", "15 % do treino, sorteado", "15 % do treino, sorteado"],
        [("Alvos", "b"), "Frequência e válvulas", "Frequência e válvulas",
         "Só a frequência (a classe de válvula testada estaria ausente do treino)"],
        [("Leitura", "b"), "Otimista: a corrida está no treino", "Estimativa honesta de generalização",
         "Teste mais duro contra a deriva entre sessões"],
    ], larguras=[3.0, 4.4, 4.3, 4.3], legenda="Composição dos folds em cada esquema (janelas de 5 s).", tam=8.5)
    D.h("6.1 BLOCOS (pseudo-repetição)", 2)
    D.p("Como não há repetições das condições, o BLOCOS cria uma \"repetição\" artificial. Cada corrida "
        "é dividida em 5 trechos consecutivos, e o trecho k de todas as corridas forma o teste do fold k. "
        "Todas as classes aparecem no teste de todos os folds. A exclusão de uma janela (5 s) de cada lado "
        "do trecho de teste evita que janelas contíguas, quase idênticas, fiquem uma no treino e outra no "
        "teste. Nos folds 1 e 5, a exclusão só existe de um lado. A limitação é estrutural: a corrida "
        "continua no treino, com seu estado de acoplamento, sua temperatura e seu limiar.")
    D.figura("fig_divisao_blocos.png", "BLOCOS, fold 3 de 5. Vermelho: teste; cinza: exclusão; azul: treino; "
                                        "amarelo: validação interna da MLP, sorteada dentro do treino.")
    D.h("6.2 LORO (deixa uma corrida de fora)", 2)
    D.p("No LORO, cada uma das 56 corridas é, uma vez, o teste inteiro. Graças ao delineamento cruzado, a "
        "frequência da corrida testada aparece no treino com as outras 6 configurações, e a configuração "
        "aparece com as outras 7 frequências. Os dois alvos são, portanto, aprendíveis sem que a corrida "
        "testada tenha sido vista. Não há janela de exclusão, porque treino e teste não compartilham corrida. "
        "As corridas vizinhas no tempo (anterior e posterior da mesma série), porém, ficam no treino. Uma "
        "variante mais rigorosa as excluiria (Seção 10).")
    D.figura("fig_divisao_loro.png", "LORO, fold da corrida 100-50 a 60 Hz.")
    D.h("6.3 LOSO (deixa uma série de fora)", 2)
    D.p("No LOSO, a série de válvulas inteira, isto é, uma sessão contínua de ~45 min, com seu próprio "
        "horário e limiar, é o teste. Só se avalia a frequência, porque a classe de válvula testada não "
        "existiria no treino.")
    D.figura("fig_divisao_loso.png", "LOSO, fold da série 100-25 (06/11, 09h49–10h32).")

    # 7 ------------------------------------------------------------------
    D.h("7. Validação interna: parada antecipada da MLP")
    D.p("A MLP é a única que usa um conjunto de validação. A configuração em mlp_ea.py é ",
        ("early_stopping=True, validation_fraction=0.15, n_iter_no_change=30, random_state=42", "c"),
        ". No scikit-learn 1.9.1 (", ("sklearn/neural_network/_multilayer_perceptron.py", "c"),
        ", método _fit), isso faz o seguinte:", alinhar=False)
    D.lista([
        [("De onde vêm os dados: ", "b"), "a cada ajuste (cada fold), ", ("train_test_split", "c"),
         " separa aleatoriamente 15 % das linhas recebidas pelo modelo, ou seja, 15 % das janelas do treino "
         "daquele fold, com embaralhamento e semente fixa."],
        [("Estratificação: ", "b"), "no MLPClassifier (válvulas), o sorteio é estratificado pela classe. No "
         "MLPRegressor (frequência), não."],
        [("Critério: ", "b"), "a cada época, mede-se o escore nessa validação (acurácia na classificação, R² "
         "na regressão). O treino para após 30 épocas sem melhora maior que tol = 1e-4, e os pesos da melhor "
         "época são restaurados."],
        [("O que a validação não vê: ", "b"), "as janelas de teste do fold. Elas nunca entram no ajuste, nem "
         "na parada antecipada."],
        [("Pré-processamento: ", "b"), "imputação e padronização são ajustadas no treino do fold inteiro, "
         "incluindo os 15 % de validação interna, porque ficam antes da MLP no pipeline. O alvo da regressão "
         "também é padronizado no treino inteiro."],
    ])
    D.h("7.1 Isso cria vazamento?", 2)
    D.p(("Para o teste, não. ", "b"), "O conjunto de teste é sempre externo ao fold, e as métricas reportadas "
        "não são afetadas por vazamento da validação interna. ", ("Dentro do treino, sim: ", "b"),
        "as janelas de validação interna são sorteadas das mesmas corridas que as de treino efetivo, "
        "intercaladas com elas (Figura 5). A validação interna, portanto, mede a memorização da corrida e é "
        "otimista. A parada antecipada escolhe a época que melhor reconhece corridas já vistas, e não a que "
        "melhor generaliza para corridas novas. O efeito foi medido em ", ("figuras_artigo.py", "c"),
        " (Tabela 6 e Figura 6).")
    D.figura("fig_validacao_interna_esquema.png",
             "Validação interna atual (acima): janelas sorteadas, misturadas às vizinhas da mesma corrida. "
             "Recomendada (abaixo): corridas inteiras de validação.")
    linhas = []
    for _, r in vi.iterrows():
        linhas.append([r.esquema, "válvulas" if r.tarefa == "config" else "frequência", r.metrica,
                       f2(r.interna, 3), "indefinido*" if pd.isna(r.teste) else f2(r.teste, 3),
                       "—" if pd.isna(r.MAE_teste) else f2(r.MAE_teste, 1) + " Hz", f2(r.epocas, 0)])
    D.tabela(["Esquema", "Alvo", "Métrica", "Validação interna", "Teste do fold", "MAE teste", "Épocas"],
             linhas, larguras=[2.2, 2.2, 2.0, 2.6, 2.4, 2.2, 1.8],
             legenda="MLP: escore médio da validação interna × escore no teste do mesmo fold "
                     "(BLOCOS: 5 folds; LORO: 14 dos 56 folds).")
    D.p("* No LORO, cada fold de teste contém uma única frequência, e o R² fica indefinido; vale o MAE. "
        "Para as válvulas, a validação interna indica cerca de 90 % de acurácia nos folds LORO, enquanto o "
        "teste dá 18 %. A diferença mostra que a validação interna sorteada não serve para estimar o "
        "desempenho em corridas novas.", alinhar=True)
    D.figura("fig_validacao_interna_vs_teste.png", "Escore da validação interna × escore no teste, por fold.")
    D.p(("Recomendação. ", "b"), "Fazer a validação interna por corrida. Por exemplo, separar 15 % das "
        "corridas de treino com GroupShuffleSplit (grupos = corrida) e usar o número de épocas encontrado "
        "nelas. Como o early_stopping do scikit-learn não aceita grupos, uma forma simples é:")
    D.codigo(r"""
from sklearn.model_selection import GroupShuffleSplit
gss = GroupShuffleSplit(n_splits=1, test_size=0.15, random_state=42)
i_fit, i_val = next(gss.split(X_tr, y_tr, groups=corrida_tr))
pre = make_pipeline(SimpleImputer(strategy="median"), StandardScaler()).fit(X_tr[i_fit])
mlp = MLPClassifier(hidden_layer_sizes=(64, 32), alpha=1e-3, random_state=42)
melhor, paciencia, classes = (-1, None), 0, np.unique(y_tr)
for epoca in range(2000):
    mlp.partial_fit(pre.transform(X_tr[i_fit]), y_tr[i_fit], classes=classes)
    s = mlp.score(pre.transform(X_tr[i_val]), y_tr[i_val])
    if s > melhor[0] + 1e-4: melhor, paciencia = (s, epoca), 0
    else: paciencia += 1
    if paciencia >= 30: break
# reajustar no treino inteiro com max_iter = melhor[1] + 1 e early_stopping=False
""")

    # 8 ------------------------------------------------------------------
    D.h("8. Hiperparâmetros fixos")
    D.p("Nenhum hiperparâmetro foi buscado. Os valores abaixo foram escolhidos antes de ver os resultados e "
        "valem para todos os folds e esquemas. Não existe, portanto, um conjunto de validação para "
        "ajuste no sentido do texto de referência. A consequência é que os números são linhas de base, sem "
        "otimismo por seleção de modelo.")
    D.tabela(["Modelo", "Frequência (regressão)", "Válvulas (7 classes)"], [
        [("MLP", "b"), "MLPRegressor (64, 32), ReLU, Adam, alpha = 1e-3, lr = 1e-3, early stopping (15 %, "
                       "paciência 30), max_iter = 2000; alvo padronizado",
         "MLPClassifier (64, 32), ReLU, Adam, alpha = 1e-3, lr = 1e-3, early stopping (15 %, paciência 30), "
         "max_iter = 2000"],
        [("RF", "b"), "RandomForestRegressor: 300 árvores, min_samples_leaf = 2, max_features = 0,5",
         "RandomForestClassifier: 300 árvores, min_samples_leaf = 2, max_features = sqrt"],
        [("GB", "b"), "HistGradientBoostingRegressor: max_iter = 300, learning_rate = 0,05, sem early stopping",
         "HistGradientBoostingClassifier: max_iter = 200, learning_rate = 0,05, sem early stopping"],
        [("Comum", "b"), "SimpleImputer(median) + StandardScaler; random_state = 42",
         "SimpleImputer(median) + StandardScaler; random_state = 42"],
    ], larguras=[1.8, 7.1, 7.1], legenda="Hiperparâmetros (definidos em modelos(), mlp_ea.py).")
    D.p("Se for necessário buscar hiperparâmetros, a busca deve ser aninhada e agrupada por corrida. No "
        "laço externo, o LORO (ou o LOSO) dá o teste. No laço interno, um GroupKFold sobre as corridas de "
        "treino dá a validação. Assim, o teste continua inédito para a seleção. Com 56 corridas, essa "
        "busca tem pouca resolução, e o ganho esperado é pequeno diante das limitações dos dados.")
    D.codigo(r"""
from sklearn.model_selection import GridSearchCV, GroupKFold
busca = GridSearchCV(pipeline, {"mlpclassifier__alpha": [1e-4, 1e-3, 1e-2]},
                     cv=GroupKFold(n_splits=5), scoring="balanced_accuracy")
busca.fit(X[treino], y[treino], groups=corrida[treino])   # dentro de cada fold externo
""")

    # 9 ------------------------------------------------------------------
    D.h("9. Imputação e padronização")
    D.lista([
        [("Faltantes: ", "b"), "os descritores de forma usam só os hits sem MAXDUR. Uma janela em que todos "
         "os hits de um canal têm MAXDUR fica sem esses descritores (6,7 % dos valores)."],
        [("Imputação: ", "b"), "mediana de cada descritor, calculada só no treino do fold e aplicada ao teste."],
        [("Padronização: ", "b"), "média e desvio de cada descritor no treino do fold. As árvores (RF, GB) "
         "não precisam dela, mas o mesmo pré-processamento foi mantido nos três modelos para comparação."],
        [("Descritores constantes ", "b"), "(sem variação em nenhuma janela) são removidos antes da validação. "
         "Essa decisão não depende dos rótulos."],
        [("O que não é ajustado no treino: ", "b"), "a construção dos descritores de cada janela, que só "
         "usa os hits da própria janela, e o corte de limiar, que é uma regra fixa. Nenhum dos dois usa "
         "informação de outras janelas ou dos rótulos."],
    ])

    # 10 -----------------------------------------------------------------
    D.h("10. Janela de exclusão entre treino e teste")
    D.p("No BLOCOS, ", ("gerar_folds()", "c"), " retira do treino, em cada corrida, a janela imediatamente "
        "anterior e a imediatamente posterior ao trecho de teste. Cada uma tem 5 s. A exclusão total por "
        f"fold é de {b['ex'].min()} a {b['ex'].max()} janelas: 56 nos folds das pontas, em que só há "
        "vizinho de um lado, e 112 nos demais. O objetivo é que nenhuma janela de treino seja contígua a "
        "uma de teste. Uma janela de 5 s é pouco diante de uma deriva de minutos: a exclusão reduz a "
        "dependência de curto prazo, mas não torna o BLOCOS independente.")
    D.p("No LORO e no LOSO não há exclusão dentro da corrida, porque o teste é uma corrida ou série "
        "inteira. Duas variantes mais rigorosas, ainda não implementadas, seriam:")
    D.lista([
        "no LORO, excluir do treino também as corridas imediatamente anterior e posterior na ordem do "
        "ensaio, que são as mais próximas em deriva;",
        "no BLOCOS, aumentar a exclusão para 2 ou 3 janelas e verificar se o desempenho cai.",
    ])

    # 11 -----------------------------------------------------------------
    D.h("11. O quarto conjunto")
    D.p("O texto de referência sugere um quarto conjunto opcional, em quatro variantes. A Tabela 8 avalia "
        "cada uma neste estudo.")
    D.tabela(["Variante", "Faz sentido aqui?", "Como implementar"], [
        [("Holdout final bloqueado", "b"),
         "Sim, é a mais importante. Hoje o LORO é ao mesmo tempo validação e teste final: qualquer ajuste "
         "feito olhando o LORO o contamina.",
         "Reservar uma campanha futura com repetições, em ordem aleatorizada. Congelar o pipeline (janela, "
         "descritores, modelos, hiperparâmetros e versão do código, registrando o hash do commit) antes de "
         "abrir esses dados. Avaliar uma única vez."],
        [("OOT (fora do tempo)", "b"),
         "Sim, para a frequência. Para as válvulas, não com os dados atuais: as configurações de cada dia "
         "são diferentes, e a classe testada não existe no treino.",
         "Agora: treino com as séries de 05/11 e teste nas de 06/11 (feito em figuras_artigo.py). Futuro: "
         "a nova campanha é, naturalmente, o conjunto OOT."],
        [("Calibração de probabilidades", "b"),
         "Ainda não. Os modelos são usados por classe prevista, sem limiar de decisão sobre a probabilidade, "
         "e as probabilidades das válvulas não generalizam (LORO perto do acaso).",
         "Quando houver alarme por probabilidade, por exemplo de cavitação: CalibratedClassifierCV com "
         "folds agrupados por corrida, num conjunto separado do teste."],
        [("Viés/equidade", "b"),
         "Não no sentido de grupos de pessoas. O análogo útil é o desempenho por subgrupo: por frequência, "
         "por série, por canal e por dia.",
         "Já reportado: MAE por corrida (mapa LORO) e por série (LOSO). Acrescentar o desempenho por canal "
         "(só A ou só B) e por limiar quando houver repetições."],
    ], larguras=[3.2, 6.0, 6.8], legenda="Variantes do quarto conjunto aplicadas a este estudo.", tam=8.5)
    D.figura("fig_divisao_oot.png", "Holdout fora do tempo usado como ilustração: treino com as séries de "
                                     "05/11 e teste nas de 06/11 (só frequência).")
    linhas = [[f"{pd.Timestamp(r.treino):%d/%m} → {pd.Timestamp(r.teste):%d/%m}", r.modelo, f2(r.MAE_Hz, 2),
               pct(r.acc_nivel), pct(r.acc_10Hz), f2(r.MAE_Hz_corrida, 2)] for _, r in oot.iterrows()]
    D.tabela(["Treino → teste", "Modelo", "MAE (Hz)", "Acerto do nível", "Acerto ±10 Hz", "MAE por corrida (Hz)"],
             linhas, larguras=[3.0, 1.8, 2.0, 2.8, 2.8, 3.6], legenda="Teste fora do tempo entre dias (frequência).")
    D.p("O MAE fora do tempo, de 4,0 a 6,6 Hz, está na mesma faixa do LOSO (4,2 a 5,1 Hz). A estimativa da "
        "frequência resiste à troca de dia. Há, porém, uma ressalva: a troca de dia coincide com a troca de "
        "limiar (65/55 → 50/50 dBAE), de modo que o OOT testa os dois efeitos ao mesmo tempo.")
    D.figura("fig_oot_por_dia.png", "Frequência prevista × real no teste entre dias.")
    D.p(("Proposta para a próxima campanha: ", "b"),
        "pelo menos 3 repetições de cada condição, em ordem aleatorizada por dia, com ganho e limiar fixos. "
        "A divisão passaria a ser:")
    D.lista([
        [("treino e validação: ", "b"), "as repetições 1 e 2, com validação interna por repetição;"],
        [("teste: ", "b"), "a repetição 3, deixada de fora, o que é a validação originalmente pretendida;"],
        [("quarto conjunto: ", "b"), "uma campanha posterior, bloqueada, fora do tempo, avaliada uma única vez."],
    ])

    # 12 -----------------------------------------------------------------
    D.h("12. Saídas")
    D.tabela(["Arquivo (resultados/)", "Conteúdo"], [
        [("metricas.csv", "c"), "Métricas por esquema, alvo, modelo e fold, com linha TOTAL."],
        [("predicoes.csv", "c"), "Previsão de cada janela em cada esquema, com o fold de origem (permite "
                                 "refazer qualquer métrica)."],
        [("janelas_descritores.csv", "c"), "Descritores de cada janela, com corrida, bloco, dia e limiar."],
        [("corridas.csv", "c"), "Metadados das 56 corridas (rótulos, início, limiar, ganho, número de hits)."],
        [("importancias.csv", "c"), "Importância por permutação (BLOCOS)."],
        [("nao_supervisionado_ARI_NMI.csv, contingencia_k*.csv", "c"), "k-means × fatores."],
        [("relatorio_auto.md", "c"), "Tabelas-resumo geradas pelo script."],
        [("fig_*.png", "c"), "Figuras; as complementares estão em artigo/figuras/."],
    ], larguras=[6.0, 10.0], legenda="Arquivos gerados.")

    # 13 -----------------------------------------------------------------
    D.h("13. Cuidados de uso")
    D.lista([
        "Reportar sempre o LORO, ou um esquema mais estrito, como a estimativa de desempenho. O BLOCOS "
        "serve apenas como referência otimista.",
        "Não ajustar janela, descritores ou hiperparâmetros olhando o LORO sem reservar outro teste "
        "(Seção 11).",
        "Conferir os nomes das pastas: as subpastas de 100REC_50SUC estão rotuladas REC100-25SUC. O script "
        "usa a pasta de 1º nível e emite um aviso.",
        "Com dados novos que tenham repetições, acrescentar ao rótulo um identificador de repetição e "
        "criar em gerar_folds() o esquema \"deixa uma repetição de fora\".",
        "ROVL = 1 em todos os hits e a cadência de ~10 hits/s por canal decorrem do rearme do detector com "
        "emissão contínua. A taxa de hits não deve ser interpretada como atividade acústica.",
    ])

    D.salvar(SAIDA)
    print("OK ->", SAIDA)


if __name__ == "__main__":
    main()
