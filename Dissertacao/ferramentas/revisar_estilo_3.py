"""Terceira passada: números de seção fixos do artigo original e frases provisórias."""
from pathlib import Path

D = Path(__file__).resolve().parent.parent / "capitulos" / "importado"
R = {
    "v2_decisoes.tex": [
        ("Ensaio executado em 15/10/2026 (Seção 6); valor preliminar", "Ensaio executado em 15/10/2026 (Seção~\\ref{sec:res_extrator}); valor preliminar"),
        ("exige instrumento com estatística de intervalos (Seção 6)", "exige instrumento com estatística de intervalos (Seção~\\ref{sec:res_extrator})"),
        ("Algumas informações básicas são: FPGA Artix-7 XC7A35TICSG324-1L, encapsulamento CSG324, grau de velocidade -1 (variante -1L, baixa tensão/industrial). E, além disso -- quando da pós-implementação -- serão discutidos os resultados da análise temporal (WNS, TNS), conforme já apontado na Tabela~4 da Seção~\\ref{sec:mm_implementacao}).",
         "O dispositivo é o Artix-7 XC7A35TICSG324-1L, encapsulamento CSG324, grau de velocidade -1L (baixa tensão, faixa industrial). A utilização e a análise temporal da versão implementada são apresentadas na Seção~\\ref{sec:mm_implementacao} (Tabela~\\ref{v2:tab:utilizacao_pos_implementacao})."),
        ("A duração de aquisição foi de 12 horas continuas, com uma taxa de amostragem de 8MS/s, compressão LZMA, reportando 0/N (nenhuma perda em N registros/segundos). As limitações observadas devem, portanto, ser acompanhadas por medições de vazão sustentada, ocupação dos buffers e perdas, evitando atribuí-las apenas à taxa nominal da interface.",
         "A aquisição durou 12 horas contínuas, a 8\\,MSa/s, com compressão LZMA e sem perdas registradas pelos contadores. Assim, as limitações da arquitetura foram avaliadas por medições de vazão sustentada, ocupação dos \\textit{buffers} e perdas, e não apenas pela taxa nominal da interface."),
    ],
    "v2_hits_fundamentos.tex": [
        ("para a comparação experimental das Seções~5 e~6.",
         "para a comparação experimental das Seções~\\ref{sec:mm_delineamento} e~\\ref{sec:res_extrator}."),
    ],
    "v2_implementacao.tex": [
        ("O protótipo avaliado nos ensaios da Seção~6 foi construído", "O protótipo avaliado nos ensaios da Seção~\\ref{sec:res_extrator} foi construído"),
    ],
}
for arq, pares in R.items():
    f = D / arq
    t = f.read_text(encoding="utf-8")
    for a, b in pares:
        if a in t:
            t = t.replace(a, b, 1)
        else:
            print(f"[não encontrado] {arq}: {a[:70]}")
    f.write_text(t, encoding="utf-8")
print("terceira passada aplicada")
