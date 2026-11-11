"""Aplica as anotações da banca ao capítulo de FPGA (03-FPGA.tex)."""
from pathlib import Path

f = Path(__file__).resolve().parent.parent / "capitulos" / "FundamentacaoTeorica" / "03-FPGA.tex"
t = f.read_text(encoding="utf-8")

# 1) figura do fluxo de desenvolvimento: tirar do fim e pôr no início da subseção
fig_ini = t.find("O processo de desenvolvimento e otimização de projetos em FPGA pode ser sintetizado na Figura~\\ref{fig:FluxoDesenvolvimentoFPGA}")
fig_fim = t.find("\\end{figure}", t.find("\\label{fig:FluxoDesenvolvimentoFPGA}")) + len("\\end{figure}")
if fig_ini > 0 and fig_fim > fig_ini:
    t = t[:fig_ini] + t[fig_fim:]
    bloco = (
        "\\subsection*{Fluxo de Desenvolvimento com FPGA}\n\n"
        "A Figura~\\ref{fig:FluxoDesenvolvimentoFPGA} resume o fluxo de desenvolvimento e otimização de projetos em FPGA, "
        "da definição do algoritmo até a validação em \\textit{hardware}, e evidencia seu caráter iterativo: os resultados de "
        "síntese, de implementação e de validação realimentam as etapas anteriores \\cite{cong2011hls,amd_ug1399}. "
        "As etapas são detalhadas a seguir.\n\n"
        "\\begin{figure}[htbp]\n    \\centering\n    \\includegraphics[width=0.8\\linewidth]{figs/FPGA/FluxoOtimização.pdf}\n"
        "    \\caption{Fluxo de desenvolvimento e otimização de projetos em FPGA. Fonte: o autor.}\n"
        "    \\label{fig:FluxoDesenvolvimentoFPGA}\n\\end{figure}\n"
    )
    t = t.replace("\\subsection*{Fluxo de Desenvolvimento com FPGA}\n", bloco, 1)
else:
    print("[aviso] bloco da figura de fluxo não encontrado (já movido?)")

R = [
    ("baseados em Field-Programmable Gate Arrays (FPGAs), dispositivos semicondutores programáveis",
     "baseados em FPGAs, dispositivos semicondutores programáveis"),
    ("Essa característica os diferencia dos \\textit{Application-Specific Integrated Circuits} (ASICs),",
     "Essa característica os diferencia dos circuitos integrados de aplicação específica (\\textit{Application-Specific Integrated Circuits}, ASICs) \\cite{kuon2008fpga,trimberger2015three},"),
    ("o avanço das aplicações em Inteligência Artificial e Aprendizado Profundo  impulsionou",
     "o avanço das aplicações de IA e de DL impulsionou"),
    ("\\textit{Look-Up Table, LU)}", "tabela de consulta (\\textit{Look-Up Table}, LUT)"),
    ("restrições de temporização\\textit{timing})", "restrições de temporização, ou \\textit{timing})"),
    ("\\caption{Imagem ilustrativa de um \\textit{CBL}. Adaptado de \\citeonline{genera_technologies_fpga}}",
     "\\caption{Ilustração de um bloco lógico configurável (CLB). Adaptado de \\citeonline{genera_technologies_fpga}.}"),
    ("\t\\item \\textbf{Blocos Lógicos Configuráveis (CLBs / Logic Elements)}\n\tCada bloco lógico geralmente contém",
     "\t\\item \\textbf{Blocos Lógicos Configuráveis (\\textit{Configurable Logic Blocks}, CLBs)}\n\tCada bloco lógico (Figura~\\ref{fig:FPGA_CBL}) geralmente contém"),
    ("garantindo assim um melhor aproveitamento dos recursos disponíveis.",
     "garantindo assim um melhor aproveitamento dos recursos disponíveis \\cite{Shawahna2019,mouri2025optimizing}."),
    ("sendo uma das bases de muitos fluxos de implement",
     "sendo uma das bases de muitos fluxos de implement"),  # marcador (sem alteração)
]
for a, b in R:
    if a in t:
        t = t.replace(a, b, 1)
    else:
        print("[não encontrado]", a[:80].replace("\n", " "))
# referência na quantização
t = t.replace("permite obter modelos mais compactos com perda limitada de desempenho,",
              "permite obter modelos mais compactos com perda limitada de desempenho \\cite{jacob2018,nagel2021},", 1)
f.write_text(t, encoding="utf-8")
print("03-FPGA revisado")
