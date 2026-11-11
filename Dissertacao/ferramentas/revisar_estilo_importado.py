"""Converte frases de relatório (prescritivas: 'deve ser informado', 'recomenda-se', 'o artigo',
'estudo complementar') em texto de dissertação. Substituições exatas; avisa se algum trecho
não for encontrado (por exemplo, se já tiver sido revisado)."""
from pathlib import Path

D = Path(__file__).resolve().parent.parent / "capitulos" / "importado"
R = {
    "sim_metodos.tex": [
        ("O estudo complementar agregou", "O estudo da Seção~\\ref{sec:mm_bomba} agregou"),
        ("As duas redes são as do estudo complementar,", "As duas redes são as da Seção~\\ref{sec:mm_bomba},"),
        ("O desempenho fora da amostra é o do estudo complementar e não muda com o retreino.",
         "O desempenho fora da amostra é o da Seção~\\ref{sec:res_bomba} e não muda com o retreino."),
    ],
    "sim_resultados.tex": [
        ("dentro e fora da corrida no estudo complementar", "dentro e fora da corrida (Seção~\\ref{sec:res_bomba})"),
        ("reproduzem os do estudo complementar;", "reproduzem os da Seção~\\ref{sec:mm_bomba};"),
        ("No estudo complementar, a frequência generalizou", "Na Seção~\\ref{sec:res_bomba}, a frequência generalizou"),
    ],
    "v2_decisoes.tex": [
        ("Para a versão do sistema efetivamente submetida aos ensaios, devem ser informados os valores quantitativos do número de canais e da taxa de amostragem. Também devem ser especificados a maior taxa sustentada de \\textit{hits}, a duração da rajada, a latência admissível, o erro máximo e o custo-alvo.",
         "Na versão submetida aos ensaios, o sistema tem dois canais a 10\\,MSa/s e sustentou até 333 \\textit{hits}/s por canal sem perda; os demais valores quantitativos são apresentados nas Seções~\\ref{sec:mm_implementacao} e~\\ref{sec:res_extrator}."),
        ("Ele deve ser apresentado como extensão futura, salvo se tiver sido implementado e avaliado na versão ensaiada.",
         "Neste trabalho, esse modo não foi implementado e é tratado como extensão futura."),
        ("Para cada domínio de relógio, devem ser apresentadas a frequência-alvo, a pior folga temporal, a folga negativa total, os caminhos não restringidos e as exceções utilizadas. A margem de recursos deve ser avaliada",
         "A margem de recursos precisa ser avaliada"),
        ("Portanto, devem ser informados o modo efetivamente negociado e a vazão útil medida.",
         "Nos ensaios, o enlace operou no modo de 100\\,Mb/s."),
        ("não devem ser apresentadas como equivalentes", "não são equivalentes"),
        ("O artigo adota, nesta etapa, o estímulo elétrico controlado. Logo, suas conclusões devem ser limitadas",
         "Este trabalho adota, nesta etapa, o estímulo elétrico controlado. Logo, suas conclusões limitam-se"),
        ("Recomenda-se aplicar a mesma forma de onda simultaneamente às duas cadeias. O divisor ou distribuidor deve ter atenuação, largura de banda, isolamento e impedâncias caracterizados. Se os ensaios forem sequenciais, a repetibilidade e a deriva do gerador devem integrar a avaliação de incerteza. Os instrumentos utilizados para medição é um gerador do sinal e um osciloscópio digital (medição de referência da forma de onda — nível 2 da hierarquia de referência): Minipa, linha MVB DSO, 100 MHz. Suas características principais na tabela~\\ref{v2:tab:instrumentos}.",
         "A mesma forma de onda é aplicada simultaneamente às duas cadeias, por meio de um divisor cujas atenuação, largura de banda, isolamento e impedâncias precisam ser caracterizados; em ensaios sequenciais, a repetibilidade e a deriva do gerador entram na avaliação de incerteza. Os instrumentos são um gerador de sinais e um osciloscópio digital Minipa MVB DSO de 100\\,MHz, usado como referência da forma de onda (nível 2 da hierarquia de referências), com as características da Tabela~\\ref{v2:tab:instrumentos}."),
        ("O erro relativo não deve ser utilizado para referências nulas ou muito pequenas; nesses casos, deve-se reportar o erro absoluto ou o desvio em unidades de resolução (LSB ou equivalente). Em ensaios repetidos, devem ser reportados viés, dispersão e intervalos de incerteza, além das taxas de detecção, perda e fragmentação. A avaliação de incerteza deve considerar",
         "O erro relativo não é usado para referências nulas ou muito pequenas; nesses casos, usa-se o erro absoluto ou o desvio em unidades de resolução (LSB ou equivalente). Em ensaios repetidos, reportam-se viés, dispersão e intervalos de incerteza, além das taxas de detecção, perda e fragmentação. A avaliação de incerteza considera"),
    ],
    "v2_implementacao.tex": [
        ("não deve empregar apenas o produto dos ganhos nominais. Deve ser utilizado um ganho efetivo",
         "não emprega apenas o produto dos ganhos nominais, mas um ganho efetivo"),
        ("não deve ser confundida automaticamente com a duração reportada", "não se confunde automaticamente com a duração reportada"),
        ("a redução global deve ser medida no mesmo intervalo", "a redução global é calculada no mesmo intervalo"),
        ("(a confirmar pós-implementação)", "(valores de síntese)"),
        ("Recomenda-se decompor a utilização por interface do ADC, filtros, máquina de estados, acumuladores, DMA, MicroBlaze, memórias, periféricos e comunicação. A decomposição por bloco funcional, extraída da hierarquia do projeto, é a seguinte:",
         "A utilização pode ser decomposta por bloco funcional, conforme a hierarquia do projeto:"),
        ("Os scripts e exemplos de configuração devem acompanhar o artigo como material suplementar ou ser depositados em repositório permanente, quando possível.",
         "Os \\textit{scripts} e os exemplos de configuração estão nos repositórios do projeto."),
        ("A análise principal deve utilizar o relatório pós-implementação do \\textit{bitstream} empregado nos experimentos. Valores pós-síntese podem ser apresentados como informação intermediária, mas não substituem a utilização e a temporização após posicionamento e roteamento.",
         "A utilização e a temporização definitivas são as do relatório pós-implementação; os valores pós-síntese servem como informação intermediária."),
        ("Os valores apresentados na Tabela~\\ref{v2:tab:utilizacao_pos_implementacao} são provenientes do relatório pós-síntese e devem ser confirmados com o relatório pós-implementação final.",
         "Os valores da Tabela~\\ref{v2:tab:utilizacao_pos_implementacao} vêm do relatório pós-síntese da versão anterior à migração da FIFO para BRAM; a situação da versão ensaiada é descrita logo após a tabela."),
        ("Para cada domínio de relógio, devem ser informados frequência-alvo, período imposto, WNS, TNS, caminhos não restringidos e exceções temporais. Uma frequência máxima não deve ser inferida",
         "Uma frequência máxima não pode ser inferida"),
        ("deve ser medido entre pontos de referência explicitamente definidos", "é medido entre pontos de referência explicitamente definidos"),
        ("Para medir desde o primeiro cruzamento, também deve ser incluída a duração do \\textit{hit}.",
         "Para medir desde o primeiro cruzamento, inclui-se também a duração do \\textit{hit}."),
    ],
    "v2_resultados.tex": [
        ("Para cada descritor, recomenda-se apresentar:", "Para cada descritor, apresentam-se:"),
        ("Recomenda-se separar:", "Distinguem-se:"),
        ("Assim, recomenda-se apresentar custo de materiais por canal e escopo funcional, sem transformar o artigo em comparação promocional nem inferir relação custo--desempenho a partir de bases incompletas.",
         "Assim, apresentam-se o custo de materiais por canal e o escopo funcional, sem inferir uma relação custo--desempenho a partir de bases incompletas."),
    ],
}
for arq, pares in R.items():
    f = D / arq
    t = f.read_text(encoding="utf-8")
    for a, b in pares:
        if a in t:
            t = t.replace(a, b)
        else:
            print(f"[não encontrado] {arq}: {a[:70]}...")
    f.write_text(t, encoding="utf-8")
print("revisão de estilo aplicada")
