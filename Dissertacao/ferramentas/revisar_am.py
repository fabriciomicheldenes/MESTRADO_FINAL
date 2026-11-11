"""Aplica as anotações da banca ao capítulo de Aprendizado de Máquina (02-AprendizadoMaquina.tex):
siglas (AM, RNA, DL, PCA), referências, transição dos paradigmas, fonte da figura,
pseudoalgoritmo como Algoritmo 1 e explicação de J_B e da inicialização."""
from pathlib import Path

f = Path(__file__).resolve().parent.parent / "capitulos" / "FundamentacaoTeorica" / "02-AprendizadoMaquina.tex"
t = f.read_text(encoding="utf-8")
R = [
    # parágrafo de abertura riscado pela banca
    ("Esta seção dedica-se à apresentação dos fundamentos conceituais e matemáticos do Aprendizado de Máquina, com ênfase nas abordagens supervisionadas e profundas aplicadas à análise de sinais. Inicialmente, discorrem-se os paradigmas de aprendizado, os principais algoritmos utilizados e suas respectivas áreas de aplicação. Em seguida, introduz-se o Aprendizado Profundo, e as Redes Neurais Multicamada Unidimensionais, destacando seu papel na extração automática de padrões complexos em sinais acústicos, como os gerados por fenômenos de cavitação e falhas incipientes em equipamentos rotativos.\n",
     ""),
    ("e processamento de grandes volumes de dados.",
     "e processamento de grandes volumes de dados \\cite{goodfellow2016deep}."),
    ("Dentro desse panorama, o Aprendizado de Máquina representa uma subárea fundamental, responsável pelo estudo de algoritmos que aprendem a partir de exemplos.",
     "Dentro desse panorama, o AM é a subárea que estuda algoritmos que aprendem a partir de exemplos \\cite{bishop2006pattern}."),
    ("Entre os métodos mais influentes de Aprendizado de Máquina destacam-se as Redes Neurais Artificiais,",
     "Entre os métodos mais influentes de AM destacam-se as Redes Neurais Artificiais (RNA),"),
    ("Um subconjunto mais avançado e especializado do ML é o Aprendizado Profundo, que emprega",
     "Um subconjunto do AM é o DL, que emprega"),
    ("devido à sua notável capacidade de generalização e robustez frente a grandes volumes de informação.",
     "devido à sua capacidade de generalização quando há grandes volumes de dados \\cite{lecun2015deep,goodfellow2016deep}."),
    ("Nos últimos anos, a combinação entre aprendizado profundo e dispositivos de alto desempenho,",
     "Nos últimos anos, a combinação entre o DL e dispositivos de alto desempenho,"),
    ("otimizar a operação de sistemas industriais complexos.",
     "otimizar a operação de sistemas industriais complexos \\cite{Shawahna2019,liu2023review}."),
    ("Os três paradigmas são sistematizados na Figura~\\ref{fig:AprendizadoDeMaquinaConceitos}, que apresenta uma visão comparativa das principais estratégias de aprendizado e suas aplicações típicas em contextos industriais.",
     "Quanto à forma de aprender, os algoritmos de AM costumam ser agrupados em três paradigmas: supervisionado, não supervisionado e por reforço \\cite{bishop2006pattern,goodfellow2016deep}. A Figura~\\ref{fig:AprendizadoDeMaquinaConceitos} compara esses paradigmas e suas aplicações típicas em contextos industriais."),
    ("\\caption{Paradigmas do aprendizado de máquina e exemplos de aplicação industrial.\\\\Fonte: Em destaque o paradigma utilizado neste trabalha. O autor.}",
     "\\caption{Paradigmas do AM e exemplos de aplicação industrial; em destaque, o paradigma utilizado neste trabalho. Fonte: o autor.}"),
    ("análise de componentes principais (PCA)",
     "análise de componentes principais (\\textit{Principal Component Analysis}, PCA) \\cite{jolliffe2016}"),
    ("ressalta a relevância das técnicas de \\textit{Machine Learning} para",
     "ressalta a relevância das técnicas de AM para"),
    ("ao monitoramento em  com baixa latência", "ao monitoramento com baixa latência"),
    ("As Redes Neurais Multicamadas (\\emph{Multilayer Perceptron}, MLP) constituem um dos\nmodelos mais difundidos de redes neurais artificiais",
     "As redes MLP constituem um dos\nmodelos mais difundidos de RNA"),
    ("cujos parâmetros são estimados a partir de dados rotulados.",
     "cujos parâmetros são estimados a partir de dados rotulados \\cite{rumelhart1986learning,haykin2009neural}."),
    ("recorrentes \\citeonline{haykin2009neural}.", "recorrentes \\cite{haykin2009neural}."),
    ("e limitada \\citeonline{cybenko1989approximation,hornik1991approximation}.",
     "e limitada \\cite{cybenko1989approximation,hornik1991approximation}."),
    ("por meio do algoritmo de retropropagação (\\emph{backpropagation}), que aplica a\nregra da cadeia ao grafo computacional definido pelas camadas da rede.",
     "por meio do algoritmo de retropropagação (\\emph{backpropagation}), que aplica a\nregra da cadeia ao grafo computacional definido pelas camadas da rede \\cite{rumelhart1986learning}."),
    ("computacional e propriedades favoráveis de generalização.",
     "computacional e propriedades favoráveis de generalização \\cite{goodfellow2016deep}."),
    ("O pseudo-algoritmo abaixo descreve o fluxo lógico do treinamento de uma MLP via \\emph{SGD puro}, omitindo detalhes de implementação específicos de bibliotecas, mas preservando todas as etapas conceituais essenciais. \nVariantes populares substituem a etapa de atualização por regras com momentos (Momentum ou Nesterov) ou estimativas adaptativas (Adam), e podem incluir dropout no \\emph{forward}, normalização por camadas e \\(\\ell_2\\) acoplado à atualização (weight decay).",
     "O Algoritmo~\\ref{alg:mlp_sgd_minibatch} descreve o treinamento de uma MLP por gradiente descendente estocástico em mini-lotes (\\textit{mini-batches}), sem detalhes de bibliotecas específicas. Nele, \\(J_{\\mathcal{B}}(\\theta)\\) é a função custo restrita ao mini-lote \\(\\mathcal{B}\\), isto é, a perda média \\(\\frac{1}{|\\mathcal{B}|}\\sum_{i\\in\\mathcal{B}}\\mathcal{L}(f_\\theta(x_i),y_i)\\) acrescida do termo de regularização \\(\\lambda\\,\\Omega(\\theta)\\). Os pesos são inicializados aleatoriamente, com variância ajustada ao número de entradas de cada camada (esquemas de Xavier ou de He), o que evita que as ativações saturem ou se anulem no início do treinamento \\cite{goodfellow2016deep}. Variantes populares substituem a etapa de atualização por regras com momento (\\textit{momentum} ou Nesterov) ou estimativas adaptativas, como o Adam \\cite{kingma2015adam}, e podem incluir \\textit{dropout}, normalização por camadas e o termo \\(\\ell_2\\) acoplado à atualização (\\textit{weight decay})."),
    ("\\subsection*{Pseudo-algoritmo de alto nível: Treinamento de MLP com SGD em mini-batches}\n", ""),
    ("Essa arquitetura permite ao MLP aprender", "Essa arquitetura permite à MLP aprender"),
    ("que introduzem não linearidade no processamento de sinais, \\cite{jaiswal2024multilayer}.",
     "que introduzem não linearidade no processamento \\cite{jaiswal2024multilayer}."),
    ("extração de características, \\cite{abd2023multi}.", "extração de características \\cite{abd2023multi}."),
    ("facilitando o controle de trade-offs entre precisão, consumo de recursos e latência de inferência.",
     "facilitando o controle dos compromissos entre precisão, consumo de recursos e latência de inferência \\cite{Shawahna2019}."),
]
for a, b in R:
    if a in t:
        t = t.replace(a, b, 1)
    else:
        print("[não encontrado]", a[:80].replace("\n", " "))
f.write_text(t, encoding="utf-8")
print("02-AprendizadoMaquina revisado")
