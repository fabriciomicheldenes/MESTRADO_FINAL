"""Segunda passada de estilo nos trechos do relatório do extrator: instruções de relatório e
critérios no imperativo passam a descrever o que foi feito."""
from pathlib import Path

D = Path(__file__).resolve().parent.parent / "capitulos" / "importado"
R = {
    "v2_decisoes.tex": [
        ("O desempenho deve ser avaliado de forma fim a fim, incluindo empacotamento, controlador USB, cópias em memória, escalonamento do sistema operacional, processamento, sistema de arquivos e unidade de armazenamento. a vazão sustentada medida foi de ~28 MB/s,",
         "O desempenho depende da cadeia inteira, incluindo empacotamento, controlador USB, cópias em memória, escalonamento do sistema operacional, processamento, sistema de arquivos e unidade de armazenamento. A vazão sustentada medida foi de cerca de 28\\,MB/s,"),
        ("cuja ocupação máxima atingiu picos de ~5\\%.", "cuja ocupação máxima atingiu picos de cerca de 5\\,\\%."),
        ("A ocorrência e a localização de perdas devem ser demonstradas por contadores, e não inferidas apenas de variações de desempenho.",
         "Por isso, neste trabalho, a ocorrência e a localização de perdas são demonstradas por contadores, e não inferidas de variações de desempenho."),
        ("Além da taxa média, deve-se analisar a maior rajada.", "Além da taxa média, importa a maior rajada."),
        ("A política adotada quando a capacidade é excedida deve ser explícita e acompanhada por contadores de perda.",
         "Quando a capacidade é excedida, o descarte é explícito e registrado por contadores de perda."),
    ],
    "v2_delineamento.tex": [
        ("Ela deve ser controlada, registrada ou amostrada por repetições adequadas.",
         "Ela é amostrada naturalmente pelo disparo assíncrono e deve ser controlada ou registrada em repetições futuras."),
        ("O Quadro~\\ref{v2:qua:configuracao_pareada_secao5} deve ser preenchido com os valo",
         "O Quadro~\\ref{v2:qua:configuracao_pareada_secao5} apresenta os valo"),
    ],
    "v2_implementacao.tex": [
        ("O indicador de saturação do ADC deve ser distinguido dos transbordos internos.",
         "O indicador de saturação do ADC é distinto dos indicadores de transbordo interno."),
        ("O número de registros gerados, entregues ao DMA, encapsulados e recebidos deve ser contado separadamente.",
         "Os registros gerados, entregues ao DMA, encapsulados e recebidos são contados separadamente."),
        ("Se comandos de configuração forem enviados por UDP, a aplicação deverá utilizar identificadores de operação, confirmação e tratamento de duplicação. Operações repetidas devem ser idempotentes ou possuir regra explícita que impeça efeitos inconsistentes.",
         "Comandos de configuração enviados por UDP exigem identificadores de operação, confirmação e tratamento de duplicação, e operações repetidas precisam ser idempotentes ou ter regra explícita que impeça efeitos inconsistentes."),
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
print("segunda passada aplicada")
