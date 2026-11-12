@echo off
REM Regenera os pesos (opcional) e roda o teste de equivalencia C x Python no PC.
REM Usa o gcc que acompanha o Vitis HLS 2022.2.
setlocal
set GCCDIR=E:\Xilinx\Vitis_HLS\2022.2\tps\win64\msys64\mingw64\bin
set PATH=%GCCDIR%;%PATH%
cd /d %~dp0
if "%1"=="--exportar" (
    ..\.venv\Scripts\python.exe exportar_modelo.py || exit /b 1
    ..\.venv\Scripts\python.exe exportar_hits.py || exit /b 1
)
cd src
set CF=-O2 -std=c99 -Wall -Wno-unknown-pragmas -Wno-unused-label -ffp-contract=off
gcc %CF% -o teste_pc.exe teste_pc.c mlp_inf.c mlp_nucleo_q8.c -lm || exit /b 1
gcc %CF% -o teste_descr_exato.exe teste_descritores.c ea_descritores.c mlp_inf.c mlp_nucleo_q8.c -lm || exit /b 1
gcc %CF% -DEA_DESCR_HISTOGRAMA -o teste_descr_hist.exe teste_descritores.c ea_descritores.c mlp_inf.c mlp_nucleo_q8.c -lm || exit /b 1
echo ===== 1/3 MLP: C x Python =====
.\teste_pc.exe ..\gerado\vetores_teste.bin || exit /b 1
echo ===== 2/3 Cadeia completa, percentis exatos =====
.\teste_descr_exato.exe ..\gerado\hits_teste.bin ..\gerado\vetores_teste.bin || exit /b 1
echo ===== 3/3 Cadeia completa, percentis por histograma =====
.\teste_descr_hist.exe ..\gerado\hits_teste.bin ..\gerado\vetores_teste.bin
