# Vitis HLS 2022.2 - núcleo int8 das MLPs de EA para o XC7A35T (MOMEA).
# Uso (a partir desta pasta):
#   E:\Xilinx\Vitis_HLS\2022.2\bin\vitis_hls.bat -f run_hls.tcl
# Etapas: simulação C (csim) -> síntese (csynth) -> cossimulação C/RTL (cosim, xsim)
#         -> exportação como IP (AXI4-Lite) para o bloco do Vivado.
# Defina a variável de ambiente HLS_SEM_COSIM=1 para pular a cossimulação.

open_project -reset proj_mlp_q8
set_top mlp_nucleo_q8
add_files ../src/mlp_nucleo_q8.c -cflags "-I../src -I../gerado"
add_files -tb tb_mlp_nucleo_q8.c -cflags "-I../src -I../gerado -Wno-unknown-pragmas"

open_solution -reset "sol_100MHz" -flow_target vivado
set_part {xc7a35ticsg324-1L}
create_clock -period 10 -name default
config_export -format ip_catalog -rtl verilog -vendor momea -library hls -version 1.0 \
    -display_name "MLP EA int8 (FREQ + CFG)"

csim_design
csynth_design
if {![info exists ::env(HLS_SEM_COSIM)]} {
    cosim_design -rtl verilog -trace_level none
}
export_design
exit
