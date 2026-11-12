# Implementação (síntese + place & route fora de contexto no Vivado) do IP já sintetizado
# por run_hls.tcl, para obter recursos e temporização pós-roteamento.
# Uso: E:\Xilinx\Vitis_HLS\2022.2\bin\vitis_hls.bat -f run_hls_impl.tcl
open_project proj_mlp_q8
open_solution sol_100MHz
export_design -flow impl -format ip_catalog -rtl verilog
exit
