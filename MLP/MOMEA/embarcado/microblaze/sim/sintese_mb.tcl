# Síntese do mesmo sistema MicroBlaze usado na simulação (sim_mb.tcl), para obter os recursos
# do processador sem e com FPU no XC7A35T.
# Uso: vivado -mode batch -source sintese_mb.tcl -tclargs <dir_trabalho> <fpu:0|2> <relatorio.txt>
set pdir_syn [file normalize [lindex $argv 0]]
set fpu   [lindex $argv 1]
set rel   [file normalize [lindex $argv 2]]

create_project -force mbsyn $pdir_syn -part xc7a35ticsg324-1L
set_property target_language Verilog [current_project]
create_bd_design "mb"
create_bd_port -dir I -type clk -freq_hz 100000000 clk_100
create_bd_port -dir I -type rst reset
set_property CONFIG.POLARITY ACTIVE_HIGH [get_bd_ports reset]
set mb [create_bd_cell -type ip -vlnv xilinx.com:ip:microblaze:11.0 microblaze_0]
set_property -dict [list CONFIG.C_AREA_OPTIMIZED {0} CONFIG.C_DEBUG_ENABLED {0} CONFIG.C_D_AXI {1} \
    CONFIG.C_USE_BARREL {1} CONFIG.C_USE_HW_MUL {2} CONFIG.C_USE_DIV {1} CONFIG.C_USE_PCMP_INSTR {1} \
    CONFIG.C_USE_FPU $fpu] $mb
apply_bd_automation -rule xilinx.com:bd_rule:microblaze -config [list \
    axi_intc 0 axi_periph Enabled cache None clk /clk_100 cores 1 \
    debug_module None ecc None local_mem 128KB preset None] $mb
set rst [get_bd_cells -filter {VLNV =~ "*proc_sys_reset*"}]
connect_bd_net [get_bd_ports reset] [get_bd_pins $rst/ext_reset_in]
set um [create_bd_cell -type ip -vlnv xilinx.com:ip:xlconstant:1.1 const_um]
connect_bd_net [get_bd_pins $um/dout] [get_bd_pins $rst/dcm_locked]
set gpio [create_bd_cell -type ip -vlnv xilinx.com:ip:axi_gpio:2.0 axi_gpio_0]
set_property -dict [list CONFIG.C_ALL_OUTPUTS {1} CONFIG.C_GPIO_WIDTH {32}] $gpio
apply_bd_automation -rule xilinx.com:bd_rule:axi4 -config [list \
    Clk_master Auto Clk_slave Auto Clk_xbar Auto Master {/microblaze_0 (Periph)} \
    Slave /axi_gpio_0/S_AXI intc_ip {New AXI Interconnect} master_apm 0] \
    [get_bd_intf_pins axi_gpio_0/S_AXI]
make_bd_intf_pins_external -name gpio [get_bd_intf_pins axi_gpio_0/GPIO]
validate_bd_design
save_bd_design
set bdf [get_files mb.bd]
set_property synth_checkpoint_mode None $bdf   ;# IPs sintetizados junto (sem caixas-pretas)
generate_target synthesis $bdf
make_wrapper -files $bdf -top
add_files -norecurse [file join $pdir_syn mbsyn.gen sources_1 bd mb hdl mb_wrapper.v]
set_property top mb_wrapper [current_fileset]
# síntese fora de contexto (sem pinos): só interessa a ocupação por hierarquia
synth_design -top mb_wrapper -part xc7a35ticsg324-1L -mode out_of_context
report_utilization -hierarchical -hierarchical_depth 3 -file $rel
report_utilization -file [file rootname $rel]_resumo.txt
exit
