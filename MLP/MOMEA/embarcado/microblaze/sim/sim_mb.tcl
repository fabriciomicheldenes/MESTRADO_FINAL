# Simulação RTL (xsim) de um sistema MicroBlaze mínimo executando um ELF, para medir ciclos.
#
# Uso (Vivado 2022.2, modo batch):
#   vivado -mode batch -source sim_mb.tcl -tclargs <dir_projeto> <fpu:0|2> <elf1> [<elf2> ...]
# Cria o projeto, gera os scripts do xsim (-scripts_only) e, para cada ELF, a imagem da
# memória local <dir_projeto>/<nome_do_elf>.mem. A compilação, a elaboração e as simulações
# são feitas fora do Vivado por rodar_sim_mb.py (o modelo da BRAM lê o .mem no início de cada
# simulação, de modo que um único snapshot elaborado serve para todos os ELFs).
#
# Sistema: MicroBlaze v11 (5 estágios, sem caches, sem depuração), barrel shifter,
# multiplicador de 64 bits (MULH), divisor, comparação de padrões, FPU opcional
# (0 = sem, 2 = estendida: conversão e raiz), 128 kB de memória local LMB (BRAM),
# AXI GPIO de 32 bits (saída) em 0x40000000, clock de 100 MHz e reset externos.
# Peça: xc7a35ticsg324-1L (a mesma do MOMEA).

set proj_dir [file normalize [lindex $argv 0]]
set fpu      [lindex $argv 1]
set elfs     [lrange $argv 2 end]
set aqui     [file dirname [file normalize [info script]]]

create_project -force mbsim $proj_dir -part xc7a35ticsg324-1L
set_property target_language Verilog [current_project]
create_bd_design "mb"

create_bd_port -dir I -type clk -freq_hz 100000000 clk_100
create_bd_port -dir I -type rst reset
set_property CONFIG.POLARITY ACTIVE_HIGH [get_bd_ports reset]

set mb [create_bd_cell -type ip -vlnv xilinx.com:ip:microblaze:11.0 microblaze_0]
set_property -dict [list \
    CONFIG.C_AREA_OPTIMIZED {0} \
    CONFIG.C_DEBUG_ENABLED {0} \
    CONFIG.C_D_AXI {1} \
    CONFIG.C_USE_BARREL {1} \
    CONFIG.C_USE_HW_MUL {2} \
    CONFIG.C_USE_DIV {1} \
    CONFIG.C_USE_PCMP_INSTR {1} \
    CONFIG.C_USE_FPU $fpu \
] $mb

apply_bd_automation -rule xilinx.com:bd_rule:microblaze -config [list \
    axi_intc 0 axi_periph Enabled cache None clk /clk_100 cores 1 \
    debug_module None ecc None local_mem 128KB preset None] $mb

# reset externo ativo em alto; dcm_locked fixo em 1
set rst [get_bd_cells -filter {VLNV =~ "*proc_sys_reset*"}]
connect_bd_net [get_bd_ports reset] [get_bd_pins $rst/ext_reset_in]
set um [create_bd_cell -type ip -vlnv xilinx.com:ip:xlconstant:1.1 const_um]
connect_bd_net [get_bd_pins $um/dout] [get_bd_pins $rst/dcm_locked]

# GPIO de marcas
set gpio [create_bd_cell -type ip -vlnv xilinx.com:ip:axi_gpio:2.0 axi_gpio_0]
set_property -dict [list CONFIG.C_ALL_OUTPUTS {1} CONFIG.C_GPIO_WIDTH {32}] $gpio
apply_bd_automation -rule xilinx.com:bd_rule:axi4 -config [list \
    Clk_master Auto Clk_slave Auto Clk_xbar Auto Master {/microblaze_0 (Periph)} \
    Slave /axi_gpio_0/S_AXI intc_ip {New AXI Interconnect} master_apm 0] \
    [get_bd_intf_pins axi_gpio_0/S_AXI]
make_bd_intf_pins_external -name gpio [get_bd_intf_pins axi_gpio_0/GPIO]
set seg [get_bd_addr_segs -of_objects [get_bd_addr_spaces microblaze_0/Data] -filter {NAME =~ "*gpio*"}]
set_property offset 0x40000000 $seg
set_property range 64K $seg

validate_bd_design
save_bd_design
set bdf [get_files mb.bd]
generate_target simulation $bdf
make_wrapper -files $bdf -top
add_files -norecurse [file join $proj_dir mbsim.gen sources_1 bd mb hdl mb_wrapper.v]
set_property top mb_wrapper [current_fileset]

add_files -fileset sim_1 -norecurse [file join $aqui tb_mb.v]
set_property top tb_mb [get_filesets sim_1]
set_property top_lib xil_defaultlib [get_filesets sim_1]
set_property -name {xsim.simulate.runtime} -value {all} -objects [get_filesets sim_1]
set_property -name {xsim.simulate.log_all_signals} -value {false} -objects [get_filesets sim_1]
# sem instrumentação de depuração: o xsim fica bem mais rápido (não há ondas a observar)
set_property -name {xsim.elaborate.debug_level} -value {off} -objects [get_filesets sim_1]

set anterior ""
foreach elf $elfs {
    set elf [file normalize $elf]
    set nome [file rootname [file tail $elf]]
    catch {puts "==== Simulando $nome (FPU=$fpu) ===="}
    if {$anterior ne ""} { remove_files -fileset sim_1 $anterior }
    # o ELF de simulação precisa estar no conjunto sim_1 (em sources_1 vale só para o bitstream)
    add_files -fileset sim_1 -norecurse $elf
    set f [get_files -of_objects [get_filesets sim_1] $elf]
    set_property SCOPED_TO_REF mb $f
    set_property SCOPED_TO_CELLS {microblaze_0} $f
    set anterior $f
    launch_simulation -simset sim_1 -mode behavioral -scripts_only
    set dir_sim [file join $proj_dir mbsim.sim sim_1 behav xsim]
    set mems [glob -directory $dir_sim *lmb_bram*.mem]
    # confere que a imagem veio do ELF pedido, e não do bootloop padrão
    set fm [open [lindex $mems 0] r]; set cab [read $fm 2000]; close $fm
    if {[string first [file tail $elf] $cab] < 0} {
        error "imagem de memória não foi gerada a partir de [file tail $elf]"
    }
    file copy -force [lindex $mems 0] [file join $proj_dir $nome.mem]
    catch {puts "==== $nome: imagem [file tail [lindex $mems 0]] -> $nome.mem ===="}
}
set fp [open [file join $proj_dir nome_mem.txt] w]
puts $fp [file tail [lindex $mems 0]]
close $fp
exit
