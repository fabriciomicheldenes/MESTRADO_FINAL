// Testbench da medição de ciclos na MicroBlaze (ver mb_sim.c).
// Registra em marcas.txt "<ciclo> <valor hex>" a cada mudança do GPIO e encerra na marca final
// (byte alto 0x0F ou 0x1F).
`timescale 1ns / 1ps
module tb_mb;
    reg clk = 1'b0;
    reg reset = 1'b1;
    wire [31:0] gpio;
    always #5 clk = ~clk;                 // 100 MHz

    mb_wrapper dut (.clk_100(clk), .reset(reset), .gpio_tri_o(gpio));

    integer ciclo = 0;
    integer f;
    reg [31:0] ant = 32'h0;
    initial begin
        f = $fopen("marcas.txt", "w");
        #205 reset = 1'b0;
    end
    always @(posedge clk) begin
        ciclo <= ciclo + 1;
        if (gpio !== ant) begin
            ant <= gpio;
            $fdisplay(f, "%0d %08h", ciclo, gpio);
            $fflush(f);
            if (gpio[31:24] == 8'h0F || gpio[31:24] == 8'h1F) begin
                $display("FIM ciclo %0d marca %08h", ciclo, gpio);
                $fclose(f);
                $finish;
            end
        end
    end
    // limite de segurança: 200 milhões de ciclos (2 s simulados)
    initial begin
        #2000000000;
        $display("TIMEOUT");
        $fclose(f);
        $finish;
    end
endmodule
