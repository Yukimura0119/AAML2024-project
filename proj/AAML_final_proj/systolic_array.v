`include "PE.v"

module systolic_array(
    clk,
    rst_n,
    clear,
    input_offset,
    weight_in1,
    weight_in2,
    weight_in3,
    weight_in4,
    Ifmap_in1,
    Ifmap_in2,
    Ifmap_in3,
    Ifmap_in4,
    Ofmap1,
    Ofmap2,
    Ofmap3,
    Ofmap4
);


input clk;
input rst_n;
input clear;

input [8:0] input_offset;
input [7:0] Ifmap_in1;
input [7:0] Ifmap_in2;
input [7:0] Ifmap_in3;
input [7:0] Ifmap_in4;

input [7:0] weight_in1;
input [7:0] weight_in2;
input [7:0] weight_in3;
input [7:0] weight_in4;

output [127:0] Ofmap1;
output [127:0] Ofmap2;
output [127:0] Ofmap3;
output [127:0] Ofmap4;

wire [7:0] Ifmap_out11, Ifmap_out12, Ifmap_out13, Ifmap_out14;
wire [7:0] weight_out11, weight_out12, weight_out13, weight_out14;
wire [31:0] Ofmap11, Ofmap12, Ofmap13, Ofmap14;

assign Ofmap1 = {Ofmap11, Ofmap12, Ofmap13, Ofmap14};
assign Ofmap2 = {Ofmap21, Ofmap22, Ofmap23, Ofmap24};
assign Ofmap3 = {Ofmap31, Ofmap32, Ofmap33, Ofmap34};
assign Ofmap4 = {Ofmap41, Ofmap42, Ofmap43, Ofmap44};

PE PE11(clk, rst_n, input_offset, Ifmap_in1, weight_in1, clear, Ifmap_out11, weight_out11, Ofmap11);
PE PE12(clk, rst_n, input_offset, Ifmap_out11, weight_in2, clear, Ifmap_out12, weight_out12, Ofmap12);
PE PE13(clk, rst_n, input_offset, Ifmap_out12, weight_in3, clear, Ifmap_out13, weight_out13, Ofmap13);
PE PE14(clk, rst_n, input_offset, Ifmap_out13, weight_in4, clear, Ifmap_out14, weight_out14, Ofmap14);

wire [7:0] Ifmap_out21, Ifmap_out22, Ifmap_out23, Ifmap_out24;
wire [7:0] weight_out21, weight_out22, weight_out23, weight_out24;
wire [31:0] Ofmap21, Ofmap22, Ofmap23, Ofmap24;

PE PE21(clk, rst_n, input_offset, Ifmap_in2, weight_out11, clear, Ifmap_out21, weight_out21, Ofmap21);
PE PE22(clk, rst_n, input_offset, Ifmap_out21, weight_out12, clear, Ifmap_out22, weight_out22, Ofmap22);
PE PE23(clk, rst_n, input_offset, Ifmap_out22, weight_out13, clear, Ifmap_out23, weight_out23, Ofmap23);
PE PE24(clk, rst_n, input_offset, Ifmap_out23, weight_out14, clear, Ifmap_out24, weight_out24, Ofmap24);

wire [7:0] Ifmap_out31, Ifmap_out32, Ifmap_out33, Ifmap_out34;
wire [7:0] weight_out31, weight_out32, weight_out33, weight_out34;
wire [31:0] Ofmap31, Ofmap32, Ofmap33, Ofmap34;

PE PE31(clk, rst_n, input_offset, Ifmap_in3, weight_out21, clear, Ifmap_out31, weight_out31, Ofmap31);
PE PE32(clk, rst_n, input_offset, Ifmap_out31, weight_out22, clear, Ifmap_out32, weight_out32, Ofmap32);
PE PE33(clk, rst_n, input_offset, Ifmap_out32, weight_out23, clear, Ifmap_out33, weight_out33, Ofmap33);
PE PE34(clk, rst_n, input_offset, Ifmap_out33, weight_out24, clear, Ifmap_out34, weight_out34, Ofmap34);

wire [7:0] Ifmap_out41, Ifmap_out42, Ifmap_out43, Ifmap_out44;
wire [7:0] weight_out41, weight_out42, weight_out43, weight_out44;
wire [31:0] Ofmap41, Ofmap42, Ofmap43, Ofmap44;

PE PE41(clk, rst_n, input_offset, Ifmap_in4, weight_out31, clear, Ifmap_out41, weight_out41, Ofmap41);
PE PE42(clk, rst_n, input_offset, Ifmap_out41, weight_out32, clear, Ifmap_out42, weight_out42, Ofmap42);
PE PE43(clk, rst_n, input_offset, Ifmap_out42, weight_out33, clear, Ifmap_out43, weight_out43, Ofmap43);
PE PE44(clk, rst_n, input_offset, Ifmap_out43, weight_out34, clear, Ifmap_out44, weight_out44, Ofmap44);

endmodule