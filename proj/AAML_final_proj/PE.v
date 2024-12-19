
module PE(
    clk,
    rst_n,
    input_offset,
    Ifmap,
    weight,
    clear,
    Ifmap_out,
    weight_out,
    Ofmap
);
 

input clk;
input rst_n;
input [7:0]      Ifmap; 
input [7:0]      weight;
input [8:0]      input_offset;
input clear;
output [7:0]     Ifmap_out;
output [7:0]     weight_out;
output [31:0]    Ofmap;

reg [7:0]    Ifmap_reg;
reg [7:0]    weight_reg;
reg [31:0]   Ofmap_reg;

wire [15:0] Ifmap_mul;
wire [16:0] Offset_mul;
assign Ifmap_mul = $signed(Ifmap) * $signed(weight);
assign Offset_mul = $signed(input_offset) * $signed(weight);


assign Ofmap = Ofmap_reg;
assign Ifmap_out = Ifmap_reg;
assign weight_out = weight_reg; 

always @(posedge clk) begin
    if(rst_n) begin
        Ofmap_reg <= 32'd0;
        Ifmap_reg <= 8'd0;
        weight_reg <= 8'd0;
    end
    else begin
        Ofmap_reg <= clear ? 32'd0 : $signed($signed(Ofmap_reg) + $signed(Ifmap_mul) + $signed(Offset_mul));
        Ifmap_reg <= clear ? 32'd0 : Ifmap;
        weight_reg <= clear ? 32'd0 : weight;
    end
end

endmodule