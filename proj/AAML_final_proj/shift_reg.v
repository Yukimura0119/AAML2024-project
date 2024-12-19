module shift_reg #(parameter length = 3) (
    clk,
    rst_n,
    data_in,
    data_out
);


input clk;
input rst_n;
input [7:0] data_in;
output [7:0] data_out;

integer i;

reg [7:0] data_reg [0:length-1];

assign data_out = data_reg[0];

always @(posedge clk) begin
    if(rst_n) begin
        for(i = 0; i < length; i = i + 1) begin
            data_reg[i] <= 8'd0;
        end
    end
    else begin
        data_reg[length - 1] <= data_in;
        for(i = 1; i < length; i = i + 1) begin
            data_reg[i-1] <= data_reg[i];
        end
    end
end
endmodule