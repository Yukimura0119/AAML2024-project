`include "shift_reg.v"
`include "systolic_array.v"

module TPU(
    clk,
    rst_n,

    in_valid,
    input_offset,
    K,
    M,
    N,
    busy,

    A_wr_en,
    A_index,
    A_data_in,
    A_data_out, 

    B_wr_en,
    B_index,
    B_data_in,
    B_data_out,

    C_wr_en,
    C_index,
    C_data_in
);

parameter STATE_IDLE   = 3'd0,
          STATE_START  = 3'd1, 
          STATE_RUN    = 3'd2,
          STATE_PENDZ  = 3'd3,
          STATE_OUTPUT = 3'd4,
          STATE_DONE   = 3'd5;

reg [2:0] state, next_state;

input clk;
input rst_n;
input            in_valid;
input [8:0]      input_offset;
input [4:0]      K;
input [4:0]      M;
input [4:0]      N;
output           busy;

output           A_wr_en;
output [5:0]    A_index;
output [31:0]    A_data_in;
input  [31:0]    A_data_out;

output           B_wr_en;
output [5:0]    B_index;
output [31:0]    B_data_in;
input  [31:0]    B_data_out;

output           C_wr_en;
output [5:0]    C_index;
output [127:0]   C_data_in;


//* Implement your design here

// SRAM A, B never write
assign A_wr_en = 1'b0;
assign B_wr_en = 1'b0;
assign C_wr_en = (state == STATE_OUTPUT);

assign A_data_in = 32'd0;
assign B_data_in = 32'd0;
assign C_data_in =  (output_counter == 0) ? Ofmap1 : 
                    (output_counter == 1) ? Ofmap2 : 
                    (output_counter == 2) ? Ofmap3 : 
                    (output_counter == 3) ? Ofmap4 : 128'd0;

assign busy = (state != STATE_IDLE);

wire [5:0] last_A, last_B;
assign last_A = (((M_reg-1)>>2) + 1) * K_reg - 1;
assign last_B = (((N_reg-1)>>2) + 1) * K_reg - 1;

wire [7:0] A_data_out1, A_data_out2, A_data_out3, A_data_out4;
wire [7:0] B_data_out1, B_data_out2, B_data_out3, B_data_out4;

wire [7:0] shift_reg_A2_data_out, shift_reg_A3_data_out, shift_reg_A4_data_out;
wire [7:0] shift_reg_B2_data_out, shift_reg_B3_data_out, shift_reg_B4_data_out;

wire [127:0] Ofmap1, Ofmap2, Ofmap3, Ofmap4;

// counters
reg [5:0] A_index_reg, B_index_reg, C_index_reg;
reg [3:0] pending_counter, output_counter;

assign A_index = A_index_reg;
assign B_index = B_index_reg;
assign C_index = C_index_reg;

// control signals 
wire pause, is_lastA, is_lastB, is_last_blkA, clear;
wire [2:0] wait_cycles, output_wait_cycles;
wire [2:0] last_blk_size;
wire [5:0] reg_A_jump, reg_B_jump;
wire set_inputz;

reg [4:0] K_reg, M_reg, N_reg;

assign clear = (output_counter == output_wait_cycles-1) && (state == STATE_OUTPUT);
assign set_inputz = (state == STATE_IDLE || state == STATE_PENDZ || state == STATE_DONE || state == STATE_OUTPUT);
assign is_lastA = (A_index_reg == last_A);
assign is_lastB = (B_index_reg == last_B);
assign is_last_blkA = (A_index_reg >= last_A + 1 - K_reg);

assign last_blk_size = (M_reg - (((M_reg-1)>>2) << 2));
assign pause = ((B_index_reg + 1)%K_reg == 0);
assign reg_A_jump = (is_last_blkA) ? ((is_lastA && is_lastB) ? A_index_reg : 0 )
                                   : A_index_reg + 1;
assign reg_B_jump = (is_last_blkA) ? ((is_lastA && is_lastB) ? B_index_reg : B_index_reg + 1 )
                                   : B_index_reg / K_reg * K_reg;

assign wait_cycles = 3'd3;

assign output_wait_cycles = (is_last_blkA) ? last_blk_size : 3'd4;

assign A_data_out4 = (set_inputz) ? 8'd0 : A_data_out[31:24];
assign A_data_out3 = (set_inputz) ? 8'd0 : A_data_out[23:16];
assign A_data_out2 = (set_inputz) ? 8'd0 : A_data_out[15:8];
assign A_data_out1 = (set_inputz) ? 8'd0 : A_data_out[7:0];

assign B_data_out4 = (set_inputz) ? 8'd0 : B_data_out[31:24];
assign B_data_out3 = (set_inputz) ? 8'd0 : B_data_out[23:16];
assign B_data_out2 = (set_inputz) ? 8'd0 : B_data_out[15:8];
assign B_data_out1 = (set_inputz) ? 8'd0 : B_data_out[7:0];

shift_reg #(.length(1)) shift_reg_A2(clk, rst_n, A_data_out2, shift_reg_A2_data_out);
shift_reg #(.length(2)) shift_reg_A3(clk, rst_n, A_data_out3, shift_reg_A3_data_out);
shift_reg #(.length(3)) shift_reg_A4(clk, rst_n, A_data_out4, shift_reg_A4_data_out);

shift_reg #(.length(1)) shift_reg_B2(clk, rst_n, B_data_out2, shift_reg_B2_data_out);
shift_reg #(.length(2)) shift_reg_B3(clk, rst_n, B_data_out3, shift_reg_B3_data_out);
shift_reg #(.length(3)) shift_reg_B4(clk, rst_n, B_data_out4, shift_reg_B4_data_out);

systolic_array systolic_array(
    .clk(clk),
    .rst_n(rst_n),
    .clear(clear),
    .input_offset(input_offset),
    .Ifmap_in1(A_data_out1),
    .Ifmap_in2(shift_reg_A2_data_out),
    .Ifmap_in3(shift_reg_A3_data_out),
    .Ifmap_in4(shift_reg_A4_data_out),
    .weight_in1(B_data_out1),
    .weight_in2(shift_reg_B2_data_out),
    .weight_in3(shift_reg_B3_data_out),
    .weight_in4(shift_reg_B4_data_out),
    .Ofmap1(Ofmap1),
    .Ofmap2(Ofmap2),
    .Ofmap3(Ofmap3),
    .Ofmap4(Ofmap4)
);

// store K, M, N
always @(posedge clk) begin
    if(rst_n) begin
        K_reg <= 5'd0;
        M_reg <= 5'd0;
        N_reg <= 5'd0;
    end
    else begin
        K_reg <= (in_valid) ? K : K_reg;
        M_reg <= (in_valid) ? M : M_reg;
        N_reg <= (in_valid) ? N : N_reg;
    end
end

// index counter
always @(posedge clk) begin
    if(rst_n) begin
        A_index_reg <= 6'd0;
        B_index_reg <= 6'd0;
        C_index_reg <= 6'd0;
    end
    else begin
        case(state)
            STATE_IDLE: begin
                A_index_reg <= 6'd0;
                B_index_reg <= 6'd0;
                C_index_reg <= 6'd0;
            end
            STATE_START: begin
                A_index_reg <= 6'd1;
                B_index_reg <= 6'd1;
                C_index_reg <= 6'd0;
            end
            STATE_RUN: begin
                A_index_reg <= pause ? A_index_reg : A_index_reg + 1;
                B_index_reg <= pause ? B_index_reg : B_index_reg + 1;
                C_index_reg <= C_index_reg;
            end
            STATE_PENDZ: begin
                A_index_reg <= A_index_reg;
                B_index_reg <= B_index_reg;
                C_index_reg <= C_index_reg;
            end
            STATE_OUTPUT: begin
                A_index_reg <= (output_counter == output_wait_cycles-1) ? reg_A_jump : A_index_reg;
                B_index_reg <= (output_counter == output_wait_cycles-1) ? reg_B_jump : B_index_reg;
                C_index_reg <= C_index_reg + 1;
            end
            STATE_DONE: begin
                A_index_reg <= 6'd0;
                B_index_reg <= 6'd0;
                C_index_reg <= 6'd0;
            end
        endcase
    end
end

// pendingz counter
always @(posedge clk) begin
    if(rst_n) begin
        pending_counter <= 4'd0;
    end
    else begin
        case (state)
            STATE_PENDZ: pending_counter <= pending_counter + 1;
            default: pending_counter <= 4'd0;
        endcase
    end
end

// output counter
always @(posedge clk) begin
    if(rst_n) begin
        output_counter <= 4'd0;
    end
    else begin
        case (state)
            STATE_OUTPUT: output_counter <= output_counter + 1;
            default: output_counter <= 4'd0;
        endcase
    end
end

// calc next state
always @(*) begin
    case(state)
        STATE_IDLE: begin
            next_state = in_valid ? STATE_START : STATE_IDLE;
        end
        STATE_START: begin
            next_state = STATE_RUN;
        end
        STATE_RUN: begin
            next_state = pause ? STATE_PENDZ : STATE_RUN;
        end
        STATE_PENDZ: begin
            next_state = (pending_counter == wait_cycles-1) ? STATE_OUTPUT : STATE_PENDZ;
        end
        STATE_OUTPUT: begin
            if(is_lastA && is_lastB) begin
                next_state = (output_counter == output_wait_cycles-1) ? STATE_DONE : STATE_OUTPUT;
            end
            else begin
                next_state = (output_counter == output_wait_cycles-1) ? STATE_RUN : STATE_OUTPUT;
            end
        end
        STATE_DONE: begin
            next_state = STATE_IDLE;
        end
    endcase
end

always @(posedge clk) begin
    if (rst_n) begin
        state <= STATE_IDLE;
    end
    else begin
        state <= next_state;
    end
end

endmodule