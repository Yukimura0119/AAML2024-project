// Copyright 2021 The CFU-Playground Authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
`include "global_buffer_bram.v"
`include "TPU.v"


module Cfu (
  input               cmd_valid,
  output              cmd_ready,
  input      [9:0]    cmd_payload_function_id,
  input      [31:0]   cmd_payload_inputs_0,
  input      [31:0]   cmd_payload_inputs_1,
  output reg          rsp_valid,
  input               rsp_ready,
  output reg [31:0]   rsp_payload_outputs_0,
  input               reset,
  input               clk
);


  wire            A_wr_en;
  wire   [5:0]    A_index;
  wire   [31:0]   A_data_in;
  wire   [31:0]   A_data_out;

  wire            B_wr_en;
  wire   [5:0]    B_index;
  wire   [31:0]   B_data_in;
  wire   [31:0]   B_data_out;

  wire            C_wr_en;
  wire   [5:0]    C_index;
  wire   [127:0]  C_data_in;
  wire   [127:0]  C_data_out;

  wire   [31:0]   C_data_sel_out;
  wire   [8:0]    input_offset;

  assign A_wr_en = A_wr_en_reg;
  assign B_wr_en = B_wr_en_reg;

  assign A_data_in = input_0_reg;
  assign B_data_in = input_1_reg;

  reg [5:0] buffer_waddr;

  assign A_index =  (~busy)  ? buffer_waddr[5:0] : A_index_tpu[5:0];
  assign B_index =  (~busy)  ? buffer_waddr[5:0] : B_index_tpu[5:0];
  assign C_index =  (~busy)  ? input_0_reg[5:0] : C_index_tpu[5:0];

  assign C_data_sel_out = (input_1_reg == 0) ? C_data_out[127:96] : 
                          (input_1_reg == 1) ? C_data_out[95:64] : 
                          (input_1_reg == 2) ? C_data_out[63:32] : 
                          (input_1_reg == 3) ? C_data_out[31:0] : 32'b0;


  assign input_offset = input_offset_reg;

  // input1 right shift 2
  // reg signed [31:0] AddFunc_input1_val;
  // reg signed [31:0] AddFunc_input2_val;
  // reg signed [31:0] AddFunc_input1_mul;
  // reg signed [31:0] AddFunc_input2_mul;
  // wire signed [63:0] result_input1, result_input2;
  // wire signed [31:0] nudge_input1, nudge_input2;
  // wire signed [31:0] result1_sat, result2_sat, result1_sat_shift, sum;
  // assign result_input1 = $signed(AddFunc_input1_val) * $signed(AddFunc_input1_mul);
  // assign result_input2 = $signed(AddFunc_input2_val) * $signed(AddFunc_input2_mul);
  // assign nudge_input1 = (result_input1 >= 64'd0) ? 32'h40000000 : 32'hC0000001;
  // assign nudge_input2 = (result_input2 >= 64'd0) ? 32'h40000000 : 32'hC0000001;
  // assign result1_sat = $signed(($signed(result_input1) + $signed(nudge_input1))) >>> 31;
  // assign result2_sat = $signed(($signed(result_input2) + $signed(nudge_input2))) >>> 31;
  // assign result1_sat_shift = result1_sat >>> 2;
  // assign sum = $signed(result1_sat_shift) + $signed(result2_sat);

  global_buffer_bram #(
      .ADDR_BITS(6),
      .DATA_BITS(32)
  )
  gbuff_A(
      .clk(clk),
      .rst_n(1'b1),
      .wr_en(A_wr_en),
      .ram_en(1'b1),
      .index(A_index),
      .data_in(A_data_in),
      .data_out(A_data_out)
  );

  global_buffer_bram #(
      .ADDR_BITS(6),
      .DATA_BITS(32)
  ) gbuff_B(
      .clk(clk),
      .rst_n(1'b1),
      .wr_en(B_wr_en),
      .ram_en(1'b1),
      .index(B_index),
      .data_in(B_data_in),
      .data_out(B_data_out)
  );

  global_buffer_bram #(
      .ADDR_BITS(6),
      .DATA_BITS(128)
  ) gbuff_C(
      .clk(clk),
      .rst_n(1'b1),
      .wr_en(C_wr_en),
      .ram_en(1'b1),
      .index(C_index),
      .data_in(C_data_in),
      .data_out(C_data_out)
  );


  wire busy;

  wire A_wr_en_dummy, B_wr_en_dummy;
  wire [31:0] A_data_in_dummy, B_data_in_dummy;

  wire [5:0] A_index_tpu;
  wire [5:0] B_index_tpu;
  wire [5:0] C_index_tpu;

  reg in_valid;
  reg [8:0] input_offset_reg;

  localparam SIMDOffset = $signed(9'd128);

  // SIMD multiply step:
  wire signed [15:0] prod_0, prod_1, prod_2, prod_3;
  assign prod_0 =  ($signed(cmd_payload_inputs_0[7 : 0]) + SIMDOffset)
                  * $signed(cmd_payload_inputs_1[7 : 0]);
  assign prod_1 =  ($signed(cmd_payload_inputs_0[15: 8]) + SIMDOffset)
                  * $signed(cmd_payload_inputs_1[15: 8]);
  assign prod_2 =  ($signed(cmd_payload_inputs_0[23:16]) + SIMDOffset)
                  * $signed(cmd_payload_inputs_1[23:16]);
  assign prod_3 =  ($signed(cmd_payload_inputs_0[31:24]) + SIMDOffset)
                  * $signed(cmd_payload_inputs_1[31:24]);

  wire signed [31:0] sum_prods;
  assign sum_prods = prod_0 + prod_1 + prod_2 + prod_3;

  TPU myTPU(
    .clk            (clk),
    .rst_n          (reset),
    .in_valid       (in_valid),
    .input_offset   (input_offset),
    .K              (5'd16),
    .M              (5'd16),
    .N              (5'd16),
    .busy           (busy),

    .A_wr_en        (A_wr_en_dummy),
    .A_index        (A_index_tpu),
    .A_data_in      (A_data_in_dummy),
    .A_data_out     (A_data_out),  

    .B_wr_en        (B_wr_en_dummy),
    .B_index        (B_index_tpu),
    .B_data_in      (B_data_in_dummy),
    .B_data_out     (B_data_out),

    .C_wr_en        (C_wr_en),
    .C_index        (C_index_tpu),
    .C_data_in      (C_data_in)
);

  assign cmd_ready = ~rsp_valid;

  reg is_run, A_wr_en_reg, B_wr_en_reg;
  reg [31:0] input_0_reg, input_1_reg;
  reg [9:0] func_id_reg;

  always @(posedge clk) begin
    if (reset) begin
      rsp_valid <= 1'b0;

      rsp_payload_outputs_0 <= 32'b0;
      in_valid <= 1'b0;

      input_0_reg <= 31'b0;
      input_1_reg <= 31'b0;
      func_id_reg <= 10'b0;

      A_wr_en_reg <= 1'b0;
      B_wr_en_reg <= 1'b0;
      buffer_waddr <= 0;

      // AddFunc_input1_val <= 32'd0;
      // AddFunc_input2_val <= 32'd0;
      // AddFunc_input1_mul <= 32'd0;
      // AddFunc_input2_mul <= 32'd0;

      input_offset_reg <= 9'b0;

      is_run <= 1'b0;
    end 
    else if (rsp_valid) begin
      rsp_valid <= ~rsp_ready;
    end 
    else if (is_run) begin
      case (func_id_reg[9:3])
        7'd0: begin
          rsp_valid <= 1'b1;
          is_run <= 1'b0;
          A_wr_en_reg <= 1'b0;
          B_wr_en_reg <= 1'b0;
          buffer_waddr <= buffer_waddr + 1;
        end
        7'd1: begin
          rsp_valid <= (busy);
          is_run <= ~(busy);
          in_valid <= ~(busy); 
        end
        7'd2: begin
          if (busy) begin
            rsp_valid <= 1'b0;
            is_run <= 1'b1;
          end
          else begin
            rsp_valid <= 1'b1;
            is_run <= 1'b0; 
            rsp_payload_outputs_0 <= C_data_sel_out;
          end
        end
        default: begin
          is_run <= 1'b0;
          rsp_valid <= 1'b0;
          rsp_payload_outputs_0 <= 32'b0;
        end
      endcase

    end
    else if (cmd_valid) begin
      case (cmd_payload_function_id[9:3])
        7'd0: begin
          rsp_valid <= 1'b0;
          is_run <= 1'b1;
          A_wr_en_reg <= 1'b1;
          B_wr_en_reg <= 1'b1;
        end
        7'd1: begin
          rsp_valid <= 1'b0;
          in_valid <= 1'b1;
          is_run <= 1'b1;
          input_offset_reg <= cmd_payload_inputs_0[8:0];
        end
        7'd2: begin
          rsp_valid <= 1'b0;
          is_run <= 1'b1;
          buffer_waddr <= 0;
        end 
        7'd3: begin
          rsp_valid <= 1'b1;
          is_run <= 1'b0;
          rsp_payload_outputs_0 <= 0;
        end
        7'd4: begin
          rsp_valid <= 1'b1;
          is_run <= 1'b0;
          rsp_payload_outputs_0 <= rsp_payload_outputs_0 + sum_prods;
        end
        // 7'd3: begin
        //   rsp_valid <= 1'b1;
        //   is_run <= 1'b0;
        //   AddFunc_input1_val <= $signed(cmd_payload_inputs_0[7:0]);
        //   AddFunc_input2_val <= $signed(cmd_payload_inputs_1[7:0]);
        // end
        // 7'd4: begin
        //   rsp_valid <= 1'b1;
        //   is_run <= 1'b0;
        //   AddFunc_input1_val <= $signed(($signed(cmd_payload_inputs_0[7:0]) + $signed(AddFunc_input1_val)) <<< 20); 
        //   AddFunc_input2_val <= $signed(($signed(cmd_payload_inputs_1[7:0]) + $signed(AddFunc_input2_val)) <<< 20); 
        // end
        // 7'd5: begin
        //   rsp_valid <= 1'b1;
        //   is_run <= 1'b0;
        //   AddFunc_input1_mul <= $signed(cmd_payload_inputs_0);
        //   AddFunc_input2_mul <= $signed(cmd_payload_inputs_1);
        // end
        // 7'd6: begin
        //   rsp_valid <= 1'b1;
        //   is_run <= 1'b0;
        //   rsp_payload_outputs_0 <= sum;
        // end
        default: begin
          rsp_valid <= 1'b1;
          is_run <= 1'b0;
        end
      endcase
      func_id_reg <= cmd_payload_function_id;
      input_0_reg <= cmd_payload_inputs_0;
      input_1_reg <= cmd_payload_inputs_1;
    end
  end

endmodule
