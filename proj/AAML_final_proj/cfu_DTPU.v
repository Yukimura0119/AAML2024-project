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


  wire            A_wr_en, A_wr_en2;
  wire   [5:0]    A_index, A_index2;
  wire   [31:0]   A_data_in, A_data_in2;
  wire   [31:0]   A_data_out, A_data_out2;

  wire            B_wr_en, B_wr_en2;
  wire   [5:0]    B_index, B_index2;
  wire   [31:0]   B_data_in, B_data_in2;
  wire   [31:0]   B_data_out, B_data_out2;

  wire            C_wr_en, C_wr_en2;
  wire   [5:0]    C_index, C_index2;
  wire   [127:0]  C_data_in , C_data_in2;
  wire   [127:0]  C_data_out, C_data_out2;

  wire   [31:0]   C_data_sel_out, C_data_sel_out2;
  wire   [8:0]    input_offset;

  assign A_wr_en = A_wr_en_reg;
  assign A_wr_en2 = A_wr_en_reg2;
  assign B_wr_en = B_wr_en_reg;
  assign B_wr_en2 = B_wr_en_reg2;

  assign A_data_in = input_0_reg;
  assign B_data_in = input_1_reg;

  assign A_data_in2 = input_0_reg;
  assign B_data_in2 = input_1_reg;

  reg [5:0] tpu1_addr, tpu2_addr;

  assign A_index =  (~busy)  ? tpu1_addr[5:0] : A_index_tpu[5:0];
  assign B_index =  (~busy)  ? tpu1_addr[5:0] : B_index_tpu[5:0];

  assign A_index2 = (~busy2) ? tpu2_addr[5:0] : A_index_tpu2[5:0];
  assign B_index2 = (~busy2) ? tpu2_addr[5:0] : B_index_tpu2[5:0];

  assign C_index =  (~busy)  ? input_0_reg[5:0] : C_index_tpu[5:0];
  assign C_index2 = (~busy2) ? input_0_reg[5:0] : C_index_tpu2[5:0];

  assign C_data_sel_out = (input_1_reg == 0) ? C_data_out[127:96] : 
                          (input_1_reg == 1) ? C_data_out[95:64] : 
                          (input_1_reg == 2) ? C_data_out[63:32] : 
                          (input_1_reg == 3) ? C_data_out[31:0] : 32'b0;

  assign C_data_sel_out2 = (input_1_reg == 0) ? C_data_out2[127:96] : 
                           (input_1_reg == 1) ? C_data_out2[95:64] : 
                           (input_1_reg == 2) ? C_data_out2[63:32] : 
                           (input_1_reg == 3) ? C_data_out2[31:0] : 32'b0;

  assign input_offset = input_offset_reg;
  

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
  ) gbuff_A2(
      .clk(clk),
      .rst_n(1'b1),
      .wr_en(A_wr_en2),
      .ram_en(1'b1),
      .index(A_index2),
      .data_in(A_data_in2),
      .data_out(A_data_out2)
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
      .DATA_BITS(32)
  ) gbuff_B2(
      .clk(clk),
      .rst_n(1'b1),
      .wr_en(B_wr_en2),
      .ram_en(1'b1),
      .index(B_index2),
      .data_in(B_data_in2),
      .data_out(B_data_out2)
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

  global_buffer_bram #(
      .ADDR_BITS(6),
      .DATA_BITS(128)
  ) gbuff_C2(
      .clk(clk),
      .rst_n(1'b1),
      .wr_en(C_wr_en2),
      .ram_en(1'b1),
      .index(C_index2),
      .data_in(C_data_in2),
      .data_out(C_data_out2)
  );

  wire busy, busy2;

  wire A_wr_en_dummy, B_wr_en_dummy;
  wire [31:0] A_data_in_dummy, B_data_in_dummy;

  wire [5:0] A_index_tpu, A_index_tpu2;
  wire [5:0] B_index_tpu, B_index_tpu2;
  wire [5:0] C_index_tpu, C_index_tpu2;

  reg in_valid, in_valid2;
  reg [8:0] input_offset_reg;




  TPU TPU1(
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

  wire [5:0] B_index_dummy;
  TPU TPU2(
    .clk            (clk),
    .rst_n          (reset),
    .in_valid       (in_valid2),
    .input_offset   (input_offset),
    .K              (5'd16),
    .M              (5'd16),
    .N              (5'd16),
    .busy           (busy2),

    .A_wr_en        (A_wr_en_dummy),
    .A_index        (A_index_tpu2),
    .A_data_in      (A_data_in_dummy),
    .A_data_out     (A_data_out2),  

    .B_wr_en        (B_wr_en_dummy),
    .B_index        (B_index_tpu2),
    .B_data_in      (B_data_in_dummy),
    .B_data_out     (B_data_out2),

    .C_wr_en        (C_wr_en2),
    .C_index        (C_index_tpu2),
    .C_data_in      (C_data_in2)
);

  assign cmd_ready = ~rsp_valid;

  reg is_run, A_wr_en_reg, A_wr_en_reg2, B_wr_en_reg, B_wr_en_reg2;
  reg [31:0] input_0_reg, input_1_reg;
  reg [9:0] func_id_reg;
  
  reg signed [31 : 0] AddFunc_input_offset, AddFunc_filter_offset, AddFunc_output_offset;
  reg [31 : 0] AddFunc_left_shift;
  reg signed [31 : 0] AddFunc_input_number, AddFunc_filter_number;
  reg signed [31 : 0] AddFunc_shifted_input1_val, AddFunc_shifted_input2_val;
  reg signed [63 : 0] AddFunc_input1_multiplier, AddFunc_input2_multiplier, AddFunc_output_multiplier;
  reg signed [31 : 0] AddFunc_input1_shift, AddFunc_input2_shift, AddFunc_output_shift;
  reg signed [31 : 0] AddFunc_input_scaled, AddFunc_filter_scaled;
  wire signed [63 : 0] AddFunc_round_1, AddFunc_round_2, AddFunc_round_3;
  wire signed [63 : 0] AddFunc_round_shift_1, AddFunc_round_shift_2, AddFunc_round_shift_3;
  wire signed [63 : 0] AddFunc_nedge_1, AddFunc_nedge_2, AddFunc_nedge_3;
  wire signed [31 : 0] AddFunc_raw_sum;


  assign AddFunc_nedge_1 = (AddFunc_round_1 >= 64'd0) ? (64'd1 <<< (~AddFunc_input1_shift))
                                      : 1 - (64'd1 <<< (-1 * AddFunc_input1_shift - 1));
  assign AddFunc_nedge_2 = (AddFunc_round_2 >= 64'd0) ? (64'd1 <<< (~AddFunc_input2_shift))
                                      : 1 - (64'd1 <<< (-1 * AddFunc_input2_shift - 1));
  assign AddFunc_nedge_3 =
             (AddFunc_round_3 >= 64'd0)
                 ? (64'd1 <<< (~$signed(input_1_reg)))
                 : 1 - (64'd1 <<< (~$signed(input_1_reg)));

  assign AddFunc_round_1 = ((AddFunc_input1_multiplier * AddFunc_input_scaled) >>> 32'd31);
  assign AddFunc_round_2 = ((AddFunc_input2_multiplier * AddFunc_filter_scaled) >>> 32'd31);
  assign AddFunc_round_3 = (($signed(input_0_reg) * AddFunc_raw_sum) >>> 32'd31);

  assign AddFunc_round_shift_1 = (AddFunc_round_1 + AddFunc_nedge_1) >>> (~AddFunc_input1_shift + 1);
  assign AddFunc_round_shift_2 = (AddFunc_round_2 + AddFunc_nedge_2) >>> (~AddFunc_input2_shift + 1);
  assign AddFunc_round_shift_3 =
             (AddFunc_round_3 + AddFunc_nedge_3) >>> (~$signed(input_1_reg)+1);

  assign AddFunc_raw_sum = AddFunc_round_shift_1[31 : 0] + AddFunc_round_shift_2[31 : 0];


  always @(posedge clk) begin
    if (reset) begin
      rsp_valid <= 1'b0;

      rsp_payload_outputs_0 <= 32'b0;
      in_valid <= 1'b0;
      in_valid2 <= 1'b0;

      input_0_reg <= 31'b0;
      input_1_reg <= 31'b0;
      func_id_reg <= 10'b0;

      A_wr_en_reg <= 1'b0;
      A_wr_en_reg2 <= 1'b0;
      B_wr_en_reg <= 1'b0;
      B_wr_en_reg2 <= 1'b0;

      AddFunc_input_offset <= 32'b0;
      AddFunc_filter_offset <= 32'b0;
      AddFunc_output_offset <= 32'b0;
      AddFunc_left_shift <= 32'b0;
      AddFunc_input_number <= 32'b0;
      AddFunc_filter_number <= 32'b0;
      AddFunc_shifted_input1_val <= 32'b0;
      AddFunc_shifted_input2_val <= 32'b0;
      AddFunc_input1_multiplier <= 64'b0;
      AddFunc_input2_multiplier <= 64'b0;
      AddFunc_output_multiplier <= 64'b0;
      AddFunc_input1_shift <= 32'b0;
      AddFunc_input2_shift <= 32'b0;
      AddFunc_output_shift <= 32'b0;
      AddFunc_input_scaled <= 32'b0;
      AddFunc_filter_scaled <= 32'b0;

      tpu1_addr <= 0;
      tpu2_addr <= 0;

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
          tpu1_addr <= tpu1_addr + 1;
        end
        7'd1: begin
          rsp_valid <= 1'b1;
          is_run <= 1'b0;
          A_wr_en_reg2 <= 1'b0;
          B_wr_en_reg2 <= 1'b0;
          tpu2_addr <= tpu2_addr + 1;
        end
        7'd2: begin
          rsp_valid <= (busy);
          is_run <= ~(busy);
          in_valid <= ~(busy); 
        end
        7'd3: begin
          rsp_valid <= (busy2);
          is_run <= ~(busy2);
          in_valid2 <= ~(busy2); 
        end
        7'd4: begin
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
        7'd5: begin
          if (busy2) begin
            rsp_valid <= 1'b0;
            is_run <= 1'b1;
          end
          else begin
            rsp_valid <= 1'b1;
            is_run <= 1'b0;
            rsp_payload_outputs_0 <= C_data_sel_out2;
          end
        end
        7'd6: begin
          rsp_valid <= 1'b1;
          is_run <= 1'b0;
          AddFunc_input_offset <= $signed(input_0_reg);
          AddFunc_filter_offset <= $signed(input_1_reg);
        end
        7'd7: begin
          rsp_valid <= 1'b1;
          is_run <= 1'b0;
          AddFunc_left_shift <= input_0_reg;
          AddFunc_output_offset <= $signed(input_1_reg);
        end
        7'd8: begin
          rsp_valid <= 1'b0;
          is_run <= 1'b1;
          AddFunc_input_number <= $signed(input_0_reg);
          AddFunc_filter_number <= $signed(input_1_reg);
          AddFunc_input_scaled <= ($signed(input_0_reg) + AddFunc_input_offset)
                          <<< AddFunc_left_shift;
          AddFunc_filter_scaled <= ($signed(input_1_reg) + AddFunc_filter_offset)
                           <<< AddFunc_left_shift;
        end
        7'd9: begin
          rsp_valid <= 1'b1;
          is_run <= 1'b0;
          AddFunc_input1_multiplier <= $signed(input_0_reg);
          AddFunc_input1_shift <= $signed(input_1_reg);
          rsp_payload_outputs_0 <= $signed(input_1_reg);  //
        end
        7'd10: begin
          rsp_valid <= 1'b1;
          is_run <= 1'b0;
          AddFunc_input2_multiplier <= $signed(input_0_reg);
          AddFunc_input2_shift <= $signed(input_1_reg);
          rsp_payload_outputs_0 <= $signed(input_1_reg);  //
        end
        7'd11: begin
          rsp_valid <= 1'b1;
          is_run <= 1'b0;
          AddFunc_output_multiplier <= $signed(input_0_reg);
          AddFunc_output_shift <= $signed(input_1_reg);
          rsp_payload_outputs_0 <= AddFunc_round_shift_3[31 : 0] + AddFunc_output_offset;
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
          is_run <= 1'b1;
          A_wr_en_reg2 <= 1'b1;
          B_wr_en_reg2 <= 1'b1;
        end
        7'd2: begin
          rsp_valid <= 1'b0;
          in_valid <= 1'b1;
          is_run <= 1'b1;
          input_offset_reg <= cmd_payload_inputs_0[8:0];
        end
        7'd3: begin
          rsp_valid <= 1'b0;
          in_valid2 <= 1'b1;
          is_run <= 1'b1;
          input_offset_reg <= cmd_payload_inputs_0[8:0];
        end
        7'd4: begin
          rsp_valid <= 1'b0;
          is_run <= 1'b1;
          tpu1_addr <= 0;
        end 
        7'd5: begin
          rsp_valid <= 1'b0;
          is_run <= 1'b1;
          tpu2_addr <= 0;
        end

        7'd6: begin
          rsp_valid <= 1'b0;
          is_run <= 1'b1;
        end
        7'd7: begin
          rsp_valid <= 1'b0;
          is_run <= 1'b1;
        end
        7'd8: begin
          rsp_valid <= 1'b1;
          is_run <= 1'b0;
        end
        7'd9: begin
          rsp_valid <= 1'b0;
          is_run <= 1'b1;
        end
        7'd10: begin
          rsp_valid <= 1'b0;
          is_run <= 1'b1;
        end
        7'd11: begin
          rsp_valid <= 1'b0;
          is_run <= 1'b1;
        end
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
