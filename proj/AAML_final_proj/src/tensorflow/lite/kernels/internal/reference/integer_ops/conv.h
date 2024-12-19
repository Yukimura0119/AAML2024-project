/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

	http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/
#ifndef TENSORFLOW_LITE_KERNELS_INTERNAL_REFERENCE_INTEGER_OPS_CONV_H_
#define TENSORFLOW_LITE_KERNELS_INTERNAL_REFERENCE_INTEGER_OPS_CONV_H_

#include <algorithm>

#include "cfu.h"
#include "perf.h"
#include "tensorflow/lite/kernels/internal/common.h"
#include "tensorflow/lite/kernels/internal/portable_tensor_utils.h"
#include <cstring>
#include <stdio.h>

namespace tflite {
namespace reference_integer_ops {

// Fixed-point per-channel-quantization convolution reference kernel.
inline void
ConvPerChannel(const ConvParams &params, const int32_t *output_multiplier,
			   const int32_t *output_shift, const RuntimeShape &input_shape,
			   const int8_t *input_data, const RuntimeShape &filter_shape,
			   const int8_t *filter_data, const RuntimeShape &bias_shape,
			   const int32_t *bias_data, const RuntimeShape &output_shape,
			   int8_t *output_data) {
	// Get parameters.
	const int stride_length = params.stride_width;
	const int pad_length = params.padding_values.width;
	const int32_t output_offset = params.output_offset;

	// Check dimensions of the tensors.
	const int input_length = input_shape.Dims(1);
	const int filter_num = filter_shape.Dims(0);
	const int filter_length = filter_shape.Dims(1);
	const int filter_input_depth = filter_shape.Dims(3);

	int8_t im2col[576][1024];
	int8_t kernel[1024][64];
	int32_t matmul[16][16];

	int patch_num = 0;
	for (int iy = -pad_length; iy < input_length + pad_length;
		 iy += stride_length) {
		int posx, posy;
		for (int ix = -pad_length; ix < input_length + pad_length;
			 ix += stride_length) {
			int patch_idx = 0;
			for (int fy = 0; fy < filter_length; fy++) {
				for (int fx = 0; fx < filter_length; fx++) {
					posx = ix + fx;
					posy = iy + fy;
					int offset =
						(posy * input_length + posx) * filter_input_depth;
					for (int k = 0; k < filter_input_depth; k++) {
						if (posx >= 0 && posx < input_length && posy >= 0 &&
							posy < input_length) {
							im2col[patch_idx][patch_num] =
								input_data[offset + k];
						} else [[unlikely]] {
							im2col[patch_idx][patch_num] = -128;
						}
						patch_idx++;
					}
				}
			}
			patch_num++;
			if (posx >= input_length + pad_length - 1) [[unlikely]]
				break;
		}
		if (posy >= input_length + pad_length - 1) [[unlikely]]
			break;
	}

	constexpr int tile_size = 16;
	int pad_K = ((filter_num + tile_size - 1) >> 4) << 4;
	int kernel_size = 0;
	for (int fy = 0; fy < filter_length; fy++) {
		for (int fx = 0; fx < filter_length; fx++) {
			for (int ic = 0; ic < filter_input_depth; ic++) {
				for (int oc = 0; oc < pad_K; oc++) {
					kernel[kernel_size][oc] =
						(oc < filter_num)
							? filter_data[Offset(filter_shape, oc, fy, fx, ic)]
							: 0;
				}
				kernel_size++;
			}
		}
	}

	int M = patch_num;	 // H x W
	int N = kernel_size; // K x K x C

	int pad_M = ((M + tile_size - 1) >> 4) << 4;
	int pad_N = ((N + tile_size - 1) >> 4) << 4;

	for (int i = kernel_size; i < pad_N; i++) {
		for (int j = 0; j < pad_K; j++) {
			kernel[i][j] = 0;
		}
	}

	for (int m = 0; m < pad_M; m += tile_size) {
		for (int k = 0; k < pad_K; k += tile_size) {
			for (int mx = 0; mx < 16; mx++) {
				for (int my = 0; my < 16; my++) {
					matmul[mx][my] = 0;
				}
			}
			for (int n = 0; n < pad_N; n += tile_size) {
				for (int i = 0; i < tile_size; i += 4) {
					for (int j = 0; j < tile_size; j += 4) {
						for (int r = 0; r < 4; r++) {
							uint32_t data1 =
								*((uint32_t *)(im2col[n + j + r] + m + i));
							uint32_t data2 =
								*((uint32_t *)(kernel[n + j + r] + k + i));
							cfu_op0(0, data1, data2);
						}
					}
				}
				cfu_op0(1, 128, 0);

				for (int i = 0; i < 16; i++) {
					for (int j = 0; j < 16; j++) {
						int32_t data = cfu_op0(2, ((j >> 2) << 4) + i, j & 0x3);
						matmul[i][j] += data;
					}
				}
			}
			for (int i = 0; i < tile_size; i++) {
				int offset = (((m + i) / patch_num) * output_shape.Dims(1) +
							  ((m + i) % patch_num)) *
							 output_shape.Dims(3);
				for (int j = 0; j < tile_size; j++) {
					matmul[i][j] += bias_data[k + j];
					matmul[i][j] = MultiplyByQuantizedMultiplier(
						matmul[i][j], output_multiplier[k + j],
						output_shift[k + j]);
					matmul[i][j] += output_offset;
					matmul[i][j] = std::max(matmul[i][j], (int32_t)-128);
					matmul[i][j] = std::min(matmul[i][j], (int32_t)127);
					output_data[offset + (k + j)] =
						static_cast<int8_t>(matmul[i][j]);
				}
			}
		}
	}
}

inline void ConvPerChannelWithPackedInt4Weights(
	const ConvParams &params, const int32_t *output_multiplier,
	const int32_t *output_shift, const RuntimeShape &input_shape,
	const int8_t *input_data, const RuntimeShape &filter_shape,
	const int8_t *filter_input, int8_t *unpacked_filter_data,
	const RuntimeShape &bias_shape, const int32_t *bias_data,
	const RuntimeShape &output_shape, int8_t *output_data) {
	TFLITE_DCHECK(unpacked_filter_data != nullptr);
	tflite::tensor_utils::UnpackDenseInt4IntoInt8(
		filter_input, filter_shape.FlatSize(), unpacked_filter_data);
	ConvPerChannel(params, output_multiplier, output_shift, input_shape,
				   input_data, filter_shape, unpacked_filter_data, bias_shape,
				   bias_data, output_shape, output_data);
}

// Fixed-point per-channel-quantization convolution reference kernel.
// 16-bit data and 8-bit filter
template <typename AccumScalar>
inline void
ConvPerChannel(const ConvParams &params, const int32_t *output_multiplier,
			   const int32_t *output_shift, const RuntimeShape &input_shape,
			   const int16_t *input_data, const RuntimeShape &filter_shape,
			   const int8_t *filter_data, const RuntimeShape &bias_shape,
			   const AccumScalar *bias_data, const RuntimeShape &output_shape,
			   int16_t *output_data) {
	// Get parameters.
	const int stride_width = params.stride_width;
	const int stride_height = params.stride_height;
	const int dilation_width_factor = params.dilation_width_factor;
	const int dilation_height_factor = params.dilation_height_factor;
	const int pad_width = params.padding_values.width;
	const int pad_height = params.padding_values.height;

	// Set min and max value of the output.
	const int32_t output_activation_min = params.quantized_activation_min;
	const int32_t output_activation_max = params.quantized_activation_max;

	// Consistency check.
	TFLITE_DCHECK_LE(output_activation_min, output_activation_max);
	TFLITE_DCHECK_EQ(input_shape.DimensionsCount(), 4);
	TFLITE_DCHECK_EQ(filter_shape.DimensionsCount(), 4);
	TFLITE_DCHECK_EQ(output_shape.DimensionsCount(), 4);
	const int batches = MatchingDim(input_shape, 0, output_shape, 0);
	const int input_depth = input_shape.Dims(3);
	const int output_depth = MatchingDim(filter_shape, 0, output_shape, 3);
	if (bias_data) {
		TFLITE_DCHECK_EQ(bias_shape.FlatSize(), output_depth);
	}

	// Check dimensions of the tensors.
	const int input_height = input_shape.Dims(1);
	const int input_width = input_shape.Dims(2);
	const int filter_height = filter_shape.Dims(1);
	const int filter_width = filter_shape.Dims(2);
	const int filter_input_depth = filter_shape.Dims(3);
	const int groups = input_depth / filter_input_depth;
	TFLITE_DCHECK_EQ(input_depth % filter_input_depth, 0);
	const int filters_per_group = output_depth / groups;
	const int output_height = output_shape.Dims(1);
	const int output_width = output_shape.Dims(2);
	for (int batch = 0; batch < batches; ++batch) {
		for (int out_y = 0; out_y < output_height; ++out_y) {
			const int in_y_origin = (out_y * stride_height) - pad_height;
			for (int out_x = 0; out_x < output_width; ++out_x) {
				const int in_x_origin = (out_x * stride_width) - pad_width;
				for (int out_channel = 0; out_channel < output_depth;
					 ++out_channel) {
					auto group = out_channel / filters_per_group;
					AccumScalar acc = 0;
					for (int filter_y = 0; filter_y < filter_height;
						 ++filter_y) {
						const int in_y =
							in_y_origin + dilation_height_factor * filter_y;
						for (int filter_x = 0; filter_x < filter_width;
							 ++filter_x) {
							const int in_x =
								in_x_origin + dilation_width_factor * filter_x;

							// Zero padding by omitting the areas outside the
							// image.
							const bool is_point_inside_image =
								(in_x >= 0) && (in_x < input_width) &&
								(in_y >= 0) && (in_y < input_height);

							if (!is_point_inside_image) {
								continue;
							}

							for (int in_channel = 0;
								 in_channel < filter_input_depth;
								 ++in_channel) {
								int32_t input_val = input_data[Offset(
									input_shape, batch, in_y, in_x,
									in_channel + group * filter_input_depth)];
								int32_t filter_val = filter_data[Offset(
									filter_shape, out_channel, filter_y,
									filter_x, in_channel)];
								// Accumulate with 64 bits accumulator.
								// int64_t += int8_t * int16_t so the highest
								// value we can get from each accumulation is
								// [-127, 127] * ([-32768, 32767] -
								// [-32768, 32767]), which is [-8322945,
								// 8322945]. log2(8322945) = 22.99.
								acc += filter_val * input_val;
							}
						}
					}
					if (bias_data) {
						acc += bias_data[out_channel];
					}
					int32_t scaled_acc = MultiplyByQuantizedMultiplier(
						acc, output_multiplier[out_channel],
						output_shift[out_channel]);
					scaled_acc = std::max(scaled_acc, output_activation_min);
					scaled_acc = std::min(scaled_acc, output_activation_max);
					output_data[Offset(output_shape, batch, out_y, out_x,
									   out_channel)] =
						static_cast<int16_t>(scaled_acc);
				}
			}
		}
	}
}

} // namespace reference_integer_ops
} // namespace tflite

#endif // TENSORFLOW_LITE_KERNELS_INTERNAL_REFERENCE_INTEGER_OPS_CONV_H_