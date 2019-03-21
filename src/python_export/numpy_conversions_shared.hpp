/*
 * numpy_conversions_shared.hpp
 *
 *  Created on: Feb 5, 2019
 *      Author: Gregory Kramida
 *   Copyright: 2019 Gregory Kramida
 *
 *   Licensed under the Apache License, Version 2.0 (the "License");
 *   you may not use this file except in compliance with the License.
 *   You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 *   Unless required by applicable law or agreed to in writing, software
 *   distributed under the License is distributed on an "AS IS" BASIS,
 *   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *   See the License for the specific language governing permissions and
 *   limitations under the License.
 */

#pragma once

//standard library
#include <algorithm>

//libraries
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/ndarraytypes.h>

//local
#include "../math/tensors.hpp"
#include "../math/typedefs.hpp"

template<typename SCALAR>
	struct NumpyEquivalentType {
};



template<>
struct NumpyEquivalentType<double> {
	enum {
		type_code = NPY_DOUBLE
	};
};
template<>
struct NumpyEquivalentType<float> {
	enum {
		type_code = NPY_FLOAT
	};
};
template<>
struct NumpyEquivalentType<int> {
	enum {
		type_code = NPY_INT
	};
};
template<>
struct NumpyEquivalentType<unsigned short>{
	enum {
		type_code = NPY_USHORT
	};
};
template<>
struct NumpyEquivalentType<unsigned char>{
	enum {
		type_code = NPY_UBYTE
	};
};
template<>
struct NumpyEquivalentType<math::Vector2<float>> {
	enum {
		type_code = NPY_FLOAT
	};
};
template<>
struct NumpyEquivalentType<math::Matrix2<float>> {
	enum {
		type_code = NPY_FLOAT
	};
};
template<>
struct NumpyEquivalentType<std::complex<double> > {
	enum {
		type_code = NPY_CDOUBLE
	};
};

template<typename SourceType, typename DestType>
static void copy_array(const SourceType* source, DestType* dest,
		const npy_int& nb_rows, const npy_int& nb_cols,
		const bool& isSourceTypeNumpy = false, const bool& isDestRowMajor = true,
		const bool& isSourceRowMajor = true,
		const npy_int& numpy_row_stride = 1, const npy_int& numpy_col_stride = 1) {
	// determine source strides
	int row_stride = 1, col_stride = 1;
	if (isSourceTypeNumpy) {
		row_stride = numpy_row_stride;
		col_stride = numpy_col_stride;
	} else {
		if (isSourceRowMajor) {
			row_stride = nb_cols;
		} else {
			col_stride = nb_rows;
		}
	}

	if (isDestRowMajor) {
		for (int r = 0; r < nb_rows; r++) {
			for (int c = 0; c < nb_cols; c++) {
				*dest = source[r * row_stride + c * col_stride];
				dest++;
			}
		}
	} else {
		for (int c = 0; c < nb_cols; c++) {
			for (int r = 0; r < nb_rows; r++) {
				*dest = source[r * row_stride + c * col_stride];
				dest++;
			}
		}
	}
}

//Assumes row-major destination
template<typename SourceType, typename DestType>
static void copy_tensor(
		const SourceType* source, DestType* dest,
		const int& num_dimensions,
		const npy_intp* shape,
		const int& size,
		const bool& is_source_row_major = false) {
	if (is_source_row_major) {
		for(int i_element = 0; i_element < size; i_element++,dest++){
			*dest = source[i_element];
		}
		return;
	}

	int col_stride = shape[0];
	std::vector<int> strides;
	std::vector<int> remaining_dims;
	int cumulative_stride = shape[0] * shape[1];
	int chunk_size = 1;

	for (int ix_dimension = 2; ix_dimension < num_dimensions; ix_dimension++) {
		int dim = shape[ix_dimension];
		strides.push_back(cumulative_stride);
		remaining_dims.push_back(dim);
		cumulative_stride *= dim;
		chunk_size *= dim;
	}

	std::reverse(remaining_dims.begin(), remaining_dims.end());
	std::reverse(strides.begin(), strides.end());

	auto proces_chunk = [&](int row_and_col_offset) {
		for (int ix_element = 0; ix_element < chunk_size; ix_element++) {
			int ix_subelement = ix_element;
			int remaining_offset = 0;
			for (size_t ix_dim = 0; ix_dim < remaining_dims.size(); ix_dim++) {
				div_t division_result = div(ix_subelement, remaining_dims[ix_dim]);
				ix_subelement = division_result.quot;
				int coord = division_result.rem;
				remaining_offset += coord * strides[ix_dim];
			}
			*dest = source[row_and_col_offset + remaining_offset];
			dest++;
		}
	};
	for (int r = 0; r < shape[0]; r++) {
		for (int c = 0; c < shape[1]; c++) {
			int row_and_col_offset = r + c * col_stride;
			proces_chunk(row_and_col_offset);
		}
	}
}
