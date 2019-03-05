//  ================================================================
//  Created by Gregory Kramida on 11/3/18.
//  Copyright (c) 2018 Gregory Kramida
//  Licensed under the Apache License, Version 2.0 (the "License");
//  you may not use this file except in compliance with the License.
//  You may obtain a copy of the License at

//  http://www.apache.org/licenses/LICENSE-2.0

//  Unless required by applicable law or agreed to in writing, software
//  distributed under the License is distributed on an "AS IS" BASIS,
//  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
//  See the License for the specific language governing permissions and
//  limitations under the License.
//  ================================================================

#include "convolution.hpp"
#include "typedefs.hpp"

namespace math {

template<typename ElementType>
inline static ElementType buffer_convolve_helper_preserve_zeros(ElementType* buffer,
		const eig::VectorXf& kernel_1d, int buffer_read_index,
		int kernel_size,
		const ElementType& original_vector) {
	if (original_vector.is_zero()) {
		return {0.0f, 0.0f};
	}
	int i_kernel_value = 0;
	float x, y;
	x = y = 0.0f;
	for (int i_buffered_vector = buffer_read_index;
			i_buffered_vector < kernel_size; i_buffered_vector++, i_kernel_value++) {
		float kernel_value = kernel_1d(i_kernel_value);
		ElementType& buffered_vector = buffer[i_buffered_vector];
		x += buffered_vector.x * kernel_value;
		y += buffered_vector.y * kernel_value;
	}
	for (int i_buffered_vector = 0; i_buffered_vector < buffer_read_index; i_buffered_vector++, i_kernel_value++) {
		float kernel_value = kernel_1d(i_kernel_value);
		ElementType& buffered_vector = buffer[i_buffered_vector];
		x += buffered_vector.x * kernel_value;
		y += buffered_vector.y * kernel_value;
	}
	return {x, y};
}

template<typename ElementType>
inline static ElementType buffer_convolve_helper(ElementType* buffer,
		const eig::VectorXf& kernel_1d, int buffer_read_index,
		int kernel_size) {
	int i_kernel_value = 0;
	ElementType element(0.0f);
	for (int i_buffered_vector = buffer_read_index;
			i_buffered_vector < kernel_size; i_buffered_vector++, i_kernel_value++) {
		float kernel_value = kernel_1d(i_kernel_value);
		ElementType& buffered_vector = buffer[i_buffered_vector];
		element += ElementType(buffered_vector) * kernel_value;
	}
	for (int i_buffered_vector = 0; i_buffered_vector < buffer_read_index; i_buffered_vector++, i_kernel_value++) {
		float kernel_value = kernel_1d(i_kernel_value);
		ElementType& buffered_vector = buffer[i_buffered_vector];
		element += ElementType(buffered_vector) * kernel_value;
	}
	return element;
}

void convolve_with_kernel_preserve_zeros(MatrixXv2f& field, const eig::VectorXf& kernel_1d) {
	eig::Index row_count = field.rows();
	eig::Index column_count = field.cols();

	eig::VectorXf kernel_inverted(kernel_1d.size());

	//flip kernel, see def of discrete convolution on Wikipedia
	for (eig::Index i_kernel_element = 0; i_kernel_element < kernel_1d.size(); i_kernel_element++) {
		kernel_inverted(kernel_1d.size() - i_kernel_element - 1) = kernel_1d(i_kernel_element);
	}

	int kernel_size = static_cast<int>(kernel_inverted.size());
	int kernel_half_size = kernel_size / 2;

	math::Vector2f buffer[kernel_size];
	MatrixXv2f y_convolved = MatrixXv2f::Zero(field.rows(), field.cols());

#pragma omp parallel for private(buffer)
	for (eig::Index i_col = 0; i_col < column_count; i_col++) {
		int i_buffer_write_index = 0;
		for (; i_buffer_write_index < kernel_half_size; i_buffer_write_index++) {
			buffer[i_buffer_write_index] = math::Vector2f(0.0f); // fill buffer with empty value
		}
		eig::Index i_row_to_sample = 0;
		//fill buffer up to the last value
		for (; i_row_to_sample < kernel_half_size; i_row_to_sample++, i_buffer_write_index++) {
			buffer[i_buffer_write_index] = field(i_row_to_sample, i_col);
		}
		int i_buffer_read_index = 0;
		eig::Index i_row_to_write = 0;
		for (; i_row_to_sample < row_count; i_row_to_write++, i_row_to_sample++,
				i_buffer_write_index = i_buffer_read_index) {
			buffer[i_buffer_write_index] = field(i_row_to_sample, i_col); // fill buffer with next value
			i_buffer_read_index = (i_buffer_write_index + 1) % kernel_size;
			y_convolved(i_row_to_write, i_col) =
					buffer_convolve_helper_preserve_zeros(buffer, kernel_inverted, i_buffer_read_index,
							kernel_size, field(i_row_to_write, i_col));
		}
		for (; i_row_to_write < row_count; i_row_to_write++, i_buffer_write_index = i_buffer_read_index) {
			buffer[i_buffer_write_index] = math::Vector2f(0.0f);
			i_buffer_read_index = (i_buffer_write_index + 1) % kernel_size;
			y_convolved(i_row_to_write, i_col) =
					buffer_convolve_helper_preserve_zeros(buffer, kernel_inverted, i_buffer_read_index,
							kernel_size, field(i_row_to_write, i_col));
		}
	}

#pragma omp parallel for private(buffer)
	for (eig::Index i_row = 0; i_row < row_count; i_row++) {
		int i_buffer_write_index = 0;
		for (; i_buffer_write_index < kernel_half_size; i_buffer_write_index++) {
			buffer[i_buffer_write_index] = math::Vector2f(0.0f); // fill buffer with empty value
		}
		eig::Index i_col_to_sample = 0;
		//fill buffer up to the last value
		for (; i_col_to_sample < kernel_half_size; i_col_to_sample++, i_buffer_write_index++) {
			buffer[i_buffer_write_index] = y_convolved(i_row, i_col_to_sample);
		}
		int i_buffer_read_index = 0;
		eig::Index i_col_to_write = 0;
		for (; i_col_to_sample < column_count; i_col_to_write++, i_col_to_sample++,
				i_buffer_write_index = i_buffer_read_index) {
			buffer[i_buffer_write_index] = y_convolved(i_row, i_col_to_sample); // fill buffer with next value
			i_buffer_read_index = (i_buffer_write_index + 1) % kernel_size;
			field(i_row, i_col_to_write) =
					buffer_convolve_helper_preserve_zeros(buffer, kernel_inverted, i_buffer_read_index,
							kernel_size, y_convolved(i_row, i_col_to_write));
		}
		for (; i_col_to_write < column_count; i_col_to_write++, i_buffer_write_index = i_buffer_read_index) {
			buffer[i_buffer_write_index] = math::Vector2f(0.0f);
			i_buffer_read_index = (i_buffer_write_index + 1) % kernel_size;
			field(i_row, i_col_to_write) =
					buffer_convolve_helper_preserve_zeros(buffer, kernel_inverted, i_buffer_read_index,
							kernel_size, y_convolved(i_row, i_col_to_write));
		}
	}
}

void convolve_with_kernel(MatrixXv2f& field, const eig::VectorXf& kernel_1d) {
	eig::Index row_count = field.rows();
	eig::Index column_count = field.cols();

	eig::VectorXf kernel_inverted(kernel_1d.size());

	//flip kernel, see def of discrete convolution on Wikipedia
	for (eig::Index i_kernel_element = 0; i_kernel_element < kernel_1d.size(); i_kernel_element++) {
		kernel_inverted(kernel_1d.size() - i_kernel_element - 1) = kernel_1d(i_kernel_element);
	}

	int kernel_size = static_cast<int>(kernel_inverted.size());
	int kernel_half_size = kernel_size / 2;

	math::Vector2f buffer[kernel_size];
	MatrixXv2f y_convolved = MatrixXv2f::Zero(field.rows(), field.cols());

#pragma omp parallel for private(buffer)
	for (eig::Index i_col = 0; i_col < column_count; i_col++) {
		int i_buffer_write_index = 0;
		for (; i_buffer_write_index < kernel_half_size; i_buffer_write_index++) {
			buffer[i_buffer_write_index] = math::Vector2f(0.0f); // fill buffer with empty value
		}
		eig::Index i_row_to_sample = 0;
		//fill buffer up to the last value
		for (; i_row_to_sample < kernel_half_size; i_row_to_sample++, i_buffer_write_index++) {
			buffer[i_buffer_write_index] = field(i_row_to_sample, i_col);
		}
		int i_buffer_read_index = 0;
		eig::Index i_row_to_write = 0;
		for (; i_row_to_sample < row_count; i_row_to_write++, i_row_to_sample++,
				i_buffer_write_index = i_buffer_read_index) {
			buffer[i_buffer_write_index] = field(i_row_to_sample, i_col); // fill buffer with next value
			i_buffer_read_index = (i_buffer_write_index + 1) % kernel_size;
			y_convolved(i_row_to_write, i_col) =
					buffer_convolve_helper(buffer, kernel_inverted, i_buffer_read_index, kernel_size);
		}
		for (; i_row_to_write < row_count; i_row_to_write++, i_buffer_write_index = i_buffer_read_index) {
			buffer[i_buffer_write_index] = math::Vector2f(0.0f);
			i_buffer_read_index = (i_buffer_write_index + 1) % kernel_size;
			y_convolved(i_row_to_write, i_col) =
					buffer_convolve_helper(buffer, kernel_inverted, i_buffer_read_index, kernel_size);
		}
	}

#pragma omp parallel for private(buffer)
	for (eig::Index i_row = 0; i_row < row_count; i_row++) {
		int i_buffer_write_index = 0;
		for (; i_buffer_write_index < kernel_half_size; i_buffer_write_index++) {
			buffer[i_buffer_write_index] = math::Vector2f(0.0f); // fill buffer with empty value
		}
		eig::Index i_col_to_sample = 0;
		//fill buffer up to the last value
		for (; i_col_to_sample < kernel_half_size; i_col_to_sample++, i_buffer_write_index++) {
			buffer[i_buffer_write_index] = y_convolved(i_row, i_col_to_sample);
		}
		int i_buffer_read_index = 0;
		eig::Index i_col_to_write = 0;
		for (; i_col_to_sample < column_count; i_col_to_write++, i_col_to_sample++,
				i_buffer_write_index = i_buffer_read_index) {
			buffer[i_buffer_write_index] = y_convolved(i_row, i_col_to_sample); // fill buffer with next value
			i_buffer_read_index = (i_buffer_write_index + 1) % kernel_size;
			field(i_row, i_col_to_write) =
					buffer_convolve_helper(buffer, kernel_inverted, i_buffer_read_index, kernel_size);
		}
		for (; i_col_to_write < column_count; i_col_to_write++, i_buffer_write_index = i_buffer_read_index) {
			buffer[i_buffer_write_index] = math::Vector2f(0.0f);
			i_buffer_read_index = (i_buffer_write_index + 1) % kernel_size;
			field(i_row, i_col_to_write) =
					buffer_convolve_helper(buffer, kernel_inverted, i_buffer_read_index, kernel_size);
		}
	}
}

void convolve_with_kernel(Tensor3v3f& field, const eig::VectorXf& kernel_1d) {
	int x_size = field.dimension(0);
	int y_size = field.dimension(1);
	int z_size = field.dimension(2);

	eig::VectorXf kernel_inverted(kernel_1d.size());

	//flip kernel, see def of discrete convolution on Wikipedia
	for (eig::Index i_kernel_element = 0; i_kernel_element < kernel_1d.size(); i_kernel_element++) {
		kernel_inverted(kernel_1d.size() - i_kernel_element - 1) = kernel_1d(i_kernel_element);
	}

	int kernel_size = static_cast<int>(kernel_inverted.size());
	int kernel_half_size = kernel_size / 2;

	math::Vector3f buffer[kernel_size];
	math::Tensor3v3f x_convolved = math::Tensor3v3f(x_size, y_size, z_size);
	x_convolved.setZero();

#pragma omp parallel for private(buffer)
	for (int z = 0; z < z_size; z++) {
		for (int y = 0; y < y_size; y++) {
			int i_buffer_write_index = 0;
			for (; i_buffer_write_index < kernel_half_size; i_buffer_write_index++) {
				buffer[i_buffer_write_index] = math::Vector3f(0.0f); // fill buffer with empty value
			}
			int x_to_sample = 0;
			//fill buffer up to the last value
			for (; x_to_sample < kernel_half_size; x_to_sample++, i_buffer_write_index++) {
				buffer[i_buffer_write_index] = field(x_to_sample, y, z);
			}
			int i_buffer_read_index = 0;
			int x_to_write = 0;
			for (; x_to_sample < x_size; x_to_write++, x_to_sample++,
					i_buffer_write_index = i_buffer_read_index) {
				buffer[i_buffer_write_index] = field(x_to_sample, y, z); // fill buffer with next value
				i_buffer_read_index = (i_buffer_write_index + 1) % kernel_size;
				x_convolved(x_to_write, y, z) =
						buffer_convolve_helper(buffer, kernel_inverted, i_buffer_read_index, kernel_size);
			}
			for (; x_to_write < x_size; x_to_write++, i_buffer_write_index = i_buffer_read_index) {
				buffer[i_buffer_write_index] = math::Vector3f(0.0f);
				i_buffer_read_index = (i_buffer_write_index + 1) % kernel_size;
				x_convolved(x_to_write, y, z) =
						buffer_convolve_helper(buffer, kernel_inverted, i_buffer_read_index, kernel_size);
			}
		}
	}

	math::Tensor3v3f y_convolved = math::Tensor3v3f(x_size, y_size, z_size);

#pragma omp parallel for private(buffer)
	for (int z = 0; z < z_size; z++) {
		for (int x = 0; x < x_size; x++) {
			int i_buffer_write_index = 0;
			for (; i_buffer_write_index < kernel_half_size; i_buffer_write_index++) {
				buffer[i_buffer_write_index] = math::Vector3f(0.0f); // fill buffer with empty value
			}
			int y_to_sample = 0;
			//fill buffer up to the last value
			for (; y_to_sample < kernel_half_size; y_to_sample++, i_buffer_write_index++) {
				buffer[i_buffer_write_index] = x_convolved(x, y_to_sample, z);
			}
			int i_buffer_read_index = 0;
			int y_to_write = 0;
			for (; y_to_sample < y_size; y_to_write++, y_to_sample++,
					i_buffer_write_index = i_buffer_read_index) {
				buffer[i_buffer_write_index] = x_convolved(x, y_to_sample, z); // fill buffer with next value
				i_buffer_read_index = (i_buffer_write_index + 1) % kernel_size;
				y_convolved(x, y_to_write, z) =
						buffer_convolve_helper(buffer, kernel_inverted, i_buffer_read_index, kernel_size);
			}
			for (; y_to_write < y_size; y_to_write++, i_buffer_write_index = i_buffer_read_index) {
				buffer[i_buffer_write_index] = math::Vector3f(0.0f);
				i_buffer_read_index = (i_buffer_write_index + 1) % kernel_size;
				y_convolved(x, y_to_write, z) =
						buffer_convolve_helper(buffer, kernel_inverted, i_buffer_read_index, kernel_size);
			}
		}
	}

#pragma omp parallel for private(buffer)
	for (int y = 0; y < y_size; y++) {
		for (int x = 0; x < x_size; x++) {
			int i_buffer_write_index = 0;
			for (; i_buffer_write_index < kernel_half_size; i_buffer_write_index++) {
				buffer[i_buffer_write_index] = math::Vector3f(0.0f); // fill buffer with empty value
			}
			eig::Index z_to_sample = 0;
			//fill buffer up to the last value
			for (; z_to_sample < kernel_half_size; z_to_sample++, i_buffer_write_index++) {
				buffer[i_buffer_write_index] = y_convolved(x, y, z_to_sample);
			}
			int i_buffer_read_index = 0;
			eig::Index z_to_write = 0;
			for (; z_to_sample < z_size; z_to_write++, z_to_sample++,
					i_buffer_write_index = i_buffer_read_index) {
				buffer[i_buffer_write_index] = y_convolved(x, y, z_to_sample); // fill buffer with next value
				i_buffer_read_index = (i_buffer_write_index + 1) % kernel_size;
				field(x, y, z_to_write) =
						buffer_convolve_helper(buffer, kernel_inverted, i_buffer_read_index, kernel_size);
			}
			for (; z_to_write < z_size; z_to_write++, i_buffer_write_index = i_buffer_read_index) {
				buffer[i_buffer_write_index] = math::Vector3f(0.0f);
				i_buffer_read_index = (i_buffer_write_index + 1) % kernel_size;
				field(x, y, z_to_write) =
						buffer_convolve_helper(buffer, kernel_inverted, i_buffer_read_index, kernel_size);
			}
		}
	}
}

void convolve_with_kernel_y(MatrixXv2f& field, const eig::VectorXf& kernel_1d) {
	eig::Index row_count = field.rows();
	eig::Index column_count = field.cols();

	eig::VectorXf kernel_inverted(kernel_1d.size());

	//flip kernel, see def of discrete convolution on Wikipedia
	for (eig::Index i_kernel_element = 0; i_kernel_element < kernel_1d.size(); i_kernel_element++) {
		kernel_inverted(kernel_1d.size() - i_kernel_element - 1) = kernel_1d(i_kernel_element);
	}

	int kernel_size = static_cast<int>(kernel_inverted.size());
	int kernel_half_size = kernel_size / 2;

	math::Vector2f buffer[kernel_size];
	MatrixXv2f y_convolved = MatrixXv2f::Zero(field.rows(), field.cols());

#pragma omp parallel for private(buffer)
	for (eig::Index i_col = 0; i_col < column_count; i_col++) {
		int i_buffer_write_index = 0;
		for (; i_buffer_write_index < kernel_half_size; i_buffer_write_index++) {
			buffer[i_buffer_write_index] = math::Vector2f(0.0f); // fill buffer with empty value
		}
		eig::Index i_row_to_sample = 0;
		//fill buffer up to the last value
		for (; i_row_to_sample < kernel_half_size; i_row_to_sample++, i_buffer_write_index++) {
			buffer[i_buffer_write_index] = field(i_row_to_sample, i_col);
		}
		int i_buffer_read_index = 0;
		eig::Index i_row_to_write = 0;
		for (; i_row_to_sample < row_count; i_row_to_write++, i_row_to_sample++,
				i_buffer_write_index = i_buffer_read_index) {
			buffer[i_buffer_write_index] = field(i_row_to_sample, i_col); // fill buffer with next value
			i_buffer_read_index = (i_buffer_write_index + 1) % kernel_size;
			y_convolved(i_row_to_write, i_col) =
					buffer_convolve_helper(buffer, kernel_inverted, i_buffer_read_index, kernel_size);
		}
		for (; i_row_to_write < row_count; i_row_to_write++, i_buffer_write_index = i_buffer_read_index) {
			buffer[i_buffer_write_index] = math::Vector2f(0.0f);
			i_buffer_read_index = (i_buffer_write_index + 1) % kernel_size;
			y_convolved(i_row_to_write, i_col) =
					buffer_convolve_helper(buffer, kernel_inverted, i_buffer_read_index, kernel_size);
		}
	}
	field = y_convolved;
}

void convolve_with_kernel_x(MatrixXv2f& field, const eig::VectorXf& kernel_1d) {
	eig::Index row_count = field.rows();
	eig::Index column_count = field.cols();

	eig::VectorXf kernel_inverted(kernel_1d.size());

	//flip kernel, see def of discrete convolution on Wikipedia
	for (eig::Index i_kernel_element = 0; i_kernel_element < kernel_1d.size(); i_kernel_element++) {
		kernel_inverted(kernel_1d.size() - i_kernel_element - 1) = kernel_1d(i_kernel_element);
	}

	int kernel_size = static_cast<int>(kernel_inverted.size());
	int kernel_half_size = kernel_size / 2;

	math::Vector2f buffer[kernel_size];
	MatrixXv2f x_convolved = MatrixXv2f::Zero(field.rows(), field.cols());

#pragma omp parallel for private(buffer)
	for (eig::Index i_row = 0; i_row < row_count; i_row++) {
		int i_buffer_write_index = 0;
		for (; i_buffer_write_index < kernel_half_size; i_buffer_write_index++) {
			buffer[i_buffer_write_index] = math::Vector2f(0.0f); // fill buffer with empty value
		}
		eig::Index i_col_to_sample = 0;
		//fill buffer up to the last value
		for (; i_col_to_sample < kernel_half_size; i_col_to_sample++, i_buffer_write_index++) {
			buffer[i_buffer_write_index] = field(i_row, i_col_to_sample);
		}
		int i_buffer_read_index = 0;
		eig::Index i_col_to_write = 0;
		for (; i_col_to_sample < column_count; i_col_to_write++, i_col_to_sample++,
				i_buffer_write_index = i_buffer_read_index) {
			buffer[i_buffer_write_index] = field(i_row, i_col_to_sample); // fill buffer with next value
			i_buffer_read_index = (i_buffer_write_index + 1) % kernel_size;
			x_convolved(i_row, i_col_to_write) =
					buffer_convolve_helper(buffer, kernel_inverted, i_buffer_read_index, kernel_size);
		}
		for (; i_col_to_write < column_count; i_col_to_write++, i_buffer_write_index = i_buffer_read_index) {
			buffer[i_buffer_write_index] = math::Vector2f(0.0f);
			i_buffer_read_index = (i_buffer_write_index + 1) % kernel_size;
			x_convolved(i_row, i_col_to_write) =
					buffer_convolve_helper(buffer, kernel_inverted, i_buffer_read_index, kernel_size);
		}
	}
	field = x_convolved;
}

} //namespace math
