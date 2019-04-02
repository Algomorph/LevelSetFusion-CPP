//  ================================================================
//  Created by Gregory Kramida on 10/26/18.
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

#include "gradients.hpp"
#include "typedefs.hpp"

namespace math {

template<typename ElementType>
struct LaplaceOperatorFunctor {
	//same as replicating the prev_row_val to the border and doing (nonborder_value - 2*border_value + border_value)
	inline static ElementType apply_border_operator(ElementType nonborder_value, ElementType border_value) {
		return nonborder_value - border_value;
	}

	inline static ElementType
	apply_row_operator(ElementType next_row_val, ElementType row_val, ElementType prev_row_val) {
		return next_row_val - 2 * row_val + prev_row_val;
	}

	inline static ElementType
	apply_column_operator(ElementType next_col_val, ElementType col_val, ElementType prev_col_val) {
		return next_col_val + prev_col_val;
	}
};

template<typename ElementType>
struct NegativeLaplaceOperatorFunctor {
	//same as replicating the prev_row_val to the border and doing (nonborder_value + 2*border_value - border_value)
	inline static ElementType apply_border_operator(ElementType nonborder_value, ElementType border_value) {
		return -nonborder_value + border_value;
	}

	inline static ElementType
	apply_row_operator(ElementType next_row_val, ElementType row_val, ElementType prev_row_val) {
		return -next_row_val + 2 * row_val - prev_row_val;
	}

	inline static ElementType
	apply_column_operator(ElementType next_col_val, ElementType col_val, ElementType prev_col_val) {
		return -next_col_val - prev_col_val;
	}
};

template<typename LaplacelikeOperatorFunctor>
inline void vector_field_laplacian_aux(const math::MatrixXv2f& field, math::MatrixXv2f& laplacian) {
	eig::Index column_count = field.cols();
	eig::Index row_count = field.rows();
	laplacian = math::MatrixXv2f(row_count, column_count);

#pragma omp parallel for
	for (eig::Index i_col = 0; i_col < column_count; i_col++) {
		math::Vector2f prev_row_val = field(0, i_col);
		math::Vector2f row_val = field(1, i_col);

		laplacian(0, i_col) = LaplacelikeOperatorFunctor::apply_border_operator(row_val, prev_row_val);
		eig::Index i_row;
		for (i_row = 1; i_row < row_count - 1; i_row++) {
			math::Vector2f next_row_val = field(i_row + 1, i_col);
			//previous/next column values will be used later
			laplacian(i_row, i_col) = LaplacelikeOperatorFunctor::apply_row_operator(next_row_val, row_val,
					prev_row_val);
			prev_row_val = row_val;
			row_val = next_row_val;
		}
		laplacian(i_row, i_col) = LaplacelikeOperatorFunctor::apply_border_operator(prev_row_val, row_val);
	}
#pragma omp parallel for
	for (eig::Index i_row = 0; i_row < row_count; i_row++) {
		math::Vector2f prev_col_val = field(i_row, 0);
		math::Vector2f col_val = field(i_row, 1);
		laplacian(i_row, 0) += LaplacelikeOperatorFunctor::apply_border_operator(col_val, prev_col_val);
		eig::Index i_col;
		for (i_col = 1; i_col < column_count - 1; i_col++) {
			math::Vector2f next_col_val = field(i_row, i_col + 1);
			laplacian(i_row, i_col) += LaplacelikeOperatorFunctor::apply_column_operator(next_col_val, col_val,
					prev_col_val);
			prev_col_val = col_val;
			col_val = next_col_val;
		}
		laplacian(i_row, i_col) += LaplacelikeOperatorFunctor::apply_border_operator(prev_col_val, col_val);
	}
}

void vector_field_laplacian_2d(math::MatrixXv2f& laplacian, const math::MatrixXv2f& field) {
	vector_field_laplacian_aux<LaplaceOperatorFunctor<math::Vector2f>>(field, laplacian);
}

void vector_field_negative_laplacian(math::MatrixXv2f& laplacian, const math::MatrixXv2f& field) {
	vector_field_laplacian_aux<NegativeLaplaceOperatorFunctor<math::Vector2f>>(field, laplacian);
}

template<typename LaplacelikeOperatorFunctor>
inline void vector_field_laplacian_3d_aux(math::Tensor3v3f& laplacian, const math::Tensor3v3f& field) {
	int x_size = field.dimension(0);
	int y_size = field.dimension(1);
	int z_size = field.dimension(2);

	laplacian = math::Tensor3v3f(x_size, y_size, z_size);

#pragma omp parallel for
	for (int z = 0; z < z_size; z++) {
		for (int y = 0; y < y_size; y++) {
			math::Vector3f prev_row_val = field(0, y, z);
			math::Vector3f row_val = field(1, y, z);

			laplacian(0, y, z) = LaplacelikeOperatorFunctor::apply_border_operator(row_val, prev_row_val);
			int x;
			for (x = 1; x < x_size - 1; x++) {
				math::Vector3f next_row_val = field(x + 1, y, z);
				//previous/next column values will be used later
				laplacian(x, y, z) = LaplacelikeOperatorFunctor::apply_row_operator(next_row_val, row_val,
						prev_row_val);
				prev_row_val = row_val;
				row_val = next_row_val;
			}
			laplacian(x, y, z) = LaplacelikeOperatorFunctor::apply_border_operator(prev_row_val, row_val);
		}
	}
#pragma omp parallel for
	for (int z = 0; z < z_size; z++) {
		for (int x = 0; x < x_size; x++) {
			math::Vector3f prev_row_val = field(x, 0, z);
			math::Vector3f row_val = field(x, 1, z);

			laplacian(x, 0, z) = LaplacelikeOperatorFunctor::apply_border_operator(row_val, prev_row_val);
			int y;
			for (y = 1; y < y_size - 1; y++) {
				math::Vector3f next_row_val = field(x, y + 1, z);
				//previous/next column values will be used later
				laplacian(x, y, z) = LaplacelikeOperatorFunctor::apply_row_operator(next_row_val, row_val,
						prev_row_val);
				prev_row_val = row_val;
				row_val = next_row_val;
			}
			laplacian(x, y, z) = LaplacelikeOperatorFunctor::apply_border_operator(prev_row_val, row_val);
		}
	}
#pragma omp parallel for
	for (int y = 0; y < y_size; y++) {
		for (int x = 0; x < x_size; x++) {
			math::Vector3f prev_row_val = field(x, y, 0);
			math::Vector3f row_val = field(x, y, 1);

			laplacian(x, y, 0) = LaplacelikeOperatorFunctor::apply_border_operator(row_val, prev_row_val);
			int z;
			for (z = 1; z < z_size - 1; z++) {
				math::Vector3f next_row_val = field(x, y, z + 1);
				//previous/next column values will be used later
				laplacian(x, y, z) = LaplacelikeOperatorFunctor::apply_row_operator(next_row_val, row_val,
						prev_row_val);
				prev_row_val = row_val;
				row_val = next_row_val;
			}
			laplacian(x, y, z) = LaplacelikeOperatorFunctor::apply_border_operator(prev_row_val, row_val);
		}
	}
}

void vector_field_laplacian_3d(math::Tensor3v3f& laplacian, const math::Tensor3v3f& field) {
	vector_field_laplacian_3d_aux<LaplaceOperatorFunctor<math::Vector3f>>(laplacian, field);
}

/**
 * Compute numerical gradient of given matrix. Uses forward and backward differences at the boundary matrix entries,
 * central difference formula for all other entries.
 * Reference: https://en.wikipedia.org/wiki/Numerical_differentiation ;
 * Central difference formula: (f(x + h) - f(x-h))/2h, with h = 1 matrix grid cell
 *
 * @param[in] field scalar field representing the implicit function to differentiate
 * @param[out] live_gradient_x output gradient along the x axis
 * @param[out] live_gradient_y output gradient along the y axis
 */
void scalar_field_gradient(eig::MatrixXf& gradient_x, eig::MatrixXf& gradient_y, const eig::MatrixXf& field) {

	eig::Index column_count = field.cols();
	eig::Index row_count = field.rows();

	gradient_x = eig::MatrixXf(row_count, column_count);
	gradient_y = eig::MatrixXf(row_count, column_count);

#pragma omp parallel for
	for (eig::Index i_col = 0; i_col < column_count; i_col++) {
		float prev_row_val = field(0, i_col);
		float row_val = field(1, i_col);
		gradient_y(0, i_col) = row_val - prev_row_val;
		eig::Index i_row;
		for (i_row = 1; i_row < row_count - 1; i_row++) {
			float next_row_val = field(i_row + 1, i_col);
			gradient_y(i_row, i_col) = 0.5 * (next_row_val - prev_row_val);
			prev_row_val = row_val;
			row_val = next_row_val;
		}
		gradient_y(i_row, i_col) = row_val - prev_row_val;
	}
#pragma omp parallel for
	for (eig::Index i_row = 0; i_row < row_count; i_row++) {
		float prev_col_val = field(i_row, 0);
		float col_val = field(i_row, 1);
		gradient_x(i_row, 0) = col_val - prev_col_val;
		eig::Index i_col;
		for (i_col = 1; i_col < column_count - 1; i_col++) {
			float next_col_val = field(i_row, i_col + 1);
			gradient_x(i_row, i_col) = 0.5 * (next_col_val - prev_col_val);
			prev_col_val = col_val;
			col_val = next_col_val;
		}
		gradient_x(i_row, i_col) = col_val - prev_col_val;
	}
}

/**
 * Compute numerical gradient of given matrix. Uses forward and backward differences at the boundary matrix entries,
 * central difference formula for all other entries.
 * Reference: https://en.wikipedia.org/wiki/Numerical_differentiation ;
 * Central difference formula: (f(x + h) - f(x-h))/2h, with h = 1 matrix grid cell
 *
 * @param[in] field scalar field representing the implicit function to differentiate
 * @param[out] live_gradient_field output gradient field with vectors containing the x and the y gradient for each location
 */
void scalar_field_gradient(math::MatrixXv2f& gradient, const eig::MatrixXf& field) {
	eig::Index column_count = field.cols();
	eig::Index row_count = field.rows();
	gradient = math::MatrixXv2f(row_count, column_count);

#pragma omp parallel for
	for (eig::Index i_col = 0; i_col < column_count; i_col++) {
		float prev_row_val = field(0, i_col);
		float row_val = field(1, i_col);
		gradient(0, i_col).y = row_val - prev_row_val;
		eig::Index i_row;
		for (i_row = 1; i_row < row_count - 1; i_row++) {
			float next_row_val = field(i_row + 1, i_col);
			gradient(i_row, i_col).y = 0.5 * (next_row_val - prev_row_val);
			prev_row_val = row_val;
			row_val = next_row_val;
		}
		gradient(i_row, i_col).y = row_val - prev_row_val;
	}
#pragma omp parallel for
	for (eig::Index i_row = 0; i_row < row_count; i_row++) {
		float prev_col_val = field(i_row, 0);
		float col_val = field(i_row, 1);
		gradient(i_row, 0).x = col_val - prev_col_val;
		eig::Index i_col;
		for (i_col = 1; i_col < column_count - 1; i_col++) {
			float next_col_val = field(i_row, i_col + 1);
			gradient(i_row, i_col).x = 0.5 * (next_col_val - prev_col_val);
			prev_col_val = col_val;
			col_val = next_col_val;
		}
		gradient(i_row, i_col).x = col_val - prev_col_val;
	}
}

/**
 * \brief Computes gradient of the given 2D vector field
 * \details If the input vector field contains vectors of the form [u v]*, the output
 * will contain, for each corresponding location, the Jacobian 2x2 matrix
 *       |u_x u_y|
 *       |v_x v_y|
 * Gradients are computed numerically using central differences, with forward & backward differences applied to the
 * boundary values.
 * \param field a matrix of 2-entry vectors
 * \param gradient numerical gradient of the input matrix
 */
void vector_field_gradient(math::MatrixXm2f& gradient, const math::MatrixXv2f& field) {

	eig::Index row_count = field.rows();
	eig::Index column_count = field.cols();

	gradient = math::MatrixXm2f(row_count, column_count);

#pragma omp parallel for
	for (eig::Index i_col = 0; i_col < column_count; i_col++) {
		math::Vector2f prev_row_vector = field(0, i_col);
		math::Vector2f current_row_vector = field(1, i_col);
		gradient(0, i_col).set_column(1, current_row_vector - prev_row_vector);
		eig::Index i_row;
		//traverse each column in vertical (y) direction
		for (i_row = 1; i_row < row_count - 1; i_row++) {
			math::Vector2f next_row_vector = field(i_row + 1, i_col);
			gradient(i_row, i_col).set_column(1, 0.5 * (next_row_vector - prev_row_vector));
			prev_row_vector = current_row_vector;
			current_row_vector = next_row_vector;
		}
		gradient(i_row, i_col).set_column(1, current_row_vector - prev_row_vector);
	}
#pragma omp parallel for
	for (eig::Index i_row = 0; i_row < row_count; i_row++) {
		math::Vector2f prev_col_vector = field(i_row, 0);
		math::Vector2f current_col_vector = field(i_row, 1);
		gradient(i_row, 0).set_column(0, current_col_vector - prev_col_vector);
		eig::Index i_col;
		for (i_col = 1; i_col < column_count - 1; i_col++) {
			math::Vector2f next_col_vector = field(i_row, i_col + 1);
			gradient(i_row, i_col).set_column(0, 0.5 * (next_col_vector - prev_col_vector));
			prev_col_vector = current_col_vector;
			current_col_vector = next_col_vector;
		}
		gradient(i_row, i_col).set_column(0, current_col_vector - prev_col_vector);
	}
}

//TODO: test which gradient method is faster
void field_gradient2(math::Tensor3v3f& gradient, const math::Tensor3f field) {
	gradient = math::Tensor3v3f(field.dimensions());
	int y_stride = field.dimension(0);
	int z_stride = y_stride * field.dimension(1);

	const int x_size = field.dimension(0);
	const int y_size = field.dimension(1);
	const int z_size = field.dimension(2);

#pragma omp parallel for
	for (eig::Index i_element = 0; i_element < field.size(); i_element++) {
		int z_field = i_element / z_stride;
		int remainder = i_element % z_stride;
		int y_field = remainder / y_stride;
		int x_field = remainder % y_stride;
		//local gradient values
		float x_grad, y_grad, z_grad;

		// use forward/backward finite differences for borders, central differences for everything else
		if (x_field == 0) {
			x_grad = field(x_field + 1, y_field, z_field) - field(x_field, y_field, z_field);
		} else if (x_field == x_size - 1) {
			x_grad = field(x_field, y_field, z_field) - field(x_field - 1, y_field, z_field);
		} else {
			x_grad = 0.5 * field(x_field + 1, y_field, z_field) - field(x_field - 1, y_field, z_field);
		}
		if (y_field == 0) {
			y_grad = field(x_field, y_field + 1, z_field) - field(x_field, y_field, z_field);
		} else if (y_field == y_size - 1) {
			y_grad = field(x_field, y_field, z_field) - field(x_field, y_field - 1, z_field);
		} else {
			y_grad = 0.5 * field(x_field, y_field + 1, z_field) - field(x_field, y_field - 1, z_field);
		}
		if (z_field == 0) {
			z_grad = field(x_field, y_field, z_field + 1) - field(x_field, y_field, z_field);
		} else if (z_field == z_size - 1) {
			z_grad = field(x_field, y_field, z_field) - field(x_field, y_field, z_field - 1);
		} else {
			z_grad = 0.5 * field(x_field, y_field, z_field + 1) - field(x_field, y_field, z_field - 1);
		}

		gradient(i_element) = math::Vector3<float>(x_grad, y_grad, z_grad);
	}
}

void field_gradient(math::Tensor3v3f& gradient, const math::Tensor3f field) {
	gradient = math::Tensor3v3f(field.dimensions());

	const int x_size = field.dimension(0);
	const int y_size = field.dimension(1);
	const int z_size = field.dimension(2);

#pragma omp parallel for
	for (int z = 0; z < z_size; z++) {
		for (int y = 0; y < y_size; y++) {
			float preceding_value = field(0, y, z);
			float current_value = field(1, y, z);
			gradient(0, y, z).u = current_value - preceding_value;
			int x;
			for (x = 1; x < x_size - 1; x++) {
				float next_value = field(x + 1, y, z);
				gradient(x, y, z).u = 0.5 * (next_value - preceding_value);
				preceding_value = current_value;
				current_value = next_value;
			}
			gradient(x, y, z).u = current_value - preceding_value;
		}
	}
#pragma omp parallel for
	for (int z = 0; z < z_size; z++) {
		for (int x = 0; x < x_size; x++) {
			float preceding_value = field(x, 0, z);
			float current_value = field(x, 1, z);
			gradient(x, 0, z).v = current_value - preceding_value;
			int y;
			for (y = 1; y < y_size - 1; y++) {
				float next_value = field(x, y + 1, z);
				gradient(x, y, z).v = 0.5 * (next_value - preceding_value);
				preceding_value = current_value;
				current_value = next_value;
			}
			gradient(x, y, z).v = current_value - preceding_value;
		}
	}
#pragma omp parallel for
	for (int y = 0; y < y_size; y++) {
		for (int x = 0; x < x_size; x++) {
			float preceding_value = field(x, y, 0);
			float current_value = field(x, y, 1);
			gradient(x, y, 0).w = current_value - preceding_value;
			int z;
			for (z = 1; z < z_size - 1; z++) {
				float next_value = field(x, y, z + 1);
				gradient(x, y, z).w = 0.5 * (next_value - preceding_value);
				preceding_value = current_value;
				current_value = next_value;
			}
			gradient(x, y, z).w = current_value - preceding_value;
		}
	}
}

} // namespace math
