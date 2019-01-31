/*
 * conics.cpp
 *
 *  Created on: Jan 31, 2019
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


//standard library
#include <cmath>
#include <cfloat>

//local
#include "conics.hpp"

namespace math {

/**
 * Compute the minimum and maximum of the tight axis-aligned bounding box of the given centered ellipse.
 * For a centered eclipse with equation,
 * Ax^2 + Bxy + y^2 = F
 * Conic matrix of the ellipse is typically denoted as:
 * Q =
 * ⎡  A  B/2⎤
 * ⎣B/2  C ⎦
 * Equation becomes:
 * p'Qp = F
 *
 * @param ellipse_matrix - conic matrix of the ellipse
 * @param ellipse_scale - scale of the ellipse / altitude of cross section on virtual cone, i.e. the constant F
 * @return axis-aligned bounding box maximum in x and y dimensions (in that order)
 */
eig::Vector2f compute_centered_ellipse_bound_points(const eig::Matrix2f& ellipse_matrix, float ellipse_scale) {
	float A = ellipse_matrix(0, 0);
	float B = ellipse_matrix(0, 1) * 2;
	float C = ellipse_matrix(1, 1);
	float F = ellipse_scale;
	eig::Vector2f bounds_max;
	if (std::abs(B) < FLT_EPSILON) {
		bounds_max << std::sqrt(F / A), std::sqrt(F / C);
	} else {
		float B_squared = B * B;
		float C_squared = C * C;
		bounds_max <<
				std::sqrt(F / ((4 * A * C_squared) / B_squared - C)),
				std::sqrt(F / (A - B_squared / (4 * C)));
	}
	return bounds_max;
}

} //namespace math

