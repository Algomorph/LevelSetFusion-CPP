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

void draw_ellipse(eig::MatrixXuc& image, eig::Vector2f ellipse_center,
		const eig::Matrix2f& ellipse_matrix, float ellipse_scale){
	eig::Vector2f bounds_max = compute_centered_ellipse_bound_points(ellipse_matrix, ellipse_scale);
	// compute bounds
	int x_start = static_cast<int>(ellipse_center(0) - bounds_max(0) + 0.5);
	int x_end = static_cast<int>(ellipse_center(0) + bounds_max(0) + 1.5);
	int y_start = static_cast<int>(ellipse_center(1) - bounds_max(1) + 0.5);
	int y_end = static_cast<int>(ellipse_center(1) + bounds_max(1) + 1.5);

	// check that at least some samples within sampling range fall within the depth image
	if (x_start > image.cols() || x_end <= 0
			|| y_start > image.rows() || y_end <= 0) {
		return;
	}

	// limit sampling bounds to image bounds
	x_start = std::max(0, x_start);
	x_end = std::min(static_cast<int>(image.cols()), x_end);
	y_start = std::max(0, y_start);
	y_end = std::min(static_cast<int>(image.rows()), y_end);

	float A = ellipse_matrix(0, 0);
	float B = ellipse_matrix(0, 1) * 2;
	float C = ellipse_matrix(1, 1);
	float F = ellipse_scale;
	for (int x = x_start; x< x_end; x++){
		float x_ellipse_space = static_cast<float>(x) - ellipse_center[0];
		float a = C;
		float b = B * x_ellipse_space;
		float c = A * x_ellipse_space*x_ellipse_space - F;
		float under_root = b*b - 4*a*c;
		if(under_root < 0.0f) continue;
		float addand = std::sqrt(under_root);
		float denominator = -b - addand;
		if(std::abs(denominator) > 10e-6){
			float y_0_ellipse_space =  denominator / (2 * a);
			int y_0 = int(y_0_ellipse_space + ellipse_center[1] + 0.5);
			image(y_0,x) = 0;
		}
		denominator = -b + addand;
		if(std::abs(denominator) > 10e-6){
			float y_1_ellipse_space =  denominator / (2 * a);
			int y_1 = int(y_1_ellipse_space + ellipse_center[1] + 0.5);
			image(y_1,x) = 0;
		}
	}
	for (int y = y_start; y< y_end; y++){
		float y_ellipse_space = static_cast<float>(y) - ellipse_center[1];
		float a = A;
		float b = B * y_ellipse_space;
		float c = C * y_ellipse_space*y_ellipse_space - F;
		float under_root = b*b - 4*a*c;
		if(under_root < 0.0f) continue;
		float addand = std::sqrt(under_root);
		float denominator = -b - addand;
		if(std::abs(denominator) > 10e-6){
			float x_0_ellipse_space =  denominator / (2 * a);
			int x_0 = int(x_0_ellipse_space + ellipse_center[0] + 0.5);
			image(y,x_0) = 0;
		}
		denominator = -b + addand;
		if(std::abs(denominator) > 10e-6){
			float x_1_ellipse_space =  denominator / (2 * a);
			int x_1 = int(x_1_ellipse_space + ellipse_center[0] + 0.5);
			image(y,x_1) = 0;
		}
	}
}

} //namespace math

