/*
 * conics.hpp
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

#pragma once

/**
 * Various representations and routines for conic sections
 */

//libraries
#include <Eigen/Eigen>

//local
#include "typedefs.hpp"

namespace eig = Eigen;

namespace math{

eig::Vector2f compute_centered_ellipse_bound_points(const eig::Matrix2f& ellipse_matrix, float ellipse_scale);
void draw_ellipse(eig::MatrixXuc& image, eig::Vector2f center,
		const eig::Matrix2f& ellipse_matrix, float ellipse_scale);
}//namespace math


