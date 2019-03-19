/*
 * optimizer2d_common.hpp
 *
 *  Created on: Mar 19, 2019
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

//libraries
#include <Eigen/Dense>

//local
#include "../field_warping.hpp"
#include "../../math/convolution.hpp"
#include "../../math/gradients.hpp"
#include "../../math/typedefs.hpp"
#include "../../math/statistics.hpp"

namespace eig = Eigen;

namespace nonrigid_optimization {
namespace hierarchical {


inline void optimize_iteration(
		math::MatrixXv2f& gradient,
		math::MatrixXv2f& warp_field,
		eig::MatrixXf& diff,
		math::MatrixXv2f& tikhonov_gradient,
		math::MatrixXv2f& data_gradient,
		float& maximum_warp_update_length,

		bool tikhonov_term_enabled,
		bool gradient_kernel_enabled,
		float data_term_amplifier,
		float tikhonov_strength,
		float rate,

		const eig::VectorXf& kernel_1d,
		const eig::MatrixXf& canonical_pyramid_level,
		const eig::MatrixXf& live_pyramid_level,
		const eig::MatrixXf& live_gradient_x_level,
		const eig::MatrixXf& live_gradient_y_level){

	// resample the live field & its gradients using current warps
	eig::MatrixXf resampled_live = resample_field(live_pyramid_level, warp_field);
	eig::MatrixXf resampled_live_gradient_x = resample_field_replacement(live_gradient_x_level, warp_field, 0.0f);
	eig::MatrixXf resampled_live_gradient_y = resample_field_replacement(live_gradient_y_level, warp_field, 0.0f);

	// see how badly our sampled values correspond to the canonical values at the same locations
	// data_gradient = (warped_live - canonical) * warped_gradient(live)
	diff = resampled_live - canonical_pyramid_level;
	eig::MatrixXf data_gradient_x = diff.cwiseProduct(resampled_live_gradient_x);
	eig::MatrixXf data_gradient_y = diff.cwiseProduct(resampled_live_gradient_y);

	// this results in the data term gradient
	data_gradient = math::stack_as_xv2f(data_gradient_x, data_gradient_y);

	if (tikhonov_term_enabled) {
		math::vector_field_laplacian(gradient, tikhonov_gradient);
		gradient = data_term_amplifier * data_gradient - tikhonov_strength * tikhonov_gradient;
	} else {
		gradient = data_term_amplifier * data_gradient;
	}

	if (gradient_kernel_enabled) {
		math::convolve_with_kernel(gradient, kernel_1d);
	}

	// apply gradient-based update to existing warps (we use negative gradient to move in direction of objective
	// function's minimum)
	warp_field -= rate * gradient;

	// perform termination condition updates
	math::Vector2i longest_vector_location;
	math::locate_max_norm2(maximum_warp_update_length, longest_vector_location, gradient);
}

} //namespace hierarchical
} //namespace nonrigid_optimization

