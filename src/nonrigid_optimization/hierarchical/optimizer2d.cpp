/*
 * hierarchical_optimizer2d.cpp
 *
 *  Created on: Dec 18, 2018
 *      Author: Gregory Kramida
 *   Copyright: 2018 Gregory Kramida
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
//stdlib
#include <limits>
#include <cmath>
#include <memory>

//_DEBUG
#include <iostream>

//libraries
#include <unsupported/Eigen/MatrixFunctions>

//local
#include "pyramid2d.hpp"
#include "../field_warping.hpp"
#include "../../math/convolution.hpp"
#include "../../math/gradients.hpp"
#include "../../math/typedefs.hpp"
#include "../../math/statistics.hpp"
#include "../../math/resampling.hpp"
#include "optimizer2d.hpp"
#include "optimizer2d_common.hpp"

namespace nonrigid_optimization {
namespace hierarchical {

Optimizer2d::Optimizer2d(
		bool tikhonov_term_enabled,
		bool gradient_kernel_enabled,

		int maximum_chunk_size,
		float rate,
		int maximum_iteration_count,
		float maximum_warp_update_threshold,

		float data_term_amplifier,
		float tikhonov_strength,
		eig::VectorXf kernel,

		Optimizer2d::ResamplingStrategy resampling_strategy
		) :
		tikhonov_term_enabled(tikhonov_term_enabled && tikhonov_strength > 0.0f),
				gradient_kernel_enabled(gradient_kernel_enabled && kernel.size() > 0),

				maximum_chunk_size(maximum_chunk_size),
				rate(rate),
				maximum_iteration_count(maximum_iteration_count),
				maximum_warp_update_threshold(maximum_warp_update_threshold),

				data_term_amplifier(data_term_amplifier),
				tikhonov_strength(tikhonov_strength),
				kernel_1d(kernel),

				resampling_strategy(resampling_strategy)
{

}

Optimizer2d::~Optimizer2d()
{
	// TODO Auto-generated destructor stub
}

math::MatrixXv2f Optimizer2d::optimize(eig::MatrixXf canonical_field, eig::MatrixXf live_field) {

	eig::MatrixXf live_gradient_x, live_gradient_y;
	math::scalar_field_gradient(live_field, live_gradient_x, live_gradient_y);
	math::DownsamplingStrategy hierarchy_downsampling_strategy;
	math::UpsamplingStrategy warp_upsampling_strategy;
	switch(this->resampling_strategy){
	case Optimizer2d::ResamplingStrategy::LINEAR:
		warp_upsampling_strategy = math::UpsamplingStrategy::LINEAR;
		hierarchy_downsampling_strategy = math::DownsamplingStrategy::LINEAR;
		break;
	case Optimizer2d::ResamplingStrategy::NEAREST_AND_AVERAGE:
	default:
		warp_upsampling_strategy = math::UpsamplingStrategy::NEAREST;
		hierarchy_downsampling_strategy = math::DownsamplingStrategy::AVERAGE;
	}
	Pyramid2d canonical_pyramid(canonical_field, this->maximum_chunk_size, hierarchy_downsampling_strategy);
	Pyramid2d live_pyramid(live_field, this->maximum_chunk_size, hierarchy_downsampling_strategy);
	Pyramid2d live_gradient_x_pyramid(live_gradient_x, this->maximum_chunk_size, hierarchy_downsampling_strategy);
	Pyramid2d live_gradient_y_pyramid(live_gradient_y, this->maximum_chunk_size, hierarchy_downsampling_strategy);

	this->current_hierarchy_level = 0;

	int level_count = static_cast<int>(canonical_pyramid.get_level_count());

	math::MatrixXv2f warp_field(0, 0);

	for (current_hierarchy_level = 0; current_hierarchy_level < level_count; current_hierarchy_level++) {
		const eig::MatrixXf& canonical_pyramid_level = canonical_pyramid.get_level(current_hierarchy_level);
		const eig::MatrixXf& live_pyramid_level = live_pyramid.get_level(current_hierarchy_level);
		const eig::MatrixXf& live_gradient_x_level = live_gradient_x_pyramid.get_level(current_hierarchy_level);
		const eig::MatrixXf& live_gradient_y_level = live_gradient_y_pyramid.get_level(current_hierarchy_level);

		if (current_hierarchy_level == 0) {
			warp_field = math::MatrixXv2f::Zero(canonical_pyramid_level.rows(), canonical_pyramid_level.cols());
		}

		this->optimize_level(warp_field, canonical_pyramid_level, live_pyramid_level,
				live_gradient_x_level, live_gradient_y_level);

		if (current_hierarchy_level != level_count - 1) {
			warp_field = math::upsampleX2(warp_field, warp_upsampling_strategy);
		}

	}

	return warp_field;
}

void Optimizer2d::optimize_level(
		math::MatrixXv2f& warp_field,
		const eig::MatrixXf& canonical_pyramid_level,
		const eig::MatrixXf& live_pyramid_level,
		const eig::MatrixXf& live_gradient_x_level,
		const eig::MatrixXf& live_gradient_y_level
		) {
	float maximum_warp_update_length = std::numeric_limits<float>::max();

	math::MatrixXv2f gradient = math::MatrixXv2f::Zero(warp_field.rows(), warp_field.cols());
	eig::MatrixXf diff;
	math::MatrixXv2f data_gradient;
	math::MatrixXv2f tikhonov_gradient;
	current_iteration = 0;

	while (not this->termination_conditions_reached(maximum_warp_update_length, current_iteration)) {

		this->optimize_iteration(
				gradient, warp_field, diff, data_gradient, tikhonov_gradient,
				maximum_warp_update_length,
				canonical_pyramid_level, live_pyramid_level,
				live_gradient_x_level, live_gradient_y_level);
		current_iteration++;
	}

}

bool Optimizer2d::termination_conditions_reached(float maximum_warp_update_length,
		int completed_iteration_count) {
	return maximum_warp_update_length < this->maximum_warp_update_threshold ||
			completed_iteration_count >= this->maximum_iteration_count;
}

void Optimizer2d::optimize_iteration(
		math::MatrixXv2f& gradient,
		math::MatrixXv2f& warp_field,
		eig::MatrixXf& diff,
		math::MatrixXv2f& data_gradient,
		math::MatrixXv2f& tikhonov_gradient,
		float& maximum_warp_update_length,
		const eig::MatrixXf& canonical_pyramid_level,
		const eig::MatrixXf& live_pyramid_level,
		const eig::MatrixXf& live_gradient_x_level,
		const eig::MatrixXf& live_gradient_y_level) {

	// resample the live field & its gradients using current warps
	eig::MatrixXf resampled_live = warp_2d(live_pyramid_level, warp_field);
	eig::MatrixXf resampled_live_gradient_x = warp_2d_replacement(live_gradient_x_level, warp_field, 0.0f);
	eig::MatrixXf resampled_live_gradient_y = warp_2d_replacement(live_gradient_y_level, warp_field, 0.0f);

	// see how badly our sampled values correspond to the canonical values at the same locations
	// data_gradient = (warped_live - canonical) * warped_gradient(live)
	diff = resampled_live - canonical_pyramid_level;
	eig::MatrixXf data_gradient_x = diff.cwiseProduct(resampled_live_gradient_x);
	eig::MatrixXf data_gradient_y = diff.cwiseProduct(resampled_live_gradient_y);

	// this results in the data term gradient
	data_gradient = math::stack_as_xv2f(data_gradient_x, data_gradient_y);

	if (this->tikhonov_term_enabled) {
		math::vector_field_laplacian_2d(gradient, tikhonov_gradient);
		gradient = this->data_term_amplifier * data_gradient - this->tikhonov_strength * tikhonov_gradient;
	} else {
		gradient = this->data_term_amplifier * data_gradient;
	}

	if (this->gradient_kernel_enabled) {
		math::convolve_with_kernel(gradient, this->kernel_1d);
	}

	// apply gradient-based update to existing warps (we use negative gradient to move in direction of objective
	// function's minimum)
	warp_field -= this->rate * gradient;

	// perform termination condition updates
	math::Vector2i longest_vector_location;
	math::locate_max_norm2(maximum_warp_update_length, longest_vector_location, gradient);
}

} /* namespace hierarchical */
} /* namespace nonrigid_optimization */
