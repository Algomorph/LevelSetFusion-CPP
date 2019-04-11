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
#include "pyramid.hpp"
#include "../field_warping.hpp"
#include "../../math/convolution.hpp"
#include "../../math/gradients.hpp"
#include "../../math/typedefs.hpp"
#include "../../math/statistics.hpp"
#include "../../math/resampling.hpp"
#include "../../math/field_like.hpp"
#include "../../math/container_wrapper.hpp"
#include "optimizer.hpp"

namespace nonrigid_optimization {
namespace hierarchical {



template<typename ScalarContainer, typename VectorContainer>
Optimizer<ScalarContainer, VectorContainer>::Optimizer(
		bool tikhonov_term_enabled,
		bool gradient_kernel_enabled,

		int maximum_chunk_size,
		float rate,
		int maximum_iteration_count,
		float maximum_warp_update_threshold,

		float data_term_amplifier,
		float tikhonov_strength,
		eig::VectorXf kernel,

		Optimizer::ResamplingStrategy resampling_strategy
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

template<typename ScalarContainer, typename VectorContainer>
VectorContainer Optimizer<ScalarContainer, VectorContainer>::optimize(const ScalarContainer& canonical_field,
		const ScalarContainer& live_field) {

	VectorContainer live_gradient;
	math::gradient(live_gradient, live_field);
	math::DownsamplingStrategy hierarchy_downsampling_strategy;
	math::UpsamplingStrategy warp_upsampling_strategy;
	switch (this->resampling_strategy) {
	case Optimizer::ResamplingStrategy::LINEAR:
		warp_upsampling_strategy = math::UpsamplingStrategy::LINEAR;
		hierarchy_downsampling_strategy = math::DownsamplingStrategy::LINEAR;
		break;
	case Optimizer::ResamplingStrategy::NEAREST_AND_AVERAGE:
		default:
		warp_upsampling_strategy = math::UpsamplingStrategy::NEAREST;
		hierarchy_downsampling_strategy = math::DownsamplingStrategy::AVERAGE;
	}
	Pyramid<ScalarContainer> canonical_pyramid(canonical_field, this->maximum_chunk_size,
			hierarchy_downsampling_strategy);
	Pyramid<ScalarContainer> live_pyramid(live_field, this->maximum_chunk_size, hierarchy_downsampling_strategy);
	Pyramid<VectorContainer> live_gradient_pyramid(live_gradient, this->maximum_chunk_size,
			hierarchy_downsampling_strategy);

	this->current_hierarchy_level = 0;

	int level_count = static_cast<int>(canonical_pyramid.get_level_count());

	VectorContainer warp_field;

	for (current_hierarchy_level = 0; current_hierarchy_level < level_count; current_hierarchy_level++) {
		const ScalarContainer& canonical_pyramid_level = canonical_pyramid.get_level(current_hierarchy_level);
		const ScalarContainer& live_pyramid_level = live_pyramid.get_level(current_hierarchy_level);
		const VectorContainer& live_gradient_level = live_gradient_pyramid.get_level(current_hierarchy_level);

		if (current_hierarchy_level == 0) {
			warp_field = math::vector_field_like(canonical_pyramid_level);
			warp_field.setZero();
		}

		this->optimize_level(warp_field, canonical_pyramid_level, live_pyramid_level, live_gradient_level);

		if (current_hierarchy_level != level_count - 1) {
			warp_field = math::upsampleX2(warp_field, warp_upsampling_strategy);
		}

	}

	return warp_field;
}

template<typename ScalarContainer, typename VectorContainer>
void Optimizer<ScalarContainer, VectorContainer>::optimize_level(
		VectorContainer& warp_field,
		const ScalarContainer& canonical_pyramid_level,
		const ScalarContainer& live_pyramid_level,
		const VectorContainer& live_gradient_level
		) {
	float maximum_warp_update_length = std::numeric_limits<float>::max();

	VectorContainer gradient = math::vector_field_like(warp_field);
	gradient.setZero();
	eig::MatrixXf diff;
	VectorContainer data_gradient;
	VectorContainer tikhonov_gradient;
	current_iteration = 0;

	while (not this->termination_conditions_reached(maximum_warp_update_length, current_iteration)) {

		this->optimize_iteration(
				gradient, 
				warp_field, 
				diff, 
				data_gradient, 
				tikhonov_gradient,
				maximum_warp_update_length,
				canonical_pyramid_level,
				live_pyramid_level,
				live_gradient_level);
		current_iteration++;
	}

}

template<typename ScalarContainer, typename VectorContainer>
bool Optimizer<ScalarContainer, VectorContainer>::termination_conditions_reached(float maximum_warp_update_length,
		int completed_iteration_count) {
	return maximum_warp_update_length < this->maximum_warp_update_threshold ||
			completed_iteration_count >= this->maximum_iteration_count;
}

template<typename ScalarContainer, typename VectorContainer>
void Optimizer<ScalarContainer, VectorContainer>::optimize_iteration(
		VectorContainer& gradient,
		VectorContainer& warp_field,
		ScalarContainer& diff,
		VectorContainer& data_gradient,
		VectorContainer& tikhonov_gradient,
		float& maximum_warp_update_length,
		const ScalarContainer& canonical_pyramid_level,
		const ScalarContainer& live_pyramid_level,
		const VectorContainer& live_gradient_level) {

	// resample the live field & its gradients using current warps
	ScalarContainer resampled_live = warp(live_pyramid_level, warp_field);
	VectorContainer resampled_live_gradient = warp_with_replacement(live_gradient_level, warp_field, VectorType(0.0f));

	// see how badly our sampled values correspond to the canonical values at the same locations
	// data_gradient = (warped_live - canonical) * warped_gradient(live)
	diff = resampled_live - canonical_pyramid_level;
	data_gradient = resampled_live_gradient.cwiseProduct(diff);

	if (this->tikhonov_term_enabled) {
		math::laplacian(tikhonov_gradient, gradient);
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
	Coordinates longest_vector_location;
	math::locate_max_norm(maximum_warp_update_length, longest_vector_location, gradient);
}

} /* namespace hierarchical */
} /* namespace nonrigid_optimization */
