/*
 * hierarchical_optimizer3d.cpp
 *
 *  Created on: Mar 1, 2019
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

//stdlib
#include <limits>
#include <algorithm>

//local
#include "../math/convolution.hpp"
#include "../math/gradients.hpp"
#include "../math/typedefs.hpp"
#include "../math/statistics.hpp"
#include "hierarchical_optimizer3d.hpp"
#include "pyramid3d.hpp"
#include "field_resampling.hpp"

namespace nonrigid_optimization {

HierarchicalOptimizer3d::HierarchicalOptimizer3d(
		bool tikhonov_term_enabled,
		bool gradient_kernel_enabled,
		int maximum_chunk_size,
		float rate,
		int maximum_iteration_count,
		float maximum_warp_update_threshold,
		float data_term_amplifier,
		float tikhonov_strength,
		eig::VectorXf kernel,
		VerbosityParameters verbosity_parameters
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
				verbosity_parameters(verbosity_parameters) {
}

HierarchicalOptimizer3d::VerbosityParameters::VerbosityParameters(
		bool print_iteration_max_warp_update,
		bool print_iteration_data_energy,
		bool print_iteration_tikhonov_energy) :
		print_iteration_max_warp_update(print_iteration_max_warp_update),
				print_iteration_data_energy(print_iteration_data_energy),
				print_iteration_tikhonov_energy(print_iteration_tikhonov_energy),
				print_per_iteration_info(
						print_iteration_max_warp_update ||
								print_iteration_data_energy ||
								print_iteration_tikhonov_energy
								) {
}
//
//math::Tensor3v3f HierarchicalOptimizer3d::optimize(eig::Tensor3f canonical_field, eig::Tensor3f live_field) {
//
//	math::Tensor3v3f live_gradient;
//	math::field_gradient(live_gradient, live_field);
//	Pyramid3d canonical_pyramid(canonical_field, this->maximum_chunk_size);
//	Pyramid3d live_pyramid(live_field, this->maximum_chunk_size);
//	Pyramid3d live_gradient_pyramid(live_gradient, this->maximum_chunk_size);
//
//	this->current_hierarchy_level = 0;
//
//	int level_count = static_cast<int>(canonical_pyramid.get_level_count());
//
//	math::Tensor3v2f warp_field(0, 0);
//
//	for (current_hierarchy_level = 0; current_hierarchy_level < level_count; current_hierarchy_level++) {
//		const eig::MatrixXf& canonical_pyramid_level = canonical_pyramid.get_level(current_hierarchy_level);
//		const eig::MatrixXf& live_pyramid_level = live_pyramid.get_level(current_hierarchy_level);
//		const eig::MatrixXf& live_gradient_x_pyramid_level = live_gradient_x_pyramid.get_level(current_hierarchy_level);
//		const eig::MatrixXf& live_gradient_y_pyramid_level = live_gradient_y_pyramid.get_level(current_hierarchy_level);
//
//		if (current_hierarchy_level == 0) {
//			warp_field = math::Tensor3v2f::Zero(canonical_pyramid_level.rows(), canonical_pyramid_level.cols());
//		}
//
//		this->optimize_level(warp_field, canonical_pyramid_level, live_pyramid_level,
//				live_gradient_x_pyramid_level, live_gradient_y_pyramid_level);
//
//		if (current_hierarchy_level != level_count - 1) {
//			warp_field = math::upsampleX2(warp_field);
//		}
//
//	}
//
//	return warp_field;
//}
//
void HierarchicalOptimizer3d::optimize_level(
		math::Tensor3v3f& warp_field,
		const eig::Tensor3f& canonical_pyramid_level,
		const eig::Tensor3f& live_pyramid_level,
		const math::Tensor3v3f& live_gradient_level
		) {
	float maximum_warp_update_length = std::numeric_limits<float>::max();
	int iteration_count = 0;

	//TODO:wtf?
	//math::Tensor3v3f gradient(warp_field.dimensions());
	//std::fill_n(gradient.data(), gradient.size(), math::Vector3(1.0f));
	//float normalized_tikhonov_energy = 0;

	while (not this->termination_conditions_reached(maximum_warp_update_length, iteration_count)) {
		// resample the live field & its gradients using current warps
		eig::Tensor3f resampled_live = warp(live_pyramid_level, warp_field);
		//TODO
//		eig::MatrixXf resampled_live_gradient = resample_field_replacement(live_gradient_x_level, warp_field, 0.0f);
//
//
//		// see how badly our sampled values correspond to the canonical values at the same locations
//		// data_gradient = (warped_live - canonical) * warped_gradient(live)
//		eig::MatrixXf diff = resampled_live - canonical_pyramid_level;
//		eig::MatrixXf data_gradient_x = diff.cwiseProduct(resampled_live_gradient_x);
//		eig::MatrixXf data_gradient_y = diff.cwiseProduct(resampled_live_gradient_y);
//
//		// this results in the data term gradient
//		math::Tensor3v2f data_gradient = math::stack_as_xv2f(data_gradient_x, data_gradient_y);
//
//		if (this->tikhonov_term_enabled) {
//			math::Tensor3v2f tikhonov_gradient;
//			math::vector_field_laplacian(gradient, tikhonov_gradient);
//			gradient = this->data_term_amplifier * data_gradient - this->tikhonov_strength * tikhonov_gradient;
//		} else {
//			gradient = this->data_term_amplifier * data_gradient;
//		}
//
//		if (this->gradient_kernel_enabled) {
//			math::convolve_with_kernel(gradient, this->kernel_1d);
//		}
//
//		// apply gradient-based update to existing warps
//		warp_field -= this->rate * gradient;
//
//		// perform termination condition updates
//		math::Vector2i longest_vector_location;
//		math::locate_max_norm2(maximum_warp_update_length, longest_vector_location, gradient);

		iteration_count++;
	}
}

bool HierarchicalOptimizer3d::termination_conditions_reached(float maximum_warp_update_length,
		int completed_iteration_count) {
	return maximum_warp_update_length < this->maximum_warp_update_threshold ||
			completed_iteration_count >= this->maximum_iteration_count;
}

} /* namespace nonrigid_optimization */

