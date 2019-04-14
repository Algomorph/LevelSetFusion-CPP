/*
 * hierarchical_optimizer2d.h
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
#pragma once

// stdlib
#include <vector>

// libraries
#include <Eigen/Eigen>

// local
#include "../../math/typedefs.hpp"
#include "../../math/container_traits.hpp"

namespace eig = Eigen;

namespace nonrigid_optimization {
namespace hierarchical {

// not thread-safe
/**
 * Hierarchical non-rigid optimizer that constructs pyramids of the initial live (source) TSDF and canonical (target)
 * TSDF fields, and then constructs composite warp vectors level-by-level, upsampling the warp field at each new level.
 */
template<typename ScalarContainer, typename VectorContainer>
class Optimizer {
public:
	typedef typename VectorContainer::Scalar VectorType;
	typedef typename math::ContainerWrapper<ScalarContainer>::Coordinates Coordinates;
	enum ResamplingStrategy {
		NEAREST_AND_AVERAGE = 0,
		LINEAR = 1
	};
	Optimizer(
			bool tikhonov_term_enabled = true,
			bool gradient_kernel_enabled = true,

			int maximum_chunk_size = 8,
			float rate = 0.1f,
			int maximum_iteration_count = 100,
			float maximum_warp_update_threshold = 0.001f,

			float data_term_amplifier = 1.0f,
			float tikhonov_strength = 0.2f,
			eig::VectorXf kernel = eig::VectorXf(0),

			ResamplingStrategy resampling_strategy = ResamplingStrategy::NEAREST_AND_AVERAGE
	);

	virtual ~Optimizer() = default;

	virtual VectorContainer optimize(const ScalarContainer& canonical_field, const ScalarContainer& live_field);

protected:
	// parameters
	const bool tikhonov_term_enabled = true;
	const bool gradient_kernel_enabled = true;
	const int maximum_chunk_size = 8;
	const float rate = 0.1f;
	const int maximum_iteration_count = 100;
	const float maximum_warp_update_threshold = 0.001f;
	const float data_term_amplifier = 1.0f;
	const float tikhonov_strength = 0.2f;
	const eig::VectorXf kernel_1d = eig::VectorXf(0);
	const ResamplingStrategy resampling_strategy;

	virtual void optimize_level(
			VectorContainer& warp_field,
			const ScalarContainer& canonical_pyramid_level,
			const ScalarContainer& live_pyramid_level,
			const VectorContainer& live_gradient_level);

	virtual void optimize_iteration(
			VectorContainer& gradient,
			VectorContainer& warp_field,
			ScalarContainer& diff,
			VectorContainer& data_gradient,
			VectorContainer& tikhonov_gradient,
			float& maximum_warp_update_length,
			const ScalarContainer& canonical_pyramid_level,
			const ScalarContainer& live_pyramid_level,
			const VectorContainer& live_gradient_level);
	inline int get_current_hierarchy_level() { return this->current_hierarchy_level; }
	inline int get_current_iteration() { return this->current_iteration; }

private:
	// optimization state variables
	int current_hierarchy_level = 0;
	int current_iteration = 0;
	bool termination_conditions_reached(float maximum_warp_update_length, int completed_iteration_count);
};

typedef Optimizer<Eigen::MatrixXf, math::MatrixXv2f> Optimizer2d;

} /* namespace hierarchical */
} /* namespace nonrigid_optimization */
