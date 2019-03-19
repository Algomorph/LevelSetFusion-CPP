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

//stdlib
#include <vector>

//libraries
#include <Eigen/Eigen>

//local
#include "../../math/tensors.hpp"
#include "../../telemetry/convergence_report.hpp"
#include "telemetry.hpp"

namespace eig = Eigen;

namespace nonrigid_optimization {
namespace hierarchical{

//not thread-safe

class Optimizer2d {

public:
	Optimizer2d(
			bool tikhonov_term_enabled = true,
			bool gradient_kernel_enabled = true,

			int maximum_chunk_size = 8,
			float rate = 0.1f,
			int maximum_iteration_count = 100,
			float maximum_warp_update_threshold = 0.001f,

			float data_term_amplifier = 1.0f,
			float tikhonov_strength = 0.2f,
			eig::VectorXf kernel = eig::VectorXf(0));
	virtual ~Optimizer2d();

	virtual math::MatrixXv2f optimize(eig::MatrixXf canonical_field, eig::MatrixXf live_field);

protected:
	//parameters
	const bool tikhonov_term_enabled = true;
	const bool gradient_kernel_enabled = true;
	const int maximum_chunk_size = 8;
	const float rate = 0.1f;
	const int maximum_iteration_count = 100;
	const float maximum_warp_update_threshold = 0.001f;
	const float data_term_amplifier = 1.0f;
	const float tikhonov_strength = 0.2f;
	const eig::VectorXf kernel_1d = eig::VectorXf(0);

	virtual void optimize_level(
			math::MatrixXv2f& warp_field,
			const eig::MatrixXf& canonical_pyramid_level,
			const eig::MatrixXf& live_pyramid_level,
			const eig::MatrixXf& live_gradient_x_level,
			const eig::MatrixXf& live_gradient_y_level
			);

	virtual void optimize_iteration(
			math::MatrixXv2f& gradient,
			math::MatrixXv2f& warp_field,
			eig::MatrixXf& diff,
			float& maximum_warp_update_length,
			const eig::MatrixXf& canonical_pyramid_level,
			const eig::MatrixXf& live_pyramid_level,
			const eig::MatrixXf& live_gradient_x_level,
			const eig::MatrixXf& live_gradient_y_level
			);
	inline int get_current_hierarchy_level() { return this->current_hierarchy_level; }
	inline int get_current_iteration() { return this->current_iteration; }
private:
	//optimization state variables
	int current_hierarchy_level = 0;
	int current_iteration = 0;
	bool termination_conditions_reached(float maximum_warp_update_length, int completed_iteration_count);

};
} /* namespace hierarchical */
} /* namespace nonrigid_optimization */
