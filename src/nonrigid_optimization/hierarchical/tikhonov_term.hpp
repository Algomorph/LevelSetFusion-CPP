/*
 * tikhonov_term.hpp
 *
 *  Created on: Mar 18, 2019
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

#include "../../math/typedefs.hpp"
#include "../../math/gradients.hpp"

namespace nonrigid_optimization {
namespace hierarchical {

inline math::MatrixXv2f& compute_tikhonov_gradient(const math::MatrixXv2f& gradient) {
	math::MatrixXv2f tikhonov_gradient;
	math::vector_field_laplacian(gradient, tikhonov_gradient);
	return tikhonov_gradient;
}

inline math::MatrixXv2f& compute_tikhonov_gradient_and_energy(float& tikhonov_energy,
		const math::MatrixXv2f& gradient, float energy_factor = 1000000.0f) {
	math::MatrixXv2f tikhonov_gradient = compute_tikhonov_gradient(gradient);

	eig::MatrixXf gradient_u_component, gradient_v_component;
	math::unstack_xv2f(gradient_u_component, gradient_v_component, gradient);
	eig::MatrixXf u_x, u_y, v_x, v_y;
	float gradient_aggregate_mean;
	math::scalar_field_gradient(gradient_u_component, u_x, u_y);
	math::scalar_field_gradient(gradient_v_component, v_x, v_y);
	gradient_aggregate_mean = (u_x.array().square() + u_y.array().square()
			+ v_x.array().square() + v_y.array().square()).mean();
	tikhonov_energy = energy_factor * 0.5 * gradient_aggregate_mean;

	return tikhonov_gradient;
}

} //namespace hierarchical
} //namespace nonrigid_optimization

