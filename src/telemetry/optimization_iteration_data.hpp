/*
 * optimization_iteration_data.hpp
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

//libraries
#include <Eigen/Dense>

//local
#include "../math/typedefs.hpp"
#include "../error_handling/throw_assert.hpp"

#pragma once
namespace telemetry {

template<typename ScalarContainer, typename VectorContainer>
class OptimizationIterationData {
public:
	OptimizationIterationData() = default;

	void add_iteration_result(ScalarContainer live_field = ScalarContainer(),
			VectorContainer warp_field = VectorContainer(),
			VectorContainer data_term_gradients = VectorContainer(),
			VectorContainer tikhonov_term_gradients = VectorContainer());

	const std::vector<ScalarContainer>  get_live_fields() const;
	const std::vector<VectorContainer> get_warp_fields() const;
	const std::vector<VectorContainer> get_data_term_gradients() const;
	const std::vector<VectorContainer> get_tikhonov_term_gradients() const;

	int get_frame_count() const;

	bool operator==(const OptimizationIterationData& rhs){
		throw_assert(false, "not implemented (properly, because of bug in Eigen Tensor module)");
		//TODO: remote the "&" and somehow fix the bug with the Tensor comparison operator
		return &this->live_fields == &rhs.live_fields &&
			   &this->warp_fields == &rhs.warp_fields &&
			   &this->data_term_gradients == &rhs.data_term_gradients &&
			   &this->tikhonov_term_gradients == &rhs.tikhonov_term_gradients
				;
	}

private:
	std::vector<ScalarContainer> live_fields;
	std::vector<VectorContainer> warp_fields;
	std::vector<VectorContainer> data_term_gradients;
	std::vector<VectorContainer> tikhonov_term_gradients;
};

typedef OptimizationIterationData<eig::MatrixXf, math::MatrixXv2f> OptimizationIterationData2d;
typedef OptimizationIterationData<math::Tensor3f, math::Tensor3v3f> OptimizationIterationData3d;

} //namespace telemetry
