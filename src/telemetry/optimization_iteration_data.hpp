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
#include "../math/tensors.hpp"

#pragma once
namespace telemetry {

//TODO: make templated on type of fields, i.e. 2D (MatrixXf and MatrixXv2f) vs 3D (Tensor3f and Tensor3v3f)

class OptimizationIterationData {
public:
	OptimizationIterationData() = default;

	void add_iteration_result(Eigen::MatrixXf live_field = Eigen::MatrixXf(),
			math::MatrixXv2f warp_field = math::MatrixXv2f(),
			math::MatrixXv2f data_term_gradients = math::MatrixXv2f(),
			math::MatrixXv2f tikhonov_term_gradients = math::MatrixXv2f());

	const std::vector<Eigen::MatrixXf>  get_live_fields() const;
	const std::vector<math::MatrixXv2f> get_warp_fields() const;
	const std::vector<math::MatrixXv2f> get_data_term_gradients() const;
	const std::vector<math::MatrixXv2f> get_tikhonov_term_gradients() const;

	int get_frame_count() const;

	bool operator==(const OptimizationIterationData& rhs){
		return this->live_fields == rhs.live_fields &&
				this->warp_fields == rhs.warp_fields &&
				this->data_term_gradients == rhs.data_term_gradients &&
				this->tikhonov_term_gradients == rhs.tikhonov_term_gradients
				;
	}

private:
	std::vector<Eigen::MatrixXf> live_fields;
	std::vector<math::MatrixXv2f> warp_fields;
	std::vector<math::MatrixXv2f> data_term_gradients;
	std::vector<math::MatrixXv2f> tikhonov_term_gradients;
};

} //namespace telemetry
