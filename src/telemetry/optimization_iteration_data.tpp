/*
 * optimization_iteration_data.cpp
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

#include "optimization_iteration_data.hpp"

namespace telemetry{

template<typename ScalarContainer, typename VectorContainer>
void OptimizationIterationData<ScalarContainer, VectorContainer>::add_iteration_result(
		ScalarContainer live_field,
		VectorContainer warp_field,
		VectorContainer data_term_gradients,
		VectorContainer tikhonov_term_gradients)
	{
	this->live_fields.push_back(live_field);
	this->warp_fields.push_back(warp_field);
	this->data_term_gradients.push_back(data_term_gradients);
	this->tikhonov_term_gradients.push_back(tikhonov_term_gradients);
}

template<typename ScalarContainer, typename VectorContainer>
const std::vector<ScalarContainer> OptimizationIterationData<ScalarContainer, VectorContainer>::get_live_fields() const {
	return this->live_fields;
};
template<typename ScalarContainer, typename VectorContainer>
const std::vector<VectorContainer> OptimizationIterationData<ScalarContainer, VectorContainer>::get_warp_fields() const {
	return this->warp_fields;
};
template<typename ScalarContainer, typename VectorContainer>
const std::vector<VectorContainer> OptimizationIterationData<ScalarContainer, VectorContainer>::get_data_term_gradients() const {
	return this->data_term_gradients;
};
template<typename ScalarContainer, typename VectorContainer>
const std::vector<VectorContainer> OptimizationIterationData<ScalarContainer, VectorContainer>::get_tikhonov_term_gradients() const {
	return this->tikhonov_term_gradients;
};

template<typename ScalarContainer, typename VectorContainer>
int OptimizationIterationData<ScalarContainer, VectorContainer>::get_frame_count() const{
	return this->live_fields.size();
}

} //namespace telemetry


