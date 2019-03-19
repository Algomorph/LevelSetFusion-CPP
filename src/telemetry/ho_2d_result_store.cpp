/*
 * hierarchical_optimization_2d_logger.cpp
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

//local
#include "../telemetry/ho_2d_result_store.hpp"

namespace logging {

HierarchicalOptimization2dResultStore::HierarchicalOptimization2dResultStore(std::time_t optimization_start_time)
: optimization_start_time(optimization_start_time) {

}

const std::vector<math::MatrixXv2f> HierarchicalOptimization2dResultStore::PerLevelResultStore::get_warp_fields() const {
	return this->warp_fields;
};
const std::vector<math::MatrixXv2f> HierarchicalOptimization2dResultStore::PerLevelResultStore::get_data_term_gradients() const {
	return this->data_term_gradients;
};
const std::vector<math::MatrixXv2f> HierarchicalOptimization2dResultStore::PerLevelResultStore::get_tikhonov_term_gradients() const {
	return this->tikhonov_term_gradients;
};
const std::vector<math::MatrixXv2f> HierarchicalOptimization2dResultStore::PerLevelResultStore::get_final_warp_updates() const {
	return this->final_warp_updates;
};


} //namespace logging

