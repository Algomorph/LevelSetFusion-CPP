/*
 * optimizer2d_log.hpp
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


#include "../../telemetry/convergence_report.hpp"

#pragma once
namespace nonrigid_optimization {
namespace hierarchical{

template<bool TOptimized>
struct Telemetry{};

template<>
class Telemetry<true>{

};

template<>
struct Telemetry<false>{



	std::vector<telementry::ConvergenceReport> per_level_convergence_reports;


	void add_level_iteration_result(int i_level, Eigen::MatrixXf live_field,
			math::MatrixXv2f warp_field = math::MatrixXv2f(),
			math::MatrixXv2f data_term_gradients = math::MatrixXv2f(),
			math::MatrixXv2f tikhonov_term_gradients = math::MatrixXv2f(),
			math::MatrixXv2f final_warp_updates = math::MatrixXv2f());

};

} /* namespace hierarchical */
} /* namespace nonrigid_optimization */

