/*
 * hierarchical_optimization_2d_logger.hpp
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

//stdlib
#include <vector>
#include <ctime>

//libraries
#include <Eigen/Dense>

//local
#include "../math/tensors.hpp"

/**
 * Class responsible for storing up results for later visualization.
 * WARNING (2019-3-28): this class doesn't have full capabilities of it's current Python
 * HierarchicalOptimizer2dVisualizer counterpart.
 * It only stores up the results for later visualization / video generation.
 */
namespace telementry {
class HierarchicalOptimization2dResultStore {
	HierarchicalOptimization2dResultStore(std::time_t optimization_start_time = std::time(nullptr));

	//void store_per_iteration_data(eig::)

private:
	class PerIterationResultStore{
		std::vector<Eigen::MatrixXf> live_fields;
		std::vector<math::MatrixXv2f> warp_fields;
		std::vector<math::MatrixXv2f> data_term_gradients;
		std::vector<math::MatrixXv2f> tikhonov_term_gradients;
		std::vector<math::MatrixXv2f> final_warp_updates;
	};
	class PerLevelResultStore{
	public:
		PerLevelResultStore() = default;

		void add_iteration_result(Eigen::MatrixXf live_field,
				math::MatrixXv2f warp_field = math::MatrixXv2f(),
				math::MatrixXv2f data_term_gradients = math::MatrixXv2f(),
				math::MatrixXv2f tikhonov_term_gradients = math::MatrixXv2f(),
				math::MatrixXv2f final_warp_updates = math::MatrixXv2f());

		const std::vector<Eigen::MatrixXf> get_live_fields() const;
		const std::vector<math::MatrixXv2f> get_warp_fields() const;
		const std::vector<math::MatrixXv2f> get_data_term_gradients() const;
		const std::vector<math::MatrixXv2f> get_tikhonov_term_gradients() const;
		const std::vector<math::MatrixXv2f> get_final_warp_updates() const;
	private:
		std::vector<Eigen::MatrixXf> live_fields;
		std::vector<math::MatrixXv2f> warp_fields;
		std::vector<math::MatrixXv2f> data_term_gradients;
		std::vector<math::MatrixXv2f> tikhonov_term_gradients;
		std::vector<math::MatrixXv2f> final_warp_updates;
	};

	std::time_t optimization_start_time;
	Eigen::MatrixXf initial_canonical_field;
	Eigen::MatrixXf initial_live_field;

	std::vector<PerLevelResultStore> per_level_results;

};
} //namespace telemetry
