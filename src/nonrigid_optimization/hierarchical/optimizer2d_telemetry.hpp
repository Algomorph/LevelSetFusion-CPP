/*
 * optimizer2d_telemetry.hpp
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

#pragma once

//stdlib
#include <chrono>

//local
#include "optimizer2d.hpp"
#include "../../math/tensors.hpp"
#include "../../telemetry/convergence_report.hpp"
#include "../../telemetry/optimization_iteration_data.hpp"

namespace nonrigid_optimization {
namespace hierarchical {

typedef std::chrono::time_point<std::chrono::high_resolution_clock> time_point;

class Optimizer2dTelemetry:
		public Optimizer2d {
public:
	struct VerbosityParameters {
		VerbosityParameters(
				bool print_max_warp_update = false,
				bool print_iteration_mean_tsdf_difference = false,
				bool print_iteration_std_tsdf_difference = false,
				bool print_iteration_data_energy = false,
				bool print_iteration_tikhonov_energy = false);
		//per-iteration parameters
		const bool print_iteration_max_warp_update = false;
		const bool print_iteration_mean_tsdf_difference = false;
		const bool print_iteration_std_tsdf_difference = false;
		const bool print_iteration_data_energy = false;
		const bool print_iteration_tikhonov_energy = false;
		const bool print_per_iteration_info = false;
		const bool print_per_level_info = false;
	};

	struct LoggingParameters {
		LoggingParameters(
				bool collect_per_level_convergence_reports = false,
				bool collect_per_level_iteration_data = false
				);
		const bool collect_per_level_convergence_reports = false;
		const bool collect_per_level_iteration_data = false;
	};

	Optimizer2dTelemetry(
			bool tikhonov_term_enabled = true,
			bool gradient_kernel_enabled = true,

			int maximum_chunk_size = 8,
			float rate = 0.1f,
			int maximum_iteration_count = 100,
			float maximum_warp_update_threshold = 0.001f,

			float data_term_amplifier = 1.0f,
			float tikhonov_strength = 0.2f,
			eig::VectorXf kernel = eig::VectorXf(0),

			VerbosityParameters verbosity_parameters = VerbosityParameters(),
			LoggingParameters logging_parameters = LoggingParameters());

	math::MatrixXv2f optimize(eig::MatrixXf canonical_field, eig::MatrixXf live_field) override;
	void optimize_level(
			math::MatrixXv2f& warp_field,
			const eig::MatrixXf& canonical_pyramid_level,
			const eig::MatrixXf& live_pyramid_level,
			const eig::MatrixXf& live_gradient_x_level,
			const eig::MatrixXf& live_gradient_y_level
			) override;
	void optimize_iteration(
			math::MatrixXv2f& gradient,
			math::MatrixXv2f& warp_field,
			eig::MatrixXf& diff,
			math::MatrixXv2f& data_gradient,
			math::MatrixXv2f& tikhonov_gradient,
			float& maximum_warp_update_length,
			const eig::MatrixXf& canonical_pyramid_level,
			const eig::MatrixXf& live_pyramid_level,
			const eig::MatrixXf& live_gradient_x_level,
			const eig::MatrixXf& live_gradient_y_level
			) override;
	std::vector<telemetry::ConvergenceReport> get_per_level_convergence_reports();
	std::vector<telemetry::OptimizationIterationData> get_per_level_iteration_data();

private:
	//TODO: set these and provide retrieval methods as appropriate
	time_point optimization_start = std::chrono::high_resolution_clock::now();
	double optimization_duration = 0.0;

	const float energy_factor = 1000000.0f;
	std::vector<telemetry::ConvergenceReport> per_level_convergence_reports;
	std::vector<telemetry::OptimizationIterationData> per_level_iteration_data;
	void clear_logs();

	const VerbosityParameters verbosity_parameters;
	const LoggingParameters logging_parameters;
};

} // namespace hierarchical
} // namespace nonrigid_optimization
