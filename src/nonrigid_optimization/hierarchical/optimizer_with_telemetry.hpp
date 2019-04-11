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
#include "../../math/typedefs.hpp"
#include "../../telemetry/convergence_report.hpp"
#include "../../telemetry/optimization_iteration_data.hpp"
#include "optimizer.hpp"

namespace nonrigid_optimization {
namespace hierarchical {

typedef std::chrono::time_point<std::chrono::high_resolution_clock> time_point;

template<typename ScalarContainer, typename VectorContainer>
class OptimizerWithTelemetry:
		public Optimizer<ScalarContainer, VectorContainer> {
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

	OptimizerWithTelemetry(
			bool tikhonov_term_enabled = true,
			bool gradient_kernel_enabled = true,

			int maximum_chunk_size = 8,
			float rate = 0.1f,
			int maximum_iteration_count = 100,
			float maximum_warp_update_threshold = 0.001f,

			float data_term_amplifier = 1.0f,
			float tikhonov_strength = 0.2f,
			eig::VectorXf kernel = eig::VectorXf(0),

			typename Optimizer<ScalarContainer, VectorContainer>::ResamplingStrategy resampling_strategy = Optimizer<ScalarContainer, VectorContainer>::ResamplingStrategy::NEAREST_AND_AVERAGE,

			VerbosityParameters verbosity_parameters = VerbosityParameters(),
			LoggingParameters logging_parameters = LoggingParameters());

	VectorContainer optimize(const ScalarContainer& canonical_field, const ScalarContainer& live_field) override;
	void optimize_level(
			VectorContainer& warp_field,
			const ScalarContainer& canonical_pyramid_level,
			const ScalarContainer& live_pyramid_level,
			const VectorContainer& live_gradient_level
			) override;
	void optimize_iteration(
			VectorContainer& gradient,
			VectorContainer& warp_field,
			ScalarContainer& diff,
			VectorContainer& data_gradient,
			VectorContainer& tikhonov_gradient,
			float& maximum_warp_update_length,
			const ScalarContainer& canonical_pyramid_level,
			const ScalarContainer& live_pyramid_level,
			const VectorContainer& live_gradient_level
			) override;
	std::vector<telemetry::ConvergenceReport> get_per_level_convergence_reports();
	std::vector<telemetry::OptimizationIterationData<ScalarContainer,VectorContainer>> get_per_level_iteration_data();

private:
	//TODO: set these and provide retrieval methods as appropriate
	time_point optimization_start = std::chrono::high_resolution_clock::now();
	double optimization_duration = 0.0;

	const float energy_factor = 1000000.0f;
	std::vector<telemetry::ConvergenceReport> per_level_convergence_reports;
	std::vector<telemetry::OptimizationIterationData<ScalarContainer,VectorContainer>> per_level_iteration_data;
	void clear_logs();

	const VerbosityParameters verbosity_parameters;
	const LoggingParameters logging_parameters;
};

typedef OptimizerWithTelemetry<Eigen::MatrixXf,math::MatrixXv2f> OptimizerWithTelemetry2d;

} // namespace hierarchical
} // namespace nonrigid_optimization
