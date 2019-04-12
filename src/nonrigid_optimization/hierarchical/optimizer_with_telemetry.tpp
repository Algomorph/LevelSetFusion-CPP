/*
 * optimizer2d_telemetry.cpp
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
//stdlib
#include "optimizer_with_telemetry.hpp"

#include <cfloat>

//local
#include "../../math/gradients.hpp"
#include "../../math/field_like.hpp"

namespace nonrigid_optimization {
namespace hierarchical {

template<typename ScalarContainer, typename VectorContainer>
OptimizerWithTelemetry<ScalarContainer,VectorContainer>::OptimizerWithTelemetry(
		bool tikhonov_term_enabled,
		bool gradient_kernel_enabled,

		int maximum_chunk_size,
		float rate,
		int maximum_iteration_count,
		float maximum_warp_update_threshold,

		float data_term_amplifier,
		float tikhonov_strength,
		eig::VectorXf kernel,

		typename Optimizer<ScalarContainer,VectorContainer>::ResamplingStrategy resampling_strategy,

		VerbosityParameters verbosity_parameters,
		LoggingParameters logging_parameters) :
		Optimizer<ScalarContainer,VectorContainer>(
				tikhonov_term_enabled,
				gradient_kernel_enabled,

				maximum_chunk_size,
				rate,
				maximum_iteration_count,
				maximum_warp_update_threshold,

				data_term_amplifier,
				tikhonov_strength,
				kernel,

				resampling_strategy),

		verbosity_parameters(verbosity_parameters),
				logging_parameters(logging_parameters)
{
}

template<typename ScalarContainer, typename VectorContainer>
VectorContainer OptimizerWithTelemetry<ScalarContainer,VectorContainer>
	::optimize(const ScalarContainer& canonical_field, const ScalarContainer& live_field) {
	this->clear_logs();
	VectorContainer result = Optimizer<ScalarContainer,VectorContainer>::optimize(canonical_field, live_field);
	return result;
}

template<typename ScalarContainer, typename VectorContainer>
void OptimizerWithTelemetry<ScalarContainer,VectorContainer>::optimize_level(
		VectorContainer& warp_field,
		const ScalarContainer& canonical_pyramid_level,
		const ScalarContainer& live_pyramid_level,
		const VectorContainer& live_gradient_level) {

	if (this->logging_parameters.collect_per_level_iteration_data) {

		telemetry::OptimizationIterationData<ScalarContainer,VectorContainer> level_optimization_data;
		if(this->get_current_hierarchy_level() == 0){
			VectorContainer empty = math::vector_field_like(live_pyramid_level);
			level_optimization_data.add_iteration_result(
					live_pyramid_level,
					empty,
					empty,
					(this->tikhonov_term_enabled ? empty : VectorContainer()));
		}
		this->per_level_iteration_data.push_back(level_optimization_data);
	}
	Optimizer<ScalarContainer,VectorContainer>::optimize_level(warp_field, canonical_pyramid_level, live_pyramid_level,
			live_gradient_level);
	if (this->verbosity_parameters.print_per_level_info) {
		std::cout << "[LEVEL " << this->get_current_hierarchy_level() << " COMPLETED]";
		std::cout << std::endl;
	}
	if (this->logging_parameters.collect_per_level_convergence_reports) {
		telemetry::WarpDeltaStatistics current_warp_statistics(warp_field,
				canonical_pyramid_level,
				live_pyramid_level,
				this->maximum_warp_update_threshold,
				FLT_MAX);
		telemetry::TsdfDifferenceStatistics<Coordinates> tsdf_difference_statistics(canonical_pyramid_level, live_pyramid_level);
		int current_iteration = this->get_current_iteration();
		this->per_level_convergence_reports.push_back( {
				current_iteration,
				current_iteration >= this->maximum_iteration_count,
				current_warp_statistics,
				tsdf_difference_statistics
		});
	}
}

template<typename ScalarContainer, typename VectorContainer>
void OptimizerWithTelemetry<ScalarContainer,VectorContainer>::optimize_iteration(
		VectorContainer& gradient,
		VectorContainer& warp_field,
		ScalarContainer& diff,
		VectorContainer& data_gradient,
		VectorContainer& tikhonov_gradient,
		float& maximum_warp_update_length,
		const ScalarContainer& canonical_pyramid_level,
		const ScalarContainer& live_pyramid_level,
		const VectorContainer& live_gradient_level
		) {
	float normalized_tikhonov_energy;
	if (this->tikhonov_term_enabled && this->verbosity_parameters.print_iteration_tikhonov_energy) {
		eig::MatrixXf gradient_u_component, gradient_v_component;
		math::unstack_xv2f(gradient_u_component, gradient_v_component, gradient);
		eig::MatrixXf u_x, u_y, v_x, v_y;
		float gradient_aggregate_mean;
		math::gradient(u_x, u_y, gradient_u_component);
		math::gradient(v_x, v_y, gradient_v_component);
		gradient_aggregate_mean = (u_x.array().square() + u_y.array().square()
				+ v_x.array().square() + v_y.array().square()).mean();
		normalized_tikhonov_energy = energy_factor * 0.5 * gradient_aggregate_mean;
	}
	Optimizer<ScalarContainer,VectorContainer>::optimize_iteration(
			gradient, warp_field, diff, data_gradient, tikhonov_gradient,
			maximum_warp_update_length,
			canonical_pyramid_level, live_pyramid_level, live_gradient_level);

	if (this->logging_parameters.collect_per_level_iteration_data) {
		this->per_level_iteration_data[this->per_level_iteration_data.size() - 1].add_iteration_result(
				live_pyramid_level,
				warp_field,
				data_gradient,
				tikhonov_gradient);
	}

	if (this->verbosity_parameters.print_per_iteration_info) {
		std::cout << "[ITERATION " << this->get_current_iteration() << " COMPLETED]";
		if (this->verbosity_parameters.print_iteration_max_warp_update) {
			std::cout << " [max upd. l.: " << maximum_warp_update_length << "]";
		}
		if (this->verbosity_parameters.print_iteration_mean_tsdf_difference) {
			std::cout << " [mean diff.: " << diff.mean() << "]";
		}
		if (this->verbosity_parameters.print_iteration_std_tsdf_difference) {
			float mean = diff.mean();
			float count = static_cast<float>(diff.size());
			float std_dev = std::sqrt((diff.array() - mean).square().sum() / count);
			std::cout << " [std diff.: " << std_dev << "]";
		}
		if (this->verbosity_parameters.print_iteration_data_energy) {
			float normalized_data_energy = energy_factor * diff.array().square().mean();
			std::cout << " [norm. data energy: " << normalized_data_energy << "]";
		}
		if (this->verbosity_parameters.print_iteration_tikhonov_energy && this->tikhonov_term_enabled) {
			std::cout << " [norm. tikhonov energy: " << normalized_tikhonov_energy << "]";
		}
		std::cout << std::endl;
	}
}

template<typename ScalarContainer, typename VectorContainer>
void OptimizerWithTelemetry<ScalarContainer,VectorContainer>::clear_logs() {
	this->per_level_convergence_reports.clear();
	this->per_level_iteration_data.clear();
}

template<typename ScalarContainer, typename VectorContainer>
std::vector<telemetry::ConvergenceReport<typename OptimizerWithTelemetry<ScalarContainer,VectorContainer>::Coordinates>>
OptimizerWithTelemetry<ScalarContainer,VectorContainer>::get_per_level_convergence_reports() {
	return this->per_level_convergence_reports;
}

template<typename ScalarContainer, typename VectorContainer>
std::vector<telemetry::OptimizationIterationData<ScalarContainer,VectorContainer>> OptimizerWithTelemetry<ScalarContainer,VectorContainer>
	::get_per_level_iteration_data() {
	return this->per_level_iteration_data;
}

template<typename ScalarContainer, typename VectorContainer>
OptimizerWithTelemetry<ScalarContainer,VectorContainer>::VerbosityParameters::VerbosityParameters(
		bool print_iteration_max_warp_update,
		bool print_iteration_mean_tsdf_difference,
		bool print_iteration_std_tsdf_difference,
		bool print_iteration_data_energy,
		bool print_iteration_tikhonov_energy) :
		print_iteration_max_warp_update(print_iteration_max_warp_update),
				print_iteration_mean_tsdf_difference(print_iteration_mean_tsdf_difference),
				print_iteration_std_tsdf_difference(print_iteration_std_tsdf_difference),
				print_iteration_data_energy(print_iteration_data_energy),
				print_iteration_tikhonov_energy(print_iteration_tikhonov_energy),

				print_per_iteration_info(
						print_iteration_max_warp_update ||
								print_iteration_data_energy ||
								print_iteration_tikhonov_energy
								),

				print_per_level_info(print_per_iteration_info)
{
}

template<typename ScalarContainer, typename VectorContainer>
OptimizerWithTelemetry<ScalarContainer,VectorContainer>::LoggingParameters::LoggingParameters(
		bool collect_per_level_convergence_reports,
		bool collect_per_level_iteration_data
		) :
		collect_per_level_convergence_reports(collect_per_level_convergence_reports),
				collect_per_level_iteration_data(collect_per_level_iteration_data) {
}

} // namespace hierarchical
} // namespace nonrigid_optimization
