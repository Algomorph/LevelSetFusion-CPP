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
#include <cfloat>

//local
#include "optimizer2d_telemetry.hpp"
#include "../../math/gradients.hpp"

namespace nonrigid_optimization {
namespace hierarchical {

Optimizer2dTelemetry::Optimizer2dTelemetry(
		bool tikhonov_term_enabled,
		bool gradient_kernel_enabled,

		int maximum_chunk_size,
		float rate,
		int maximum_iteration_count,
		float maximum_warp_update_threshold,

		float data_term_amplifier,
		float tikhonov_strength,
		eig::VectorXf kernel,

		VerbosityParameters verbosity_parameters,
		LoggingParameters logging_parameters
		) :
		Optimizer2d(
				tikhonov_term_enabled,
				gradient_kernel_enabled,

				maximum_chunk_size,
				rate,
				maximum_iteration_count,
				maximum_warp_update_threshold,

				data_term_amplifier,
				tikhonov_strength,
				kernel),

				verbosity_parameters(verbosity_parameters),
				logging_parameters(logging_parameters)
{}

math::MatrixXv2f Optimizer2dTelemetry::optimize(eig::MatrixXf canonical_field, eig::MatrixXf live_field) {
	this->clear_logs();
	math::MatrixXv2f result = Optimizer2d::optimize(canonical_field, live_field);
	return result;
}

void Optimizer2dTelemetry::optimize_level(
		math::MatrixXv2f& warp_field,
		const eig::MatrixXf& canonical_pyramid_level,
		const eig::MatrixXf& live_pyramid_level,
		const eig::MatrixXf& live_gradient_x_level,
		const eig::MatrixXf& live_gradient_y_level
		) {
	Optimizer2d::optimize_level(warp_field, canonical_pyramid_level, live_pyramid_level,
			live_gradient_x_level, live_gradient_y_level);
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
		telemetry::TsdfDifferenceStatistics tsdf_difference_statistics(canonical_pyramid_level, live_pyramid_level);
		int current_iteration = this->get_current_iteration();
		this->per_level_convergence_reports.push_back( {
				current_iteration,
				current_iteration >= this->maximum_iteration_count,
				current_warp_statistics,
				tsdf_difference_statistics
		});
	}
}

void Optimizer2dTelemetry::optimize_iteration(
		math::MatrixXv2f& gradient,
		math::MatrixXv2f& warp_field,
		eig::MatrixXf& diff,
		float& maximum_warp_update_length,
		const eig::MatrixXf& canonical_pyramid_level,
		const eig::MatrixXf& live_pyramid_level,
		const eig::MatrixXf& live_gradient_x_level,
		const eig::MatrixXf& live_gradient_y_level
		) {
	float normalized_tikhonov_energy;
	if (this->tikhonov_term_enabled && this->verbosity_parameters.print_iteration_tikhonov_energy) {
		eig::MatrixXf gradient_u_component, gradient_v_component;
		math::unstack_xv2f(gradient_u_component, gradient_v_component, gradient);
		eig::MatrixXf u_x, u_y, v_x, v_y;
		float gradient_aggregate_mean;
		math::scalar_field_gradient(gradient_u_component, u_x, u_y);
		math::scalar_field_gradient(gradient_v_component, v_x, v_y);
		gradient_aggregate_mean = (u_x.array().square() + u_y.array().square()
				+ v_x.array().square() + v_y.array().square()).mean();
		normalized_tikhonov_energy = energy_factor * 0.5 * gradient_aggregate_mean;
	}
	Optimizer2d::optimize_iteration(
			gradient, warp_field, diff, maximum_warp_update_length,
			canonical_pyramid_level, live_pyramid_level,
			live_gradient_x_level, live_gradient_y_level);

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

void Optimizer2dTelemetry::clear_logs() {
	this->per_level_convergence_reports.clear();
}

std::vector<telemetry::ConvergenceReport> Optimizer2dTelemetry::get_per_level_convergence_reports() {
	return this->per_level_convergence_reports;
}

Optimizer2dTelemetry::VerbosityParameters::VerbosityParameters(
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

Optimizer2dTelemetry::LoggingParameters::LoggingParameters(bool collect_per_level_convergence_reports) :
		collect_per_level_convergence_reports(collect_per_level_convergence_reports) {
}

} // namespace hierarchical
} // namespace nonrigid_optimization

