/*
 * hierarchical_optimizer.tpp
 *
 *  Created on: Apr 25, 2019
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

//libraries
#include <Eigen/Eigen>
#include <boost/python.hpp>
#include <boost/shared_ptr.hpp>

//local
#include "hierarchical_optimizer.hpp"
#include "../nonrigid_optimization/hierarchical/optimizer_with_telemetry.hpp"
#include "eigen_numpy.hpp"
#include "../math/typedefs.hpp"

namespace bp = boost::python;
namespace impl = nonrigid_optimization::hierarchical;
namespace eig = Eigen;

namespace python_export {
namespace hierarchical_optimizer {

template<typename ScalarContainer, typename VectorContainer>
void export_algorithms(const char* suffix) {
	auto sufy = [&](const char* name) {
		return (std::string(name) + std::string(suffix));
	};
	bp::scope outer =
			bp::class_<impl::OptimizerWithTelemetry<ScalarContainer, VectorContainer>>(sufy("HierarchicalOptimizer").c_str(),
					bp::init<
							bp::optional<
									bool, bool,
									int, float, int, float,
									float, float, eig::VectorXf,
									typename impl::OptimizerWithTelemetry<ScalarContainer, VectorContainer>::ResamplingStrategy,
									typename impl::OptimizerWithTelemetry<ScalarContainer, VectorContainer>::VerbosityParameters,
									typename impl::OptimizerWithTelemetry<ScalarContainer, VectorContainer>::LoggingParameters> >(
							bp::args("tikhonov_term_enabled",
									"gradient_kernel_enabled",

									"maximum_chunk_size",
									"rate",
									"maximum_iteration_count",
									"maximum_warp_update_threshold",

									"data_term_amplifier",
									"tikhonov_strength",
									"kernel",

									"resampling_strategy",

									"verbosity_parameters",
									"logging_parameters"
									)))
					.def("optimize", &impl::OptimizerWithTelemetry<ScalarContainer, VectorContainer>::optimize,
					"Find optimal warp to map given live SDF to given canonical SDF",
					bp::args("canonical_field", "live_field"))
					.def("get_per_level_convergence_reports",
					&impl::OptimizerWithTelemetry<ScalarContainer, VectorContainer>::get_per_level_convergence_reports)
					.def("get_per_level_iteration_data",
					&impl::OptimizerWithTelemetry<ScalarContainer, VectorContainer>::get_per_level_iteration_data)
					;
	bp::enum_<typename impl::OptimizerWithTelemetry<ScalarContainer, VectorContainer>::ResamplingStrategy>(
			"ResamplingStrategy")
			.value("NEAREST_AND_AVERAGE",
			impl::OptimizerWithTelemetry<ScalarContainer, VectorContainer>::ResamplingStrategy::NEAREST_AND_AVERAGE)
			.value("LINEAR",
			impl::OptimizerWithTelemetry<ScalarContainer, VectorContainer>::ResamplingStrategy::LINEAR);
	bp::class_<typename impl::OptimizerWithTelemetry<ScalarContainer, VectorContainer>::VerbosityParameters>(
			"VerbosityParameters",
			"Parameters that control verbosity to stdout. "
					"Assumes being used in an \"immutable\" manner, i.e. just a structure that holds values",
			bp::init<bp::optional<bool, bool, bool, bool, bool>>(
					bp::args(/*"self",*/
					"print_max_warp_update",
							"print_iteration_mean_tsdf_difference",
							"print_iteration_std_tsdf_difference",
							"print_iteration_data_energy",
							"print_iteration_tikhonov_energy")))
			.add_property("print_iteration_max_warp_update",
			&impl::OptimizerWithTelemetry<ScalarContainer, VectorContainer>::VerbosityParameters
			::print_iteration_max_warp_update)
			.add_property("print_iteration_mean_tsdf_difference",
			&impl::OptimizerWithTelemetry<ScalarContainer, VectorContainer>::VerbosityParameters
			::print_iteration_mean_tsdf_difference)
			.add_property("print_iteration_std_tsdf_difference",
			&impl::OptimizerWithTelemetry<ScalarContainer, VectorContainer>::VerbosityParameters
			::print_iteration_std_tsdf_difference)
			.add_property("print_iteration_data_energy",
			&impl::OptimizerWithTelemetry<ScalarContainer, VectorContainer>::VerbosityParameters
			::print_iteration_data_energy)
			.add_property("print_iteration_tikhonov_energy",
			&impl::OptimizerWithTelemetry<ScalarContainer, VectorContainer>::VerbosityParameters
			::print_iteration_tikhonov_energy)
			//============================================
			.add_property("print_per_iteration_info",
			&impl::OptimizerWithTelemetry<ScalarContainer, VectorContainer>::VerbosityParameters
			::print_per_iteration_info)
			.add_property("print_per_level_info",
			&impl::OptimizerWithTelemetry<ScalarContainer, VectorContainer>::VerbosityParameters
			::print_per_level_info)
			;

	bp::class_<typename impl::OptimizerWithTelemetry<ScalarContainer, VectorContainer>::LoggingParameters>(
			"LoggingParameters",
			"Parameters that control what intermediate results are gathered during run",
			bp::init<bp::optional<bool, bool>>(
					bp::args(
							"collect_per_level_convergence_reports",
							"collect_per_level_iteration_data"
							)))
			.add_property("collect_per_level_convergence_reports",
			&impl::OptimizerWithTelemetry<ScalarContainer, VectorContainer>::LoggingParameters::collect_per_level_convergence_reports)
			.add_property("collect_per_level_iteration_data",
			&impl::OptimizerWithTelemetry<ScalarContainer, VectorContainer>::LoggingParameters::collect_per_level_iteration_data)
			;

}

} // namespace hierarchical_optimizer
} // namespace python_export

