/*
 * telemetry.tpp
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
//stdlib
#include <vector>
#include <string>

#ifdef __GNUC__
#pragma GCC diagnostic ignored "-Wmaybe-uninitialized"
#endif

//libraries
#include <boost/python.hpp>
#include <boost/python/suite/indexing/vector_indexing_suite.hpp>

//local
#include "telemetry.hpp"
#include "../math/typedefs.hpp"
#include "../telemetry/warp_delta_statistics.hpp"
#include "../telemetry/tsdf_difference_statistics.hpp"
#include "../telemetry/convergence_report.hpp"
#include "../telemetry/optimization_iteration_data.hpp"

namespace bp = boost::python;

namespace python_export {


template<typename Coordinates, typename ScalarContainer, typename VectorContainer>
void export_telemetry_utilities(const char* suffix) {
	auto sufy = [&](const char* name){
			return (std::string(name) + std::string(suffix));
	};
	bp::class_<telemetry::WarpDeltaStatistics<Coordinates>>(sufy("WarpDeltaStatistics").c_str(), bp::init<>())
			.def(bp::init<float, float, float, float, float, Coordinates, bool, bool>(
			bp::args("ratio_above_min_threshold",
					"length_min",
					"length_max",
					"length_mean",
					"length_standard_deviation",
					"longest_warp_location",
					"is_largest_below_min_threshold",
					"is_largest_above_max_threshold")
					))
			.def_readwrite("ratio_above_min_threshold",
			&telemetry::WarpDeltaStatistics<Coordinates>::ratio_above_min_threshold)
			.def_readwrite("length_min",
			&telemetry::WarpDeltaStatistics<Coordinates>::length_min)
			.def_readwrite("length_max",
			&telemetry::WarpDeltaStatistics<Coordinates>::length_max)
			.def_readwrite("length_mean",
			&telemetry::WarpDeltaStatistics<Coordinates>::length_mean)
			.def_readwrite("length_standard_deviation",
			&telemetry::WarpDeltaStatistics<Coordinates>::length_standard_deviation)
			.def_readwrite("longest_warp_location",
			&telemetry::WarpDeltaStatistics<Coordinates>::longest_warp_location)
			.def_readwrite("is_largest_below_min_threshold",
			&telemetry::WarpDeltaStatistics<Coordinates>::is_largest_below_min_threshold)
			.def_readwrite("is_largest_above_max_threshold",
			&telemetry::WarpDeltaStatistics<Coordinates>::is_largest_above_max_threshold)
			.def("to_array", &telemetry::WarpDeltaStatistics<Coordinates>::to_array)
			.def("__eq__", &telemetry::WarpDeltaStatistics<Coordinates>::operator==)
			.def("__ne__", &telemetry::WarpDeltaStatistics<Coordinates>::operator!=)
			.def(bp::self_ns::str(bp::self_ns::self))
			;

	bp::def(sufy("build_warp_delta_statistics_").c_str(),
			&telemetry::build_warp_delta_statistics<Coordinates, ScalarContainer, VectorContainer>);
	//bp::class_<telemetry::TsdfDifferenceStatistics<Coordinates>>(sufy("TsdfDifferenceStatistics"), bp::init<>())
	bp::class_<telemetry::TsdfDifferenceStatistics<Coordinates>>(sufy("TsdfDifferenceStatistics").c_str(), bp::init<>())
			.def(bp::init<float, float, float, float, Coordinates>(
			bp::args("difference_min",
					"difference_max",
					"difference_mean",
					"difference_standard_deviation",
					"biggest_difference_location")))
			.def_readwrite("difference_min",
			&telemetry::TsdfDifferenceStatistics<Coordinates>::difference_min)
			.def_readwrite("difference_max",
			&telemetry::TsdfDifferenceStatistics<Coordinates>::difference_max)
			.def_readwrite("difference_mean",
			&telemetry::TsdfDifferenceStatistics<Coordinates>::difference_mean)
			.def_readwrite("difference_standard_deviation",
			&telemetry::TsdfDifferenceStatistics<Coordinates>::difference_standard_deviation)
			.def_readwrite("biggest_difference_location",
			&telemetry::TsdfDifferenceStatistics<Coordinates>::biggest_difference_location)
			.def("to_array",
			&telemetry::TsdfDifferenceStatistics<Coordinates>::to_array)
			.def("__eq__", &telemetry::TsdfDifferenceStatistics<Coordinates>::operator==)
			.def("__ne__", &telemetry::TsdfDifferenceStatistics<Coordinates>::operator!=)
			.def(bp::self_ns::str(bp::self_ns::self))
			;

	bp::def(sufy("build_tsdf_difference_statistics_").c_str(),
			&telemetry::build_tsdf_difference_statistics<Coordinates, ScalarContainer>,
			bp::args("canonical_tsdf", "live_tsdf"));

	bp::class_<telemetry::ConvergenceReport<Coordinates>>(sufy("ConvergenceReport").c_str(), bp::init<>())
			.def(bp::init<int, bool, telemetry::WarpDeltaStatistics<Coordinates>, telemetry::TsdfDifferenceStatistics<Coordinates>>(
			bp::args("iteration_count",
					"iteration_limit_reached",
					"warp_delta_statistics",
					"tsdf_difference_statistics")))
			.def_readwrite("iteration_count",
			&telemetry::ConvergenceReport<Coordinates>::iteration_count)
			.def_readwrite("iteration_limit_reached",
			&telemetry::ConvergenceReport<Coordinates>::iteration_limit_reached)
			.def_readwrite("warp_delta_statistics",
			&telemetry::ConvergenceReport<Coordinates>::warp_delta_statistics)
			.def_readwrite("tsdf_difference_statistics",
			&telemetry::ConvergenceReport<Coordinates>::tsdf_difference_statistics)
			.def("__eq__", &telemetry::ConvergenceReport<Coordinates>::operator==)
			.def("__ne__", &telemetry::ConvergenceReport<Coordinates>::operator!=)
			.def(bp::self_ns::str(bp::self_ns::self))
			;

	bp::class_<std::vector<telemetry::ConvergenceReport<Coordinates>>>(sufy("ConvergenceReportVector").c_str())
			.def(bp::vector_indexing_suite<std::vector<telemetry::ConvergenceReport<Coordinates>>>());

	bp::class_<telemetry::OptimizationIterationData<ScalarContainer,VectorContainer>>(sufy("OptimizationIterationData").c_str(), bp::init<>())
			.def("get_live_fields", &telemetry::OptimizationIterationData<ScalarContainer, VectorContainer>::get_live_fields)
			.def("get_warp_fields", &telemetry::OptimizationIterationData<ScalarContainer, VectorContainer>::get_warp_fields)
			.def("get_data_term_gradients", &telemetry::OptimizationIterationData<ScalarContainer, VectorContainer>::get_data_term_gradients)
			.def("get_tikhonov_term_gradients", &telemetry::OptimizationIterationData<ScalarContainer, VectorContainer>::get_tikhonov_term_gradients)
			.def("get_frame_count", &telemetry::OptimizationIterationData<ScalarContainer, VectorContainer>::get_frame_count)
			;

	bp::class_<std::vector<telemetry::OptimizationIterationData<ScalarContainer,VectorContainer>>>(sufy("OptimizationIterationDataVector").c_str())
			.def(bp::vector_indexing_suite<std::vector<telemetry::OptimizationIterationData<ScalarContainer,VectorContainer>>>());

}

} //namespace python_export

