/*
 * logging.cpp
 *
 *  Created on: Mar 12, 2019
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

void export_telemetry_utilities() {
	bp::class_<telemetry::WarpDeltaStatistics>("WarpDeltaStatistics", bp::init<>())
			.def(bp::init<float, float, float, float, float, math::Vector2i, bool, bool>(
			bp::args("ratio_above_min_threshold",
					"length_min",
					"length_max",
					"length_mean",
					"length_standard_deviation",
					"longest_warp_location",
					"is_largest_below_min_threshold",
					"is_largest_above_max_threshold")
					))
			.def(bp::init<math::MatrixXv2f, eig::MatrixXf, eig::MatrixXf, float, float>(
			bp::args(
					"warp_field",
					"canonical_field",
					"live_field",
					"min_threshold",
					"max_threshold"
					)))
			.def_readwrite("ratio_above_min_threshold",
			&telemetry::WarpDeltaStatistics::ratio_above_min_threshold)
			.def_readwrite("length_min",
			&telemetry::WarpDeltaStatistics::length_min)
			.def_readwrite("length_max",
			&telemetry::WarpDeltaStatistics::length_max)
			.def_readwrite("length_mean",
			&telemetry::WarpDeltaStatistics::length_mean)
			.def_readwrite("length_standard_deviation",
			&telemetry::WarpDeltaStatistics::length_standard_deviation)
			.def_readwrite("longest_warp_location",
			&telemetry::WarpDeltaStatistics::longest_warp_location)
			.def_readwrite("is_largest_below_min_threshold",
			&telemetry::WarpDeltaStatistics::is_largest_below_min_threshold)
			.def_readwrite("is_largest_above_max_threshold",
			&telemetry::WarpDeltaStatistics::is_largest_above_max_threshold)
			.def("to_array", &telemetry::WarpDeltaStatistics::to_array)
			.def("__eq__", &telemetry::WarpDeltaStatistics::operator==)
			.def("__ne__", &telemetry::WarpDeltaStatistics::operator!=)
			.def(bp::self_ns::str(bp::self_ns::self))
			;

	bp::class_<telemetry::TsdfDifferenceStatistics>("TsdfDifferenceStatistics", bp::init<>())
			.def(bp::init<float, float, float, float, math::Vector2i>(
			bp::args("difference_min",
					"difference_max",
					"difference_mean",
					"difference_standard_deviation",
					"biggest_difference_location")))
			.def(bp::init<eig::MatrixXf, eig::MatrixXf>(bp::args("canonical_field", "live_field")))
			.def_readwrite("difference_min",
			&telemetry::TsdfDifferenceStatistics::difference_min)
			.def_readwrite("difference_max",
			&telemetry::TsdfDifferenceStatistics::difference_max)
			.def_readwrite("difference_mean",
			&telemetry::TsdfDifferenceStatistics::difference_mean)
			.def_readwrite("difference_standard_deviation",
			&telemetry::TsdfDifferenceStatistics::difference_standard_deviation)
			.def_readwrite("biggest_difference_location",
			&telemetry::TsdfDifferenceStatistics::biggest_difference_location)
			.def("to_array",
			&telemetry::TsdfDifferenceStatistics::to_array)
			.def("__eq__", &telemetry::TsdfDifferenceStatistics::operator==)
			.def("__ne__", &telemetry::TsdfDifferenceStatistics::operator!=)
			.def(bp::self_ns::str(bp::self_ns::self))
			;

	bp::class_<telemetry::ConvergenceReport>("ConvergenceReport", bp::init<>())
			.def(bp::init<int, bool, telemetry::WarpDeltaStatistics, telemetry::TsdfDifferenceStatistics>(
			bp::args("iteration_count",
					"iteration_limit_reached",
					"warp_delta_statistics",
					"tsdf_difference_statistics")))
			.def_readwrite("iteration_count",
			&telemetry::ConvergenceReport::iteration_count)
			.def_readwrite("iteration_limit_reached",
			&telemetry::ConvergenceReport::iteration_limit_reached)
			.def_readwrite("warp_delta_statistics",
			&telemetry::ConvergenceReport::warp_delta_statistics)
			.def_readwrite("tsdf_difference_statistics",
			&telemetry::ConvergenceReport::tsdf_difference_statistics)
			.def("__eq__", &telemetry::ConvergenceReport::operator==)
			.def("__ne__", &telemetry::ConvergenceReport::operator!=)
			.def(bp::self_ns::str(bp::self_ns::self))
			;

	bp::class_<std::vector<telemetry::ConvergenceReport>>("ConvergenceReportVector")
			.def(bp::vector_indexing_suite<std::vector<telemetry::ConvergenceReport>>());

	bp::class_<telemetry::OptimizationIterationData2d>("OptimizationIterationData", bp::init<>())
			.def("get_live_fields", &telemetry::OptimizationIterationData2d::get_live_fields)
			.def("get_warp_fields", &telemetry::OptimizationIterationData2d::get_warp_fields)
			.def("get_data_term_gradients", &telemetry::OptimizationIterationData2d::get_data_term_gradients)
			.def("get_tikhonov_term_gradients", &telemetry::OptimizationIterationData2d::get_tikhonov_term_gradients)
			.def("get_frame_count", &telemetry::OptimizationIterationData2d::get_frame_count)
			;

	bp::class_<std::vector<telemetry::OptimizationIterationData2d>>("OptimizationIterationDataVector")
				.def(bp::vector_indexing_suite<std::vector<telemetry::OptimizationIterationData2d>>());
}
} //namespace python_export
