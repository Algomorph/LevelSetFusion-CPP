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
	bp::class_<telemetry::WarpDeltaStatistics2d>("WarpDeltaStatistics2d", bp::init<>())
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
			.def_readwrite("ratio_above_min_threshold",
			&telemetry::WarpDeltaStatistics2d::ratio_above_min_threshold)
			.def_readwrite("length_min",
			&telemetry::WarpDeltaStatistics2d::length_min)
			.def_readwrite("length_max",
			&telemetry::WarpDeltaStatistics2d::length_max)
			.def_readwrite("length_mean",
			&telemetry::WarpDeltaStatistics2d::length_mean)
			.def_readwrite("length_standard_deviation",
			&telemetry::WarpDeltaStatistics2d::length_standard_deviation)
			.def_readwrite("longest_warp_location",
			&telemetry::WarpDeltaStatistics2d::longest_warp_location)
			.def_readwrite("is_largest_below_min_threshold",
			&telemetry::WarpDeltaStatistics2d::is_largest_below_min_threshold)
			.def_readwrite("is_largest_above_max_threshold",
			&telemetry::WarpDeltaStatistics2d::is_largest_above_max_threshold)
			.def("to_array", &telemetry::WarpDeltaStatistics2d::to_array)
			.def("__eq__", &telemetry::WarpDeltaStatistics2d::operator==)
			.def("__ne__", &telemetry::WarpDeltaStatistics2d::operator!=)
			.def(bp::self_ns::str(bp::self_ns::self))
			;

	bp::def("build_warp_delta_statistics2d", &telemetry::build_warp_delta_statistics<math::Vector2i, eig::MatrixXf, math::MatrixXv2f>);

	bp::class_<telemetry::TsdfDifferenceStatistics2d>("TsdfDifferenceStatistics2d", bp::init<>())
			.def(bp::init<float, float, float, float, math::Vector2i>(
			bp::args("difference_min",
					"difference_max",
					"difference_mean",
					"difference_standard_deviation",
					"biggest_difference_location")))
			.def(bp::init<eig::MatrixXf, eig::MatrixXf>(bp::args("canonical_field", "live_field")))
			.def_readwrite("difference_min",
			&telemetry::TsdfDifferenceStatistics2d::difference_min)
			.def_readwrite("difference_max",
			&telemetry::TsdfDifferenceStatistics2d::difference_max)
			.def_readwrite("difference_mean",
			&telemetry::TsdfDifferenceStatistics2d::difference_mean)
			.def_readwrite("difference_standard_deviation",
			&telemetry::TsdfDifferenceStatistics2d::difference_standard_deviation)
			.def_readwrite("biggest_difference_location",
			&telemetry::TsdfDifferenceStatistics2d::biggest_difference_location)
			.def("to_array",
			&telemetry::TsdfDifferenceStatistics2d::to_array)
			.def("__eq__", &telemetry::TsdfDifferenceStatistics2d::operator==)
			.def("__ne__", &telemetry::TsdfDifferenceStatistics2d::operator!=)
			.def(bp::self_ns::str(bp::self_ns::self))
			;

	bp::class_<telemetry::ConvergenceReport2d>("ConvergenceReport2d", bp::init<>())
			.def(bp::init<int, bool, telemetry::WarpDeltaStatistics2d, telemetry::TsdfDifferenceStatistics2d>(
			bp::args("iteration_count",
					"iteration_limit_reached",
					"warp_delta_statistics",
					"tsdf_difference_statistics")))
			.def_readwrite("iteration_count",
			&telemetry::ConvergenceReport2d::iteration_count)
			.def_readwrite("iteration_limit_reached",
			&telemetry::ConvergenceReport2d::iteration_limit_reached)
			.def_readwrite("warp_delta_statistics",
			&telemetry::ConvergenceReport2d::warp_delta_statistics)
			.def_readwrite("tsdf_difference_statistics",
			&telemetry::ConvergenceReport2d::tsdf_difference_statistics)
			.def("__eq__", &telemetry::ConvergenceReport2d::operator==)
			.def("__ne__", &telemetry::ConvergenceReport2d::operator!=)
			.def(bp::self_ns::str(bp::self_ns::self))
			;

	bp::class_<std::vector<telemetry::ConvergenceReport2d>>("ConvergenceReport2dVector")
			.def(bp::vector_indexing_suite<std::vector<telemetry::ConvergenceReport2d>>());

	bp::class_<telemetry::OptimizationIterationData2d>("OptimizationIterationData2d", bp::init<>())
			.def("get_live_fields", &telemetry::OptimizationIterationData2d::get_live_fields)
			.def("get_warp_fields", &telemetry::OptimizationIterationData2d::get_warp_fields)
			.def("get_data_term_gradients", &telemetry::OptimizationIterationData2d::get_data_term_gradients)
			.def("get_tikhonov_term_gradients", &telemetry::OptimizationIterationData2d::get_tikhonov_term_gradients)
			.def("get_frame_count", &telemetry::OptimizationIterationData2d::get_frame_count)
			;

	bp::class_<std::vector<telemetry::OptimizationIterationData2d>>("OptimizationIterationData2dVector")
				.def(bp::vector_indexing_suite<std::vector<telemetry::OptimizationIterationData2d>>());
}
} //namespace python_export
