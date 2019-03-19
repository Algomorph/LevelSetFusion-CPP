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
#include "logging.hpp"
#include "../telemetry/warp_delta_statistics.hpp"
#include "../telemetry/tsdf_difference_statistics.hpp"
#include "../telemetry/convergence_report.hpp"
#include "../math/typedefs.hpp"
#include "../math/tensors.hpp"

namespace bp = boost::python;

namespace python_export {

void export_logging_utilities() {
	bp::class_<logging::WarpDeltaStatistics>("WarpDeltaStatistics", bp::init<>())
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
			&logging::WarpDeltaStatistics::ratio_above_min_threshold)
			.def_readwrite("length_min",
			&logging::WarpDeltaStatistics::length_min)
			.def_readwrite("length_max",
			&logging::WarpDeltaStatistics::length_max)
			.def_readwrite("length_mean",
			&logging::WarpDeltaStatistics::length_mean)
			.def_readwrite("length_standard_deviation",
			&logging::WarpDeltaStatistics::length_standard_deviation)
			.def_readwrite("longest_warp_location",
			&logging::WarpDeltaStatistics::longest_warp_location)
			.def_readwrite("is_largest_below_min_threshold",
			&logging::WarpDeltaStatistics::is_largest_below_min_threshold)
			.def_readwrite("is_largest_above_max_threshold",
			&logging::WarpDeltaStatistics::is_largest_above_max_threshold)
			.def("to_array", &logging::WarpDeltaStatistics::to_array)
			.def("__eq__", &logging::WarpDeltaStatistics::operator==)
			.def("__ne__", &logging::WarpDeltaStatistics::operator!=)
			.def(bp::self_ns::str(bp::self_ns::self))
			;

	bp::class_<logging::TsdfDifferenceStatistics>("TsdfDifferenceStatistics", bp::init<>())
			.def(bp::init<float, float, float, float, math::Vector2i>(
			bp::args("difference_min",
					"difference_max",
					"difference_mean",
					"difference_standard_deviation",
					"biggest_difference_location")))
			.def(bp::init<eig::MatrixXf, eig::MatrixXf>(bp::args("canonical_field", "live_field")))
			.def_readwrite("difference_min",
			&logging::TsdfDifferenceStatistics::difference_min)
			.def_readwrite("difference_max",
			&logging::TsdfDifferenceStatistics::difference_max)
			.def_readwrite("difference_mean",
			&logging::TsdfDifferenceStatistics::difference_mean)
			.def_readwrite("difference_standard_deviation",
			&logging::TsdfDifferenceStatistics::difference_standard_deviation)
			.def_readwrite("biggest_difference_location",
			&logging::TsdfDifferenceStatistics::biggest_difference_location)
			.def("to_array",
			&logging::TsdfDifferenceStatistics::to_array)
			.def("__eq__", &logging::TsdfDifferenceStatistics::operator==)
			.def("__ne__", &logging::TsdfDifferenceStatistics::operator!=)
			.def(bp::self_ns::str(bp::self_ns::self))
			;

	bp::class_<logging::ConvergenceReport>("ConvergenceReport", bp::init<>())
			.def(bp::init<int, bool, logging::WarpDeltaStatistics, logging::TsdfDifferenceStatistics>(
			bp::args("iteration_count",
					"iteration_limit_reached",
					"warp_delta_statistics",
					"tsdf_difference_statistics")))
			.def_readwrite("iteration_count",
			&logging::ConvergenceReport::iteration_count)
			.def_readwrite("iteration_limit_reached",
			&logging::ConvergenceReport::iteration_limit_reached)
			.def_readwrite("warp_delta_statistics",
			&logging::ConvergenceReport::warp_delta_statistics)
			.def_readwrite("tsdf_difference_statistics",
			&logging::ConvergenceReport::tsdf_difference_statistics)
			.def("__eq__", &logging::ConvergenceReport::operator==)
			.def("__ne__", &logging::ConvergenceReport::operator!=)
			.def(bp::self_ns::str(bp::self_ns::self))
			;

	bp::class_<std::vector<logging::ConvergenceReport>>("ConvergenceReportVector")
			.def(bp::vector_indexing_suite<std::vector<logging::ConvergenceReport>>());

}
} //namespace python_export

