/*
 * slavcheva_optimizer.cpp
 *
 *  Created on: Jan 11, 2019
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
#include <iostream>

//libraries
#include <Eigen/Eigen>
#include <boost/python.hpp>
#include <boost/shared_ptr.hpp>

//local
#include "eigen_numpy.hpp"
#include "../nonrigid_optimization/slavcheva/sobolev_optimizer2d.hpp"
#include "../nonrigid_optimization/field_warping.hpp"
#include "../nonrigid_optimization/slavcheva/data_term.hpp"
#include "../logging/warp_delta_statistics.hpp"
#include "../logging/convergence_report.hpp"

namespace bp = boost::python;
namespace nro = nonrigid_optimization;

struct NullDeleter {
	void operator()(const void*) {
	}
};

typedef nro::SobolevOptimizer2d::SobolevParameters SobolevParameters;
typedef nro::Optimizer2d::SharedParameters SharedParameters;

boost::shared_ptr<SobolevParameters> get_shared_sobolev_parameters() {
	return boost::shared_ptr<SobolevParameters>(
			&SobolevParameters::get_instance(), NullDeleter());
}

boost::shared_ptr<SharedParameters> get_shared_shared_parameters() {
	return boost::shared_ptr<SharedParameters>(&SharedParameters::get_instance(), NullDeleter());
}

BOOST_PYTHON_FUNCTION_OVERLOADS(resample_overloads, nonrigid_optimization::py_resample, 4, 8)
BOOST_PYTHON_FUNCTION_OVERLOADS(resample_warp_unchanged_overloads, nonrigid_optimization::py_resample_warp_unchanged, 4,
		8)

namespace python_export {

namespace slavcheva {
void export_auxiliary_functions() {
	bp::def("data_term_at_location", nro::py_data_term_at_location);

	bp::def("resample", nro::py_resample,
			resample_overloads(
					bp::args("warped_live_field", "canonical_field", "warp_field_u", "warp_field_v", "band_union_only",
							"known_values_only", "substitute_original", "truncation_float_threshold"),
					" resample (warp) a 2D scalar field using a 2D vector field, such that new value at (x,y)"
							" is interpolated bilinearly from the original scalar field (warped_live_field) at the location "
							"(x+u, y+v), where"
							" the vector (u,v) is located at (x,y) within the warp field (i.e. u = warp_field_u[y,x]).\n"
							" :param band_union_only when set to True, skips processing locations for which both"
							" warped_live_field and canonical_field contain the values +1 or -1 (truncated sdf values)")
					);
	bp::def("resample_warp_unchanged", nro::py_resample_warp_unchanged,
			resample_warp_unchanged_overloads(
					bp::args("warped_live_field", "canonical_field", "warp_field_u", "warp_field_v", "band_union_only",
							"known_values_only", "substitute_original", "truncation_float_threshold"),
					" resample (warp) a 2D scalar field using a 2D vector field, such that new value at (x,y)"
							" is interpolated bilinearly from the original scalar field (warped_live_field) at the location "
							"(x+u, y+v), where"
							" the vector (u,v) is located at (x,y) within the warp field (i.e. u = warp_field_u[y,x]).\n"
							" :param band_union_only when set to True, skips processing locations for which both"
							" warped_live_field and canonical_field contain the values +1 or -1 (truncated sdf values)")
					);
}
void export_setting_singletons() {
	bp::class_<SobolevParameters, boost::shared_ptr<SobolevParameters>, boost::noncopyable>(
			"SobolevParameters", bp::no_init)
			.def("get_instance", &get_shared_sobolev_parameters)
			.staticmethod("get_instance")
			//.def_readwrite("sobolev_kernel", &SobolevParameters::sobolev_kernel)
			.def("get_sobolev_kernel", &SobolevParameters::get_sobolev_kernel)
			.def("set_sobolev_kernel", &SobolevParameters::set_sobolev_kernel)
			.def_readwrite("smoothing_term_weight", &SobolevParameters::smoothing_term_weight);

	bp::class_<SharedParameters, boost::shared_ptr<SharedParameters>, boost::noncopyable>(
			"SharedParameters", bp::no_init)
			.def("get_instance", &get_shared_shared_parameters)
			.staticmethod("get_instance")

	.def_readwrite("gradient_descent_rate", &SharedParameters::gradient_descent_rate)

	//termination condition parameters
	.def_readwrite("maximum_warp_length_lower_threshold",
			&SharedParameters::maximum_warp_length_lower_threshold)
			.def_readwrite("maximum_warp_length_upper_threshold",
			&SharedParameters::maximum_warp_length_upper_threshold)
			.def_readwrite("maximum_iteration_count", &SharedParameters::maximum_iteration_count)
			.def_readwrite("minimum_iteration_count", &SharedParameters::minimum_iteration_count)

	//logging
	.def_readwrite("enable_focus_spot_analytics", &SharedParameters::enable_focus_spot_analytics)
			.def_readwrite("enable_convergence_reporting",
			&SharedParameters::enable_convergence_reporting)
			.def_readwrite("enable_live_sdf_progression_logging",
			&SharedParameters::enable_live_sdf_progression_logging)
			.def_readwrite("enable_gradient_logging", &SharedParameters::enable_gradient_logging)
			.def_readwrite("enable_gradient_component_logging", &SharedParameters::enable_gradient_component_logging)
			.def_readwrite("enable_warp_statistics_logging", &SharedParameters::enable_warp_statistics_logging)

	//focus spot
	//TODO: math::Vector<X> C++-->numpy converter is needed for this
	.def_readwrite("focus_spot", &SharedParameters::focus_spot);
}

void export_algorithms() {
	bp::class_<nro::SobolevOptimizer2d>("SobolevOptimizer2d", bp::init<>())
			.def("optimize", &nro::SobolevOptimizer2d::optimize)
			.def("get_convergence_report", &nro::SobolevOptimizer2d::get_convergence_report)
			.def("get_warp_statistics_as_matrix", &nro::SobolevOptimizer2d::get_warp_statistics_as_matrix)
			;

}
} // namespace slavcheva
} // namespace python_export

