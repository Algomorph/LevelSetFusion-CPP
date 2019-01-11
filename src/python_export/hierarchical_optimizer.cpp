/*
 * hierarchical_optimizer.cpp
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
//libraries
#include <Eigen/Eigen>
#include <boost/python.hpp>
#include <boost/shared_ptr.hpp>
//local
#include "../nonrigid_optimization/hierarchical_optimizer2d.hpp"

namespace bp = boost::python;
namespace nro = nonrigid_optimization;

//VerbosityParameters verbosity_parameters = VerbosityParameters(),
//			int maximum_chunk_size = 8,
//			float rate = 0.1f,
//			float data_term_amplifier = 1.0f,
//			float tikhonov_strength = 0.2f,
//			eig::VectorXf kernel = eig::VectorXf(0),
//			float maximum_warp_update_threshold = 0.001f,
//			int maximum_iteration_count = 100,
//			bool tikhonov_term_enabled = true,
//			bool gradient_kernel_enabled = true);
namespace python_export{
namespace hierarchical_optimizer{
void export_algorithms(){
	{
		bp::scope outer =
				bp::class_<nro::HierarchicalOptimizer2d>("HierarchicalOptimizer",
						bp::init<bp::optional<nro::HierarchicalOptimizer2d::VerbosityParameters,
						int,float,float,float,eig::VectorXf,float,int,bool,bool> >(
								bp::args("self", "verbosity_parameters",
										"maximum_chunk_size",
										"rate", "data_term_amplifier", "tikhonov_strength","kernel",
										"maximum_warp_update_threshold","maximum_iteration_count",
										"tikhonov_term_enabled", "gradient_kernel_enabled")))
						.def("optimize", &nro::HierarchicalOptimizer2d::optimize,
								"Find optimal warp to map given live SDF to given canonical SDF",
								bp::args("canonical_field","live_field"))
						;
		bp::class_<nro::HierarchicalOptimizer2d::VerbosityParameters>("VerbosityParameters",
				bp::init<bool,bool,bool>());

	}
}
} // namespace hierarchical_optimizer
} // namespace python_export
