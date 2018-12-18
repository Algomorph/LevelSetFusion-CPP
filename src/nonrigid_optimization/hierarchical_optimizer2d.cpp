/*
 * hierarchical_optimizer2d.cpp
 *
 *  Created on: Dec 18, 2018
 *      Author: Gregory Kramida
 *   Copyright: 2018 Gregory Kramida
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

#include "hierarchical_optimizer2d.h"

namespace nonrigid_optimization {

HierarchicalOptimizer2d::HierarchicalOptimizer2d(
		int maximum_chunk_size,
		float rate,
		float data_term_amplifier,
		float tikhonov_strength,
		eig::VectorXf kernel,
		float maximum_warp_update_threshold,
		int maximum_iteration_count,
		bool tikhonov_term_enabled,
		bool gradient_kernel_enabled):
				maximum_chunk_size(maximum_chunk_size),
				rate(rate),
				data_term_amplifier(data_term_amplifier),
				tikhonov_strength(tikhonov_strength),
				kernel(kernel),
				maximum_warp_update_threshold(maximum_warp_update_threshold),
				maximum_iteration_count(maximum_iteration_count),
				tikhonov_term_enabled(tikhonov_term_enabled && tikhonov_strength > 0.0f),
				gradient_kernel_enabled(gradient_kernel_enabled && kernel.size > 0)
		{

}

HierarchicalOptimizer2d::~HierarchicalOptimizer2d()
{
	// TODO Auto-generated destructor stub
}

HierarchicalOptimizer2d::VerbosityParameters::VerbosityParameters(
		bool print_max_warp_update=false,
		bool print_iteration_data_energy=false,
		bool print_iteration_tikhonov_energy=false):
	print_max_warp_update(print_max_warp_update),
	print_iteration_data_energy(print_iteration_data_energy),
	print_iteration_tikhonov_energy(print_iteration_tikhonov_energy)
{

}

} /* namespace nonrigid_optimization */
