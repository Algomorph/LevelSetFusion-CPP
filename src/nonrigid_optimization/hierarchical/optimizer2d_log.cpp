/*
 * optimizer2d_log.cpp
 *
 *  Created on: Mar 18, 2019
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

#include "optimizer2d_log.hpp"
#include "../../logging/convergence_report.hpp"
namespace nonrigid_optimization {
namespace hierarchical{
template<>
struct Optimizer2d_log<true>{

};

template<>
struct Optimizer2d_log<false>{
	std::vector<logging::ConvergenceReport> per_level_convergence_reports;
};

} /* namespace hierarchical */
} /* namespace nonrigid_optimization */


