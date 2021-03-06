/*
 * optimizer_with_telemetry.cpp
 *
 *  Created on: Apr 10, 2019
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


//local
#include "optimizer_with_telemetry.tpp"
#include "../../math/typedefs.hpp"

namespace nonrigid_optimization {
namespace hierarchical{

template class OptimizerWithTelemetry<eig::MatrixXf,math::MatrixXv2f>;
template class OptimizerWithTelemetry<math::Tensor3f,math::Tensor3v3f>;

} /* namespace hierarchical */
} /* namespace nonrigid_optimization */


