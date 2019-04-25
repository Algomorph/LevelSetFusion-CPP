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


#include "hierarchical_optimizer.tpp"

namespace python_export {
namespace hierarchical_optimizer {

template void export_algorithms<eig::MatrixXf, math::MatrixXv2f>(const char* suffix);
template void export_algorithms<math::Tensor3f, math::Tensor3v3f>(const char* suffix);

} // namespace hierarchical_optimizer
} // namespace python_export
