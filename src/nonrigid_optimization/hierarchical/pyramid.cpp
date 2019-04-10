/*
 * pyramid.cpp
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

//libraries
#include <Eigen/Dense>

//local
#include "pyramid.tpp"
#include "../../math/typedefs.hpp"

namespace eig = Eigen;

namespace nonrigid_optimization {
namespace hierarchical{

template class Pyramid<eig::MatrixXf>;
template class Pyramid<math::MatrixXv2f>;
template class Pyramid<math::Tensor3f>;
template class Pyramid<math::Tensor3v3f>;

} //namespace hierarchical
} //namespace nonrigid_optimization
