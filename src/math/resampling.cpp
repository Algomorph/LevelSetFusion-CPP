/*
 * resampling.cpp
 *
 *  Created on: Mar 28, 2019
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
#include "resampling.tpp"
#include "typedefs.hpp"

namespace math{

template eig::MatrixXf upsampleX2<float>(const eig::MatrixXf& matrix);
template math::MatrixXv2f upsampleX2<math::Vector2f>(const math::MatrixXv2f& matrix);

}// namespace math

