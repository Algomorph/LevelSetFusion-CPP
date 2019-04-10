/*
 * padding.cpp
 *
 *  Created on: Apr 8, 2019
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
#include "padding.tpp"
#include "typedefs.hpp"

namespace math{

template math::Tensor3f pad_replicate<float>(const math::Tensor3f& tensor, int border_width);
template math::Tensor3v3f pad_replicate<math::Vector3f>(const math::Tensor3v3f& tensor, int border_width);

} //end namespace math
