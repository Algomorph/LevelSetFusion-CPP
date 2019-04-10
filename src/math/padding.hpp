/*
 * padding.hpp
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

#pragma once

//libraries
#include <Eigen/Dense>
#include <unsupported/Eigen/CXX11/Tensor>

namespace math{

/**
 * Pad tensor by repeating its border values.
 * Same border behavior as mode='edge' for NumPy or BORDER_REPLICATE for OpenCV
 * NB: For simple padding with zeros, use the Tensor::pad method provided by Eigen.
 * @param tensor input tensor
 * @return padded tensor
 */
template<typename Scalar>
Eigen::Tensor<Scalar,3,Eigen::ColMajor> pad_replicate(const Eigen::Tensor<Scalar,3,Eigen::ColMajor>& tensor, int border_width=1);

//TODO: provide Eigen::Matrix overload (if there is no Eigen equivalent)


} //end namespace math




