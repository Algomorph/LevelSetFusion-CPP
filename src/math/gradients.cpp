//  ================================================================
//  Created by Gregory Kramida on 04/14/19.
//  Copyright (c) 2019 Gregory Kramida
//  Licensed under the Apache License, Version 2.0 (the "License");
//  you may not use this file except in compliance with the License.
//  You may obtain a copy of the License at

//  http://www.apache.org/licenses/LICENSE-2.0

//  Unless required by applicable law or agreed to in writing, software
//  distributed under the License is distributed on an "AS IS" BASIS,
//  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
//  See the License for the specific language governing permissions and
//  limitations under the License.
//  ================================================================

//local
#include "gradients.tpp"

namespace math{

template void laplacian<float>(math::MatrixXv2f& laplacian, const math::MatrixXv2f& field);
template void negative_laplacian<float>(math::MatrixXv2f& laplacian, const math::MatrixXv2f& field);
template void laplacian<float>(math::Tensor3v3f& laplacian, const math::Tensor3v3f& field);

template void gradient<float>(math::MatrixXv2f& gradient, const eig::MatrixXf& field);
template void gradient<float>(eig::MatrixXf& gradient_x, eig::MatrixXf& gradient_y, const eig::MatrixXf& field);
template void gradient<float>(math::MatrixXm2f& gradient, const math::MatrixXv2f& field);
template void gradient<float>(math::Tensor3v3f& gradient, const math::Tensor3f& field);
template void gradient2<float>(math::Tensor3v3f& gradient, const math::Tensor3f& field);
template void gradient<float>(math::Tensor3m3f& gradient, const math::Tensor3v3f& field);

}  // namespace math
