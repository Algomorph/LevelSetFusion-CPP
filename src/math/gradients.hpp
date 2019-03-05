//  ================================================================
//  Created by Gregory Kramida on 10/26/18.
//  Copyright (c) 2018 Gregory Kramida
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
#pragma once

#include "typedefs.hpp"
#include "tensor_operations.hpp"

namespace math {

//TODO: refactor function arguments -- outputs should come before inputs
void vector_field_laplacian_2d(const math::MatrixXv2f& field, math::MatrixXv2f& laplacian);
void vector_field_laplacian_3d(math::Tensor3v3f& laplacian, const math::Tensor3v3f& field);
void vector_field_negative_laplacian(const math::MatrixXv2f& field, math::MatrixXv2f& laplacian);
void vector_field_gradient(const math::MatrixXv2f& field, math::MatrixXm2f& gradient);
void scalar_field_gradient(const eig::MatrixXf& field, eig::MatrixXf& gradient_x, eig::MatrixXf& gradient_y);
void scalar_field_gradient(const eig::MatrixXf& field, math::MatrixXv2f& gradient);
// ^ end task above (the rest are fine already)
void field_gradient(math::Tensor3v3f& gradient, const eig::Tensor3f field);
void field_gradient2(math::Tensor3v3f& gradient, const eig::Tensor3f field);

}

