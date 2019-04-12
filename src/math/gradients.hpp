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
#include "stacking.hpp"

namespace math {

void laplacian(math::MatrixXv2f& laplacian, const math::MatrixXv2f& field);
void laplacian(math::Tensor3v3f& laplacian, const math::Tensor3v3f& field);
void negative_laplacian(math::MatrixXv2f& laplacian, const math::MatrixXv2f& field);
void gradient(math::MatrixXm2f& gradient, const math::MatrixXv2f& field);
void gradient(eig::MatrixXf& gradient_x, eig::MatrixXf& gradient_y, const eig::MatrixXf& field);
void gradient(math::MatrixXv2f& gradient, const eig::MatrixXf& field);
void gradient(math::Tensor3v3f& gradient, const math::Tensor3f field);
void gradient2(math::Tensor3v3f& gradient, const math::Tensor3f field);

}

