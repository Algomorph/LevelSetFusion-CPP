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

// *** upsampling ***
template eig::MatrixXf upsampleX2<float>(const eig::MatrixXf& field, UpsamplingStrategy upsampling_strategy);
template math::MatrixXv2f upsampleX2<math::Vector2f>(const math::MatrixXv2f& field, UpsamplingStrategy upsampling_strategy);
template math::Tensor3f upsampleX2<float>(const math::Tensor3f& field, UpsamplingStrategy upsampling_strategy);
template math::Tensor3v3f upsampleX2<math::Vector3f>(const math::Tensor3v3f& field, UpsamplingStrategy upsampling_strategy);

template eig::MatrixXf upsampleX2_nearest<float>(const eig::MatrixXf& field);
template math::MatrixXv2f upsampleX2_nearest<math::Vector2f>(const math::MatrixXv2f& field);
template math::Tensor3f upsampleX2_nearest<float>(const math::Tensor3f& field);
template math::Tensor3v3f upsampleX2_nearest<math::Vector3f>(const math::Tensor3v3f& field);

template eig::MatrixXf upsampleX2_linear<float>(const eig::MatrixXf& field);
template math::MatrixXv2f upsampleX2_linear<math::Vector2f>(const math::MatrixXv2f& field);
template math::Tensor3f upsampleX2_linear<float>(const math::Tensor3f& field);
template math::Tensor3v3f upsampleX2_linear<math::Vector3f>(const math::Tensor3v3f& field);

//*** downsampling ***
template eig::MatrixXf downsampleX2<float>(const eig::MatrixXf& field, DownsamplingStrategy downsampling_strategy);
template math::MatrixXv2f downsampleX2<math::Vector2f>(const math::MatrixXv2f& field, DownsamplingStrategy downsampling_strategy);
template math::Tensor3f downsampleX2<float>(const math::Tensor3f& field, DownsamplingStrategy downsampling_strategy);
template math::Tensor3v3f downsampleX2<math::Vector3f>(const math::Tensor3v3f& field, DownsamplingStrategy downsampling_strategy);

template eig::MatrixXf downsampleX2_average<float>(const eig::MatrixXf& field);
template math::MatrixXv2f downsampleX2_average<math::Vector2f>(const math::MatrixXv2f& field);
template math::Tensor3f downsampleX2_average<float>(const math::Tensor3f& field);
template math::Tensor3v3f downsampleX2_average<math::Vector3f>(const math::Tensor3v3f& field);

template eig::MatrixXf downsampleX2_linear<float>(const eig::MatrixXf& field);
template math::MatrixXv2f downsampleX2_linear<math::Vector2f>(const math::MatrixXv2f& field);
template math::Tensor3f downsampleX2_linear<float>(const math::Tensor3f& field);
template math::Tensor3v3f downsampleX2_linear<math::Vector3f>(const math::Tensor3v3f& field);

}// namespace math

