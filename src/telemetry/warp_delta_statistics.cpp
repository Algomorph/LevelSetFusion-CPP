/*
 * warp_delta_statistics.cpp
 *
 *  Created on: Apr 15, 2019
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

#include "warp_delta_statistics.tpp"

namespace telemetry {

template struct WarpDeltaStatistics<math::Vector2i>;
template struct WarpDeltaStatistics<math::Vector3i>;

template WarpDeltaStatistics<math::Vector2i> build_warp_delta_statistics<math::Vector2i, eig::MatrixXf, math::MatrixXv2f>(
		const math::MatrixXv2f& warp_field,
		const eig::MatrixXf& canonical_field,
		const eig::MatrixXf& live_field,
		float min_threshold, float max_threshold);

template WarpDeltaStatistics<math::Vector3i> build_warp_delta_statistics<math::Vector3i, math::Tensor3f, math::Tensor3v3f>(
		const math::Tensor3v3f& warp_field,
		const math::Tensor3f& canonical_field,
		const math::Tensor3f& live_field,
		float min_threshold, float max_threshold);

template std::ostream &operator<<(std::ostream &ostr, const WarpDeltaStatistics<math::Vector2i> &ts);
template std::ostream &operator<<(std::ostream &ostr, const WarpDeltaStatistics<math::Vector3i> &ts);

} //namespace telemetry


