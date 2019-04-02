/*
 * warping.tpp
 *
 *  Created on: Mar 4, 2019
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

#include "field_warping.hpp"
#include "../traversal/index_raveling.hpp"

namespace nonrigid_optimization{

template<typename ElementType>
inline ElementType sample_tsdf_value(const eig::Tensor<ElementType,3>& tsdf_field, int x, int y, int z) {
	if (x < 0 || x >= tsdf_field.dimension(0) ||
			y < 0 || y >= tsdf_field.dimension(1) ||
			z < 0 || z >= tsdf_field.dimension(2)) {
		return ElementType(1.0f);
	}
	return tsdf_field(x, y, z);
}

template<typename ElementType>
inline ElementType sample_tsdf_value_with_replacement(const eig::Tensor<ElementType,3>& tsdf_field, int x, int y, int z,
		ElementType replacement_value) {
	if (x < 0 || x >= tsdf_field.dimension(0) ||
			y < 0 || y >= tsdf_field.dimension(1) ||
			z < 0 || z >= tsdf_field.dimension(2)) {
		return replacement_value;
	}
	return tsdf_field(x, y, z);
}

template<typename ElementType, bool WithReplacement>
static inline eig::Tensor<ElementType, 3> warp_3d_auxilary(
		const eig::Tensor<ElementType, 3>& scalar_field,
		math::Tensor3v3f& warp_field,
		ElementType replacement_value = ElementType(0.0f)) {
	int matrix_size = static_cast<int>(scalar_field.size());

	eig::Tensor<ElementType, 3> warped_field(scalar_field.dimensions());

	int y_stride = scalar_field.dimension(0);
	int z_stride = y_stride * scalar_field.dimension(1);

#pragma omp parallel for
	for (int i_element = 0; i_element < matrix_size; i_element++) {
		int x, y, z;
		traversal::unravel_3d_index(x, y, z, i_element, y_stride, z_stride);

		math::Vector3f local_warp = warp_field(i_element);
		float lookup_x = x + local_warp.u;
		float lookup_y = y + local_warp.v;
		float lookup_z = z + local_warp.w;
		int base_x = static_cast<int>(std::floor(lookup_x));
		int base_y = static_cast<int>(std::floor(lookup_y));
		int base_z = static_cast<int>(std::floor(lookup_z));
		float ratio_x = lookup_x - base_x;
		float ratio_y = lookup_y - base_y;
		float ratio_z = lookup_z - base_z;
		float inverse_ratio_x = 1.0f - ratio_x;
		float inverse_ratio_y = 1.0f - ratio_y;
		float inverse_ratio_z = 1.0f - ratio_z;

		ElementType value000, value010, value100, value110, value001, value011, value101, value111;

		if (WithReplacement) {
			value000 = sample_tsdf_value_with_replacement(scalar_field, base_x, base_y, base_z,
					replacement_value);
			value010 = sample_tsdf_value_with_replacement(scalar_field, base_x, base_y + 1, base_z,
					replacement_value);
			value100 = sample_tsdf_value_with_replacement(scalar_field, base_x + 1, base_y, base_z,
					replacement_value);
			value110 = sample_tsdf_value_with_replacement(scalar_field, base_x + 1, base_y + 1, base_z,
					replacement_value);
			value001 = sample_tsdf_value_with_replacement(scalar_field, base_x, base_y, base_z + 1,
					replacement_value);
			value011 = sample_tsdf_value_with_replacement(scalar_field, base_x, base_y + 1, base_z + 1,
					replacement_value);
			value101 = sample_tsdf_value_with_replacement(scalar_field, base_x + 1, base_y, base_z + 1,
					replacement_value);
			value111 = sample_tsdf_value_with_replacement(scalar_field, base_x + 1, base_y + 1, base_z + 1,
					replacement_value);
		} else {
			value000 = sample_tsdf_value(scalar_field, base_x, base_y, base_z);
			value010 = sample_tsdf_value(scalar_field, base_x, base_y + 1, base_z);
			value100 = sample_tsdf_value(scalar_field, base_x + 1, base_y, base_z);
			value110 = sample_tsdf_value(scalar_field, base_x + 1, base_y + 1, base_z);
			value001 = sample_tsdf_value(scalar_field, base_x, base_y, base_z + 1);
			value011 = sample_tsdf_value(scalar_field, base_x, base_y + 1, base_z + 1);
			value101 = sample_tsdf_value(scalar_field, base_x + 1, base_y, base_z + 1);
			value111 = sample_tsdf_value(scalar_field, base_x + 1, base_y + 1, base_z + 1);
		}
		ElementType interpolated_value00 = value000 * inverse_ratio_z + value001 * ratio_z;
		ElementType interpolated_value01 = value010 * inverse_ratio_z + value011 * ratio_z;
		ElementType interpolated_value10 = value100 * inverse_ratio_z + value101 * ratio_z;
		ElementType interpolated_value11 = value110 * inverse_ratio_z + value111 * ratio_z;

		ElementType interpolated_value0 = interpolated_value00 * inverse_ratio_y + interpolated_value01 * ratio_y;
		ElementType interpolated_value1 = interpolated_value10 * inverse_ratio_y + interpolated_value11 * ratio_y;

		ElementType new_value = interpolated_value0 * inverse_ratio_x + interpolated_value1 * ratio_x;

		warped_field(i_element) = new_value;
	}

	return warped_field;
}

template<typename ElementType>
eig::Tensor<ElementType,3> warp_3d(const eig::Tensor<ElementType,3>& scalar_field, math::Tensor3v3f& warp_field) {
	return warp_3d_auxilary<ElementType,false>(scalar_field,warp_field);
}

template<typename ElementType>
eig::Tensor<ElementType,3> warp_3d_with_replacement(const eig::Tensor<ElementType,3>& scalar_field, math::Tensor3v3f& warp_field,
		ElementType replacement_value){
	return warp_3d_auxilary<ElementType,true>(scalar_field,warp_field, replacement_value);
}

}//namespace nonrigid_optimization




