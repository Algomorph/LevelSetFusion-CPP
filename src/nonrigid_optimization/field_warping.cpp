//  ================================================================
//  Created by Gregory Kramida on 10/11/18.
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
//stdlib
#include <cstdlib>
#include <iostream>

//local
#include "field_warping.tpp"
#include "../math/boolean_operations.hpp"
#include "../math/typedefs.hpp"
#include "../math/stacking.hpp"
#include "../traversal/index_raveling.hpp"

namespace eig = Eigen;

namespace nonrigid_optimization {


template
math::Tensor3f warp<float,float>(const math::Tensor3f& scalar_field, const math::Tensor3v3f& warp_field);

template
math::Tensor3f warp_with_replacement<float, float>( // @suppress("Ambiguous problem")
		const math::Tensor3f& scalar_field,
		const math::Tensor3v3f& warp_field,
		float replacement_value);

template
math::Tensor3v3f warp_with_replacement<math::Vector3f, float>(
		const math::Tensor3v3f& vector_field,
		const math::Tensor3v3f& warp_field,
		math::Vector3f replacement_value);

template
eig::MatrixXf warp<float,float>(const eig::MatrixXf& scalar_field, const math::MatrixXv2f& warp_field);

template
eig::MatrixXf warp_with_replacement<float,float>(const eig::MatrixXf& scalar_field, // @suppress("Ambiguous problem")
		const math::MatrixXv2f& warp_field,
		float replacement_value);

template
math::MatrixXv2f warp_with_replacement<math::Vector2f,float>(const math::MatrixXv2f& scalar_field,
		const math::MatrixXv2f& warp_field,
		math::Vector2f replacement_value);

//================= legacy (Sobolev/KillingFusion only) ========================



template<bool TModifyWarpField>
inline eig::MatrixXf
warp_2d_advanced_aux(math::MatrixXv2f& warp_field,
		const eig::MatrixXf& warped_live_field, const eig::MatrixXf& canonical_field,
		bool band_union_only, bool known_values_only,
		bool substitute_original, float truncation_float_threshold) {
	int matrix_size = static_cast<int>(warped_live_field.size());
	const int column_count = static_cast<int>(warped_live_field.cols());
	const int row_count = static_cast<int>(warped_live_field.rows());

	eig::MatrixXf new_live_field(row_count, column_count);

#pragma omp parallel for
	for (int i_element = 0; i_element < matrix_size; i_element++) {
		// Any MatrixXf in Eigen is column-major
		// i_element = x * column_count + y
		div_t division_result = div(i_element, column_count);
		int x = division_result.quot;
		int y = division_result.rem;

		float live_tsdf_value = warped_live_field(i_element);
		if (band_union_only) {
			float canonical_tsdf_value = canonical_field(i_element);
			if (math::are_both_SDF_values_truncated(live_tsdf_value, canonical_tsdf_value)) {
				new_live_field(i_element) = live_tsdf_value;
				continue;
			}
		}
		if (known_values_only) {
			if (std::abs(live_tsdf_value) == 1.0) {
				new_live_field(i_element) = live_tsdf_value;
				continue;
			}
		}
		math::Vector2f local_warp = warp_field(i_element);
		float lookup_x = x + local_warp.u;
		float lookup_y = y + local_warp.v;
		int base_x = static_cast<int>(std::floor(lookup_x));
		int base_y = static_cast<int>(std::floor(lookup_y));
		float ratio_x = lookup_x - base_x;
		float ratio_y = lookup_y - base_y;
		float inverse_ratio_x = 1.0F - ratio_x;
		float inverse_ratio_y = 1.0F - ratio_y;

		float value00, value01, value10, value11;
		if (substitute_original) {
			value00 = sample_tsdf_value_with_replacement(warped_live_field, base_x, base_y,
					live_tsdf_value);
			value01 = sample_tsdf_value_with_replacement(warped_live_field, base_x, base_y + 1,
					live_tsdf_value);
			value10 = sample_tsdf_value_with_replacement(warped_live_field, base_x + 1, base_y,
					live_tsdf_value);
			value11 = sample_tsdf_value_with_replacement(warped_live_field, base_x + 1, base_y + 1,
					live_tsdf_value);
		} else {
			value00 = sample_tsdf_value(warped_live_field, base_x, base_y);
			value01 = sample_tsdf_value(warped_live_field, base_x, base_y + 1);
			value10 = sample_tsdf_value(warped_live_field, base_x + 1, base_y);
			value11 = sample_tsdf_value(warped_live_field, base_x + 1, base_y + 1);
		}

		float interpolated_value0 = value00 * inverse_ratio_y + value01 * ratio_y;
		float interpolated_value1 = value10 * inverse_ratio_y + value11 * ratio_y;
		float new_value = interpolated_value0 * inverse_ratio_x + interpolated_value1 * ratio_x;
		if (TModifyWarpField && (1.0 - std::abs(new_value) < truncation_float_threshold)) {
			new_value = std::copysign(1.0f, new_value);
			warp_field(i_element) = math::Vector2f(0.0f);
		}
		new_live_field(i_element) = new_value;
	}

	return new_live_field;
}

eig::MatrixXf warp_2d_advanced(math::MatrixXv2f& warp_field,
		const eig::MatrixXf& warped_live_field, const eig::MatrixXf& canonical_field,
		bool band_union_only, bool known_values_only,
		bool substitute_original, float truncation_float_threshold) {
	return warp_2d_advanced_aux<true>(warp_field, warped_live_field, canonical_field, band_union_only, known_values_only,
			substitute_original, truncation_float_threshold);
}

eig::MatrixXf warp_2d_advanced_warp_unchanged(
		math::MatrixXv2f& warp_field,
		const eig::MatrixXf& warped_live_field,
		const eig::MatrixXf& canonical_field,
		bool band_union_only, bool known_values_only,
		bool substitute_original, float truncation_float_threshold) {
	return warp_2d_advanced_aux<false>(warp_field, warped_live_field, canonical_field, band_union_only, known_values_only,
			substitute_original, truncation_float_threshold);
}

bp::object py_warp_field(const eig::MatrixXf& warped_live_field,
		const eig::MatrixXf& canonical_field, eig::MatrixXf warp_field_u,
		eig::MatrixXf warp_field_v, bool band_union_only, bool known_values_only,
		bool substitute_original, float truncation_float_threshold)
		{
	math::MatrixXv2f warp_field = math::stack_as_xv2f(warp_field_u, warp_field_v);
	eig::MatrixXf new_warped_live_field = warp_2d_advanced(warp_field, warped_live_field,
			canonical_field, band_union_only, known_values_only, substitute_original,
			truncation_float_threshold);
	math::unstack_xv2f(warp_field_u, warp_field_v, warp_field);
	bp::object warp_field_u_out(warp_field_u);
	bp::object warp_field_v_out(warp_field_v);
	bp::object warped_live_field_out(new_warped_live_field);
	return bp::make_tuple(warped_live_field_out, bp::make_tuple(warp_field_u_out, warp_field_v_out));
}

bp::object py_warp_field_no_warp_change(const eig::MatrixXf& warped_live_field,
		const eig::MatrixXf& canonical_field, eig::MatrixXf warp_field_u,
		eig::MatrixXf warp_field_v, bool band_union_only, bool known_values_only,
		bool substitute_original, float truncation_float_threshold)
		{
	math::MatrixXv2f warp_field = math::stack_as_xv2f(warp_field_u, warp_field_v);
	eig::MatrixXf new_warped_live_field = warp_2d_advanced_warp_unchanged(warp_field, warped_live_field,
			canonical_field, band_union_only, known_values_only, substitute_original,
			truncation_float_threshold);
	bp::object warped_live_field_out(new_warped_live_field);
	return warped_live_field_out;
}

}		//namespace nonrigid_optimization
