/*
 * test_data_tsdf.hpp
 *
 *  Created on: Jan 30, 2019
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

#include <Eigen/Eigen>
#include "../src/math/tensors.hpp"

namespace eig = Eigen;

namespace test_data {
static eig::Matrix<unsigned short, eig::Dynamic, eig::Dynamic> depth_image_region = [] {
	eig::Matrix<unsigned short, eig::Dynamic, eig::Dynamic> depth_image_region(3,18);
	depth_image_region <<
	3233, 3246, 3243, 3256, 3253, 3268, 3263, 3279, 3272, 3289, 3282, 3299, 3291, 3308, 3301, 3317, 3310, 3326,
	3233, 3246, 3243, 3256, 3253, 3268, 3263, 3279, 3272, 3289, 3282, 3299, 3291, 3308, 3301, 3317, 3310, 3326,
	3233, 3246, 3243, 3256, 3253, 3268, 3263, 3279, 3272, 3289, 3282, 3299, 3291, 3308, 3301, 3317, 3310, 3326;
	return depth_image_region;
}();

static eig::MatrixXf out_sdf_field =
		[] {
			eig::MatrixXf out_sdf_field(16,16);
			out_sdf_field <<
			0.874189f, 0.9932184f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
			0.7656811f, 0.8814954f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
			0.6504901f, 0.7699194f, 0.89273447f, 0.994546f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
			0.5409636f, 0.65847576f, 0.7820716f, 0.88598526f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
			0.43140593f, 0.5471409f, 0.66704684f, 0.7770867f, 0.8991682f, 0.9970366f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
			0.32196245f, 0.4186114f, 0.55751824f, 0.653232f, 0.78992105f, 0.8830927f, 0.99349546f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
			0.20707877f, 0.30999166f, 0.43462086f, 0.54464984f, 0.6617056f, 0.77344257f, 0.88291764f, 0.9927177f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
			0.09879705f, 0.18642762f, 0.3247289f, 0.42850664f, 0.5502955f, 0.6636105f, 0.77237403f, 0.88033676f, 0.97516453f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
			-0.00902274f, 0.07867973f, 0.2152125f, 0.31820253f, 0.43899724f, 0.55378044f, 0.6443364f, 0.7718065f, 0.8531987f, 0.9859558f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
			-0.13057046f, -0.02952692f, 0.08697064f, 0.2076269f, 0.32776707f, 0.44127387f, 0.5363708f, 0.65067255f, 0.744596f, 0.85670555f, 0.9630711f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
			-0.23877731f, -0.14415736f, -0.02455216f, 0.09695236f, 0.21658102f, 0.33207837f, 0.41394132f, 0.5418293f, 0.6321864f, 0.7453086f, 0.8534187f, 0.9555119f, 1.0f, 1.0f, 1.0f, 1.0f,
			-0.34670642f, -0.25351036f, -0.135938f, -0.01366691f, 0.08872667f, 0.22346376f, 0.30653092f, 0.43353093f, 0.52243364f, 0.6339851f, 0.7438599f, 0.8284237f, 0.95121753f, 1.0f, 1.0f, 1.0f,
			-0.47345358f, -0.36296278f, -0.24720258f, -0.12887469f, -0.01996832f, 0.10148326f, 0.19841304f, 0.30502844f, 0.4125015f, 0.52266896f, 0.63263124f, 0.72018117f, 0.8302009f, 0.9177792f, 1.0f, 1.0f,
			-0.5835536f, -0.47238582f, -0.37705892f, -0.23851247f, -0.1440167f, -0.00778072f, 0.08470707f, 0.19444697f, 0.30259356f, 0.39240432f, 0.52388203f, 0.59920317f, 0.7218766f, 0.8063176f, 0.91247654f, 1.0f,
			-0.6936296f, -0.5816594f, -0.48549518f, -0.34763974f, -0.25230578f, -0.1352762f, -0.02480315f, 0.08396093f, 0.19290383f, 0.28379405f, 0.4030714f, 0.49141586f, 0.59485704f, 0.69655037f, 0.8012409f, 0.906666f,
			-0.80368346f, -0.6961099f, -0.5941181f, -0.47093272f, -0.36117333f, -0.24660648f, -0.13451731f, -0.02647889f, 0.08059675f, 0.17470567f, 0.29374388f, 0.38287818f, 0.48430267f, 0.58674127f, 0.6899247f, 0.79556435f;
			return out_sdf_field;
		}();

} //namespace test_data
