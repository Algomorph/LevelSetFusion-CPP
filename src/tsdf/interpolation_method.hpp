/*
 * interpolation_method.hpp
 *
 *  Created on: Apr 23, 2019
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

namespace tsdf {

/**
 * @brief Different interpolation methods used to generate the TSDF (might change to "FilteringMethod" in the future)
 *
 * @details EWA methods generate a the TSDF field using Elliptical Weighed Average resampling approach on the depth values.
 * A 3D Gaussian (standard deviation of 1 voxel) around every voxel is projected onto the depth image, the resulting
 * projection is convolved with a 2D Gaussian (standard deviation of 1 pixel), the resulting gaussian is used
 * as a weighted-average filter to sample in different ways, as described further.
 *
 * For details on EWA methods, refer to [1] and [2].
 * [1] P. S. Heckbert, “Fundamentals of Texture Mapping and Image Warping,” Jun. 1989.
 * [2] M. Zwicker, H. Pfister, J. Van Baar, and M. Gross, “EWA volume splatting,” in Visualization, 2001.
 *     VIS’01. Proceedings, 2001, pp. 29–538.
 **/
enum class FilteringMethod {
	NONE = 0,                    //!< NONE no interpolation method
	BILINEAR_IMAGE_SPACE = 1, //!< BILINEAR_IMAGE_SPACE (not yet supported)
	BILINEAR_VOXEL_SPACE = 2, //!< BILINEAR_VOXEL_SPACE (not yet supported)
	EWA_IMAGE_SPACE = 3, //!< EWA_IMAGE_SPACE the samples from the weighted-average filtered depth-image values compose a "filtered" depth value, which is then truncated
	EWA_VOXEL_SPACE = 4, //!< EWA_VOXEL_SPACE the samples from the weighted-average filtered depth-image values are first truncated, then combined into a final TSDF value. Samples are gathered only for voxels that can be projected directly onto the image.
	EWA_VOXEL_SPACE_INCLUSIVE = 5 //!< EWA_VOXEL_SPACE_INCLUSIVE the samples from the weighted-average filtered depth-image values are first truncated, then combined into a final TSDF value. Samples are gathered for any voxel which intersects with the influence are of a potential filter.
};
}  // namespace tsdf

