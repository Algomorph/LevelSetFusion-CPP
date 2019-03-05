/*
 * tsdf.cpp
 *
 *  Created on: Feb 1, 2019
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

#include "tsdf.hpp"

//libraries
#include <boost/python.hpp>

//local
#include "../tsdf/ewa.hpp"

namespace bp = boost::python;

namespace python_export {

void export_ewa() {

	bp::def("generate_3d_tsdf_field_from_depth_image_ewa", &tsdf::generate_3d_TSDF_field_from_depth_image_EWA,
			"Generate a 3D TSDF field from the provided depth image using Elliptical Weighed Average resampling approach. "
					"A 3D Gaussian (standard deviation of 1 voxel) around every voxel is projected onto the depth image, the resulting "
					"projection is convolved with a 2D Gaussian (standard deviation of 1 pixel), the resulting gaussian is used "
					"as a weighted-average filter to sample from the depth image."
					"For details on EWA methods, refer to [1] and [2].\n"
					"[1] P. S. Heckbert, “Fundamentals of Texture Mapping and Image Warping,” Jun. 1989."
					"[2] M. Zwicker, H. Pfister, J. Van Baar, and M. Gross, “EWA volume splatting,” in Visualization, 2001."
					"    VIS’01. Proceedings, 2001, pp. 29–538.",
			bp::args("depth_image",
					"depth_unit_ratio",
					"camera_intrinsic_matrix",
					"camera_pose",
					"array_offset",
					"field_shape",
					"voxel_size",
					"narrow_band_width_voxels",
					"gaussian_covariance_scale"));

	bp::def("generate_2d_tsdf_field_from_depth_image_ewa", &tsdf::generate_2d_TSDF_field_from_depth_image_EWA,
			"Generate 2D TSDF field from depth image using Elliptical Weighed Average resampling approach. "
					"A 3D Gaussian (standard deviation of 1 voxel) around every voxel is projected onto the depth image, the resulting "
					"projection is convolved with a 2D Gaussian (standard deviation of 1 pixel), the resulting gaussian is used "
					"as a weighted-average filter to sample from the depth image."
					"For details on EWA methods, refer to [1] and [2].\n"
					"[1] P. S. Heckbert, “Fundamentals of Texture Mapping and Image Warping,” Jun. 1989."
					"[2] M. Zwicker, H. Pfister, J. Van Baar, and M. Gross, “EWA volume splatting,” in Visualization, 2001."
					"    VIS’01. Proceedings, 2001, pp. 29–538.",
			bp::args("image_y_coordinate",
					"depth_image",
					"depth_unit_ratio",
					"camera_intrinsic_matrix",
					"camera_pose",
					"array_offset",
					"field_size",
					"voxel_size",
					"narrow_band_width_voxels",
					"gaussian_covariance_scale"));

	bp::def("generate_3d_tsdf_field_from_depth_image_ewa_viz", &tsdf::generate_3d_TSDF_field_from_depth_image_EWA_viz,
			"Draw a visualization of voxel sampling over image space using Elliptical Weighed Average resampling approach."
			"To limit the density, only ellipses corresponding to voxels with abs(TSDF value) below the given truncation "
			"threshold will be drawn."
			"See generate_3d_TSDF_field_from_depth_image_EWA() for method description. "
			"Draws the projected ellipses corresponding to 1 standard deviation away from each voxel sampling center.",
			bp::args("depth_image",
					"depth_unit_ratio",
					"field",
					"camera_intrinsic_matrix",
					"camera_pose",
					"array_offset",
					"voxel_size",
					"scale",
					"tsdf_threshold",
					"gaussian_covariance_scale"));
}

} //namespace python_export
