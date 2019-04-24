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
#include <Eigen/Dense>

//local
#include "../math/typedefs.hpp"
#include "../tsdf/ewa_viz.hpp"
#include "../tsdf/generator.hpp"
#include "../tsdf/ewa_common.hpp"
#include "../tsdf/interpolation_method.hpp"
#include "../tsdf/parameters.hpp"

namespace bp = boost::python;

namespace python_export {
namespace pe_tsdf {
struct TSDF_scope_dummy{

};

void export_algorithms() {
	bp::scope tsdf_scope = bp::class_<TSDF_scope_dummy>("tsdf", bp::no_init);
	bp::enum_<tsdf::FilteringMethod>("FilteringMethod")
			.value("NONE", tsdf::FilteringMethod::NONE)
			.value("BILINEAR_IMAGE_SPACE", tsdf::FilteringMethod::BILINEAR_IMAGE_SPACE)
			.value("BILINEAR_VOXEL_SPACE", tsdf::FilteringMethod::BILINEAR_VOXEL_SPACE)
			.value("EWA_IMAGE_SPACE", tsdf::FilteringMethod::EWA_IMAGE_SPACE)
			.value("EWA_VOXEL_SPACE", tsdf::FilteringMethod::EWA_VOXEL_SPACE)
			.value("EWA_VOXEL_SPACE_INCLUSIVE", tsdf::FilteringMethod::EWA_VOXEL_SPACE_INCLUSIVE);

	bp::class_<tsdf::Parameters2d>("Parameters2d",
			bp::init<bp::optional<float,Eigen::Matrix3f,float,math::Vector2i, math::Vector2i, float, int,
				tsdf::FilteringMethod,float>>(bp::args("depth_unit_ratio","projection_matrix",
						"near_clipping_distance", "array_offset", "field_shape","voxel_size","narrow_band_width_voxels",
						"interpolation_method","smoothing_factor")))
			.def_readwrite("depth_unit_ratio", &tsdf::Parameters2d::depth_unit_ratio)
			.def_readwrite("projection_matrix", &tsdf::Parameters2d::projection_matrix)
			.def_readwrite("near_clipping_distance", &tsdf::Parameters2d::near_clipping_distance)
			.def_readwrite("array_offset", &tsdf::Parameters2d::array_offset)
			.def_readwrite("field_shape", &tsdf::Parameters2d::field_shape)
			.def_readwrite("voxel_size", &tsdf::Parameters2d::voxel_size)
			.def_readwrite("narrow_band_width_voxels", &tsdf::Parameters2d::narrow_band_width_voxels)
			.def_readwrite("interpolation_method", &tsdf::Parameters2d::interpolation_method)
			.def_readwrite("smoothing_factor", &tsdf::Parameters2d::smoothing_factor);

	bp::class_<tsdf::Generator2d>("Generator2d",
			bp::init<tsdf::Parameters2d>("parameters"))
		.def("generate", &tsdf::Generator2d::generate,
				"Generate a discrete implicit TSDF (Truncated Signed Distance "
				"Function) from the give depth image presumed to have been taken at the specified camera pose.",
				bp::args("depth_image","camera_pose","image_y_coordinate"));

	bp::def("generate_tsdf_3d_ewa_image_visualization", &tsdf::generate_TSDF_3D_EWA_image_visualization,
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

	//DEBUG
	bp::def("sampling_area_heatmap_2d_ewa_image", &tsdf::sampling_area_heatmap_2D_EWA_image,
			"DEBUGGING PURPOSE ONLY -- compute matrix of 2D EWA sampling areas for each pixel (~2d voxel) in resulting TSDF",
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
}
}  // namespace pe_tsdf
}  // namespace python_export
