/*
 * test_tsdf.cpp
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

#define BOOST_TEST_MODULE test_tsdf

//standard library
#include <climits>

//libraries
#include <boost/test/unit_test.hpp>
#include <boost/python.hpp>
#include <Eigen/Dense>

//test data
#include "data/test_data_tsdf.hpp"

//test targets
#include "../src/tsdf/ewa.hpp"
#include "../src/math/typedefs.hpp"
#include "../src/math/assessment.hpp"
#include "../src/imageio/png_eigen.hpp"

namespace eig = Eigen;

BOOST_AUTO_TEST_CASE(test_EWA_2D_generation01) {

	eig::MatrixXus depth_image = eig::MatrixXus::Constant(3, 640, USHRT_MAX);
	depth_image.block(0, 399, 3, 18) = test_data::depth_image_region;

	eig::Matrix3f camera_intrinsic_matrix;
	camera_intrinsic_matrix <<
			700.0f, 0.0f, 320.0f,
			0.0f, 700.0f, 240.0f,
			0.0f, 0.0f, 1.0f;
	eig::Vector3i offset;
	offset << 94, -256, 804;

	eig::MatrixXf field = tsdf::generate_2d_TSDF_field_from_depth_image_EWA(
			1, // y coord
			depth_image,
			0.001f, //depth unit ratio
			camera_intrinsic_matrix,
			eig::Matrix4f::Identity(), //camera pose
			offset,
			16, //field size
			0.004f, //voxel size
			20 // narrow band width
			);

	BOOST_REQUIRE(math::almost_equal_verbose(field, test_data::out_sdf_field, 3e-5));
}

BOOST_AUTO_TEST_CASE(test_EWA_3D_generation01) {

	eig::Matrix3f camera_intrinsic_matrix;
	camera_intrinsic_matrix <<
			700.0f, 0.0f, 320.0f,
			0.0f, 700.0f, 240.0f,
			0.0f, 0.0f, 1.0f;

	eig::MatrixXus depth_image;

#define PNG_FILENAME "zigzag2_depth_00108.png"
//#define PNG_FILENAME "zigzag_depth_00064.png"
	bool image_read = imageio::png::read_GRAY16("test_data/" PNG_FILENAME, depth_image);
	if(!image_read){
		//are we running from the project root dir, maybe?
		image_read = imageio::png::read_GRAY16("tests/data/" PNG_FILENAME, depth_image);
	}
#undef PNG_FILENAME

	BOOST_REQUIRE(image_read);

	//TODO: this works for PNG_FILENAME "zigzag_depth_00064.png" ONLY, reset that later
//	BOOST_REQUIRE_EQUAL(depth_image.rows(), 480);
//	BOOST_REQUIRE_EQUAL(depth_image.cols(), 640);
//	BOOST_REQUIRE_EQUAL(depth_image(0, 0), (unsigned short )1997);
//	BOOST_REQUIRE_EQUAL(depth_image(479, 0), (unsigned short )1997);
//	BOOST_REQUIRE_EQUAL(depth_image(479, 639), (unsigned short ) 5154);
//eig::MatrixXus sample = depth_image.block(40, 60, 1, 20);
//	BOOST_REQUIRE(sample.isApprox(test_data::depth_00064_sample));
//#define THIN_SLICE
#define CENTER_BOX

#if !defined(THIN_SLICE) && !defined(CENTER_BOX)
	eig::Vector3i offset;
	offset << -256, -256, 480;

	eig::Vector3i field_size;
	field_size << 512, 512, 512;

	eig::Tensor<float, 3> field = tsdf::generate_3d_TSDF_field_from_depth_image_EWA(
			depth_image,
			0.001f, //depth unit ratio
			camera_intrinsic_matrix,
			eig::Matrix4f::Identity(), //camera pose
			offset,
			field_size, //field size
			0.004f, //voxel size
			20 // narrow band width
			);
#elif defined(THIN_SLICE)
	eig::Vector3i offset;
	offset << -224, -256, 480;

	eig::Vector3i field_size;
	field_size << 1, 512, 512;

	eig::Tensor<float, 3> field = tsdf::generate_3d_TSDF_field_from_depth_image_EWA(
			depth_image,
			0.001f, //depth unit ratio
			camera_intrinsic_matrix,
			eig::Matrix4f::Identity(), //camera pose
			offset,
			field_size, //field size
			0.004f, //voxel size
			20 // narrow band width
			);
#elif defined(CENTER_BOX)
	eig::Vector3i offset;
	//offset << -32, -32, 0; //zigzag-64
	offset << -46, -8, 105; //zigzag2-108

	eig::Vector3i field_size;
	//field_size << 64, 64, 64; //zigzag-64
	field_size << 16, 16, 16; //zigzag2-108

	eig::Tensor<float, 3> field = tsdf::generate_3d_TSDF_field_from_depth_image_EWA(
			depth_image,
			0.001f, //depth unit ratio
			camera_intrinsic_matrix,
			eig::Matrix4f::Identity(), //camera pose
			offset,
			field_size, //field size
			0.004f, //voxel size
			20 // narrow band width
			);
#endif

	eig::MatrixXuc image = tsdf::generate_3d_TSDF_field_from_depth_image_EWA_viz(
			depth_image, 0.001f, field,
			camera_intrinsic_matrix, eig::Matrix4f::Identity(),
			offset, 0.004f);

}
