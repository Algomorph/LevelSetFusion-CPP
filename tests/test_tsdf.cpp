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


static inline bool read_image_helper(eig::MatrixXus& depth_image, std::string filename){
	std::string full_path = "test_data/" + filename;
	bool image_read = imageio::png::read_GRAY16( full_path.c_str(), depth_image);
	if (!image_read) {
		//are we running from the project root dir, maybe?
		std::string full_path = "tests/data/" + filename;
		image_read = imageio::png::read_GRAY16(full_path.c_str(), depth_image);
	}
	return image_read;
}


//TODO: move image testing to it's own test suite
BOOST_AUTO_TEST_CASE(test_image_read01) {
	eig::MatrixXus depth_image;
	bool image_read = read_image_helper(depth_image, "zigzag_depth_00064.png");
	BOOST_REQUIRE(image_read);
	BOOST_REQUIRE_EQUAL(depth_image.rows(), 480);
	BOOST_REQUIRE_EQUAL(depth_image.cols(), 640);
	BOOST_REQUIRE_EQUAL(depth_image(0, 0), (unsigned short )1997);
	BOOST_REQUIRE_EQUAL(depth_image(479, 0), (unsigned short )1997);
	BOOST_REQUIRE_EQUAL(depth_image(479, 639), (unsigned short ) 5154);
	eig::MatrixXus sample = depth_image.block(40, 60, 1, 20);
	BOOST_REQUIRE(sample.isApprox(test_data::depth_00064_sample));

}

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

	BOOST_REQUIRE(math::matrix_almost_equal_verbose(field, test_data::out_sdf_field, 1e-6));
}

BOOST_AUTO_TEST_CASE(test_EWA_2D_generation02) {
	eig::MatrixXus depth_image;
	bool image_read = read_image_helper(depth_image, "zigzag2_depth_00108.png");
	BOOST_REQUIRE(image_read);
	eig::Matrix3f camera_intrinsic_matrix;
	camera_intrinsic_matrix <<
			700.0f, 0.0f, 320.0f,
			0.0f, 700.0f, 240.0f,
			0.0f, 0.0f, 1.0f;
	eig::Vector3i offset;
	offset << -256, -256, 0;

	eig::MatrixXf field = tsdf::generate_2d_TSDF_field_from_depth_image_EWA(
			200, // y coord
			depth_image,
			0.001f, //depth unit ratio
			camera_intrinsic_matrix,
			eig::Matrix4f::Identity(), //camera pose
			offset,
			512, //field size
			0.004f, //voxel size
			20 // narrow band width
			);

	eig::MatrixXf field_chunk = field.block(103,210,16,16);

	BOOST_REQUIRE(math::matrix_almost_equal_verbose(field_chunk, test_data::out_sdf_chunk, 1e-6));
}


BOOST_AUTO_TEST_CASE(test_EWA_3D_generation01) {
	eig::MatrixXus depth_image;
	bool image_read = read_image_helper(depth_image, "zigzag2_depth_00108.png");
	BOOST_REQUIRE(image_read);

	eig::Vector3i offset;
	offset << -46, -8, 105; //zigzag2-108

	eig::Vector3i field_size;
	field_size << 16, 1, 16; //zigzag2-108

	eig::Matrix3f camera_intrinsic_matrix;
	camera_intrinsic_matrix <<
			700.0f, 0.0f, 320.0f,
			0.0f, 700.0f, 240.0f,
			0.0f, 0.0f, 1.0f;

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

	BOOST_REQUIRE(math::tensor_almost_equal_verbose(field, test_data::TSDF_slice01 , 1e-6));
}
