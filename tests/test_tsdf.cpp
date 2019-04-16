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
#include "../src/tsdf/tsdf.hpp"
#include "../src/tsdf/ewa.hpp"
#include "../src/math/typedefs.hpp"
#include "../src/math/almost_equal.hpp"
#include "../src/image_io/png_eigen.hpp"
#include "common.hpp"

namespace eig = Eigen;

BOOST_AUTO_TEST_CASE(test_EWA_2D_generation01) {

	math::MatrixXus depth_image = math::MatrixXus::Constant(3, 640, USHRT_MAX);
	depth_image.block(0, 399, 3, 18) = test_data::depth_image_region;

	eig::Matrix3f camera_intrinsic_matrix;
	camera_intrinsic_matrix <<
			700.0f, 0.0f, 320.0f,
			0.0f, 700.0f, 240.0f,
			0.0f, 0.0f, 1.0f;
	eig::Vector3i offset;
	offset << 94, -256, 804;

	eig::MatrixXf field = tsdf::generate_TSDF_2D_EWA_image(
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

	BOOST_REQUIRE(math::almost_equal_verbose(field, test_data::out_sdf_field, 1e-6f));
}

BOOST_AUTO_TEST_CASE(test_EWA_2D_generation02) {
	math::MatrixXus depth_image;
	bool image_read = read_image_helper(depth_image, "zigzag2_depth_00108.png");
	BOOST_REQUIRE(image_read);
	bool test_full_image = true;

	eig::Matrix3f camera_intrinsic_matrix;
	camera_intrinsic_matrix <<
			700.0f, 0.0f, 320.0f,
			0.0f, 700.0f, 240.0f,
			0.0f, 0.0f, 1.0f;
	eig::Vector3i offset;
	offset << -256, -256, 0;
	int chunk_x_start = 210, chunk_y_start = 103;
	int chunk_size = 16;
	eig::Vector3i offset_chunk_from_image;
	offset_chunk_from_image << chunk_x_start, 0.0f, chunk_y_start;
	eig::Vector3i offset_chunk = offset + offset_chunk_from_image;

	eig::MatrixXf field_chunk;
	if (test_full_image) {
		eig::MatrixXf field = tsdf::generate_TSDF_2D_EWA_image(
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

		field_chunk = field.block(chunk_y_start, chunk_x_start, chunk_size, chunk_size);
	} else {
		field_chunk = tsdf::generate_TSDF_2D_EWA_image(
				200, // y coord
				depth_image,
				0.001f, //depth unit ratio
				camera_intrinsic_matrix,
				eig::Matrix4f::Identity(), //camera pose
				offset_chunk,
				chunk_size, //field size
				0.004f, //voxel size
				20 // narrow band width
				);
	}

	BOOST_REQUIRE(math::almost_equal_verbose(field_chunk, test_data::out_sdf_chunk, 1e-6f));
}

BOOST_AUTO_TEST_CASE(test_EWA_2D_generation05) {
	math::MatrixXus depth_image;
	bool image_read = read_image_helper(depth_image, "zigzag2_depth_00108.png");
	BOOST_REQUIRE(image_read);
	bool test_full_image = true;

	eig::Matrix3f camera_intrinsic_matrix;
	camera_intrinsic_matrix <<
			700.0f, 0.0f, 320.0f,
			0.0f, 700.0f, 240.0f,
			0.0f, 0.0f, 1.0f;
	eig::Vector3i offset;
	offset << -256, -256, 0;
	int chunk_x_start = 210, chunk_y_start = 103;
	int chunk_size = 16;
	eig::Vector3i offset_chunk_from_image;
	offset_chunk_from_image << chunk_x_start, 0.0f, chunk_y_start;
	eig::Vector3i offset_chunk = offset + offset_chunk_from_image;

	eig::MatrixXf field_chunk;
	if (test_full_image) {
		eig::MatrixXf field = tsdf::generate_TSDF_2D_EWA_TSDF_inclusive(
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

		field_chunk = field.block(chunk_y_start, chunk_x_start, chunk_size, chunk_size);
	} else {
		field_chunk = tsdf::generate_TSDF_2D_EWA_TSDF_inclusive(
				200, // y coord
				depth_image,
				0.001f, //depth unit ratio
				camera_intrinsic_matrix,
				eig::Matrix4f::Identity(), //camera pose
				offset_chunk,
				chunk_size, //field size
				0.004f, //voxel size
				20 // narrow band width
				);
	}
	//TODO: add data for this test
	//BOOST_REQUIRE(math::almost_equal_verbose(field_chunk, test_data::out_sdf_chunk, 1e-6f));
}

BOOST_AUTO_TEST_CASE(test_EWA_2D_generation06) {
	math::MatrixXus depth_image;
	bool image_read = read_image_helper(depth_image, "zigzag_depth_00064.png");
	BOOST_REQUIRE(image_read);
	bool test_full_image = false;

	eig::Matrix3f camera_intrinsic_matrix;
	camera_intrinsic_matrix <<
			700.0f, 0.0f, 320.0f,
			0.0f, 700.0f, 240.0f,
			0.0f, 0.0f, 1.0f;
	eig::Vector3i offset;
	offset << -256, -256, 480;
	int chunk_x_start = 24, chunk_y_start = 10;
	int chunk_size = 16;
	eig::Vector3i offset_chunk_from_image;
	offset_chunk_from_image << chunk_x_start, 0.0f, chunk_y_start;
	eig::Vector3i offset_chunk = offset + offset_chunk_from_image;

	eig::MatrixXf field_chunk;
	if (test_full_image) {

		eig::MatrixXf field = tsdf::generate_TSDF_2D_EWA_TSDF_inclusive(
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
		field_chunk = field.block(chunk_y_start, chunk_x_start, chunk_size, chunk_size);
	} else {
		field_chunk = tsdf::generate_TSDF_2D_EWA_TSDF_inclusive(
				200, // y coord
				depth_image,
				0.001f, //depth unit ratio
				camera_intrinsic_matrix,
				eig::Matrix4f::Identity(), //camera pose
				offset_chunk,
				chunk_size, //field size
				0.004f, //voxel size
				20 // narrow band width
				);
	}
	//TODO test against data
	//BOOST_REQUIRE(math::almost_equal_verbose(field_chunk, test_data::out_sdf_chunk, 1e-6f));
}

BOOST_AUTO_TEST_CASE(test_EWA_3D_generation01) {
	math::MatrixXus depth_image;
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

	eig::Tensor<float, 3> field = tsdf::generate_TSDF_3D_EWA_image(
			depth_image,
			0.001f, //depth unit ratio
			camera_intrinsic_matrix,
			eig::Matrix4f::Identity(), //camera pose
			offset,
			field_size, //field size
			0.004f, //voxel size
			20 // narrow band width
			);

	BOOST_REQUIRE(math::almost_equal_verbose(field, test_data::TSDF_slice01, 1e-6f));
}

BOOST_AUTO_TEST_CASE(test_EWA_3D_generation02) {
	math::MatrixXus depth_image;
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

	eig::Tensor<float, 3> field = tsdf::generate_TSDF_3D_EWA_image(
			depth_image,
			0.001f, //depth unit ratio
			camera_intrinsic_matrix,
			eig::Matrix4f::Identity(), //camera pose
			offset,
			field_size, //field size
			0.004f, //voxel size
			20, // narrow band width
			0.5f
			);
	BOOST_REQUIRE(math::almost_equal_verbose(field, test_data::TSDF_slice02, 1e-5f));
}

// Following tests are for regular tsdf generation without interpolation
BOOST_AUTO_TEST_CASE(test_TSDF_2D_generation01) {
	math::MatrixXus depth_image;
	bool image_read = read_image_helper(depth_image, "zigzag2_depth_00108.png");
	BOOST_REQUIRE(image_read);
	eig::Matrix3f camera_intrinsic_matrix;
	camera_intrinsic_matrix <<
			700.0f, 0.0f, 320.0f,
			0.0f, 700.0f, 240.0f,
			0.0f, 0.0f, 1.0f;
	eig::Vector3i offset;
	offset << -8, -8, 144;
	int field_size = 16;
	eig::MatrixXf field = tsdf::generate_TSDF_2D(
			200, // y coord
			depth_image,
			0.001f, //depth unit ratio
			camera_intrinsic_matrix,
			eig::Matrix4f::Identity(), //camera pose
			offset,
			field_size, //field size
			0.004f, //voxel size
			20, // narrow band width
			-999.f
			);
	BOOST_REQUIRE(math::almost_equal_verbose(field, test_data::expected_tsdf_field01, 1e-6f));
}

BOOST_AUTO_TEST_CASE(test_TSDF_2D_generation02) {
	math::MatrixXus depth_image;
	bool image_read = read_image_helper(depth_image, "zigzag2_depth_00108.png");
	BOOST_REQUIRE(image_read);
	eig::Matrix3f camera_intrinsic_matrix;
	camera_intrinsic_matrix <<
			700.0f, 0.0f, 320.0f,
			0.0f, 700.0f, 240.0f,
			0.0f, 0.0f, 1.0f;
	eig::Matrix4f camera_extrinsic_matrix = eig::Matrix4f::Identity();
	camera_extrinsic_matrix(2, 3) = 0.004;

	eig::Vector3i offset;
	offset << -8, -8, 144;
	int field_size = 16;
	eig::MatrixXf field = tsdf::generate_TSDF_2D(
			200, // y coord
			depth_image,
			0.001f, //depth unit ratio
			camera_intrinsic_matrix,
			camera_extrinsic_matrix, //camera pose
			offset,
			field_size, //field size
			0.004f, //voxel size
			20, // narrow band width
			-999.f
	);
	BOOST_REQUIRE(math::almost_equal_verbose(field, test_data::expected_tsdf_field02, 1e-6f));
}