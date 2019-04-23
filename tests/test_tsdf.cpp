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
//_DEBUG
#include <iostream>
#include "../src/console/pretty_printers.hpp"

//libraries
#include <boost/test/unit_test.hpp>
#include <boost/python.hpp>
#include <Eigen/Dense>

//test data
#include "data/test_data_tsdf.hpp"

//test targets
#include "../src/tsdf/generator.hpp"
#include "../src/math/typedefs.hpp"
#include "../src/math/almost_equal.hpp"
#include "../src/image_io/png_eigen.hpp"
#include "common.hpp"

namespace eig = Eigen;


BOOST_AUTO_TEST_CASE(test_tsdf_generation_no_interpolation_1) {
	math::MatrixXus depth_image;
	bool image_read = read_image_helper(depth_image, "zigzag2_depth_00108.png");
	BOOST_REQUIRE(image_read);
	eig::Matrix3f camera_intrinsic_matrix;
	camera_intrinsic_matrix <<
			700.0f, 0.0f, 320.0f,
			0.0f, 700.0f, 240.0f,
			0.0f, 0.0f, 1.0f;
	tsdf::Parameters2d parameters(
				0.001f, //depth unit ratio
				camera_intrinsic_matrix, //projection matrix
				0.05f, //near clipping distance
				math::Vector2i(-8, 144), //offset
				math::Vector2i(16, 16), //shape
				0.004f, //voxel size
				20, //narrow band width
				tsdf::InterpolationMethod::NONE
				);
	tsdf::Generator2d generator(parameters);
	eig::MatrixXf field = generator.generate(depth_image, eig::Matrix4f::Identity(), 200);
	BOOST_REQUIRE(math::almost_equal_verbose(field, test_data::expected_tsdf_field01, 1e-6f));
}

BOOST_AUTO_TEST_CASE(test_tsdf_generation_no_interpolation_2) {
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
	tsdf::Parameters2d parameters(
					0.001f, //depth unit ratio
					camera_intrinsic_matrix, //projection matrix
					0.05f, //near clipping distance
					math::Vector2i(-8, 144), //offset
					math::Vector2i(16, 16), //shape
					0.004f, //voxel size
					20, //narrow band width
					tsdf::InterpolationMethod::NONE
					);
	tsdf::Generator2d generator(parameters);
	eig::MatrixXf field = generator.generate(depth_image, camera_extrinsic_matrix, 200);
	BOOST_REQUIRE(math::almost_equal_verbose(field, test_data::expected_tsdf_field02, 1e-6f));
}

BOOST_AUTO_TEST_CASE(test_tsdf_generation_EWA_2D_image_space_1) {

	math::MatrixXus depth_image = math::MatrixXus::Constant(3, 640, USHRT_MAX);
	depth_image.block(0, 399, 3, 18) = test_data::depth_image_region;

	eig::Matrix3f camera_intrinsic_matrix;
	camera_intrinsic_matrix <<
			700.0f, 0.0f, 320.0f,
			0.0f, 700.0f, 240.0f,
			0.0f, 0.0f, 1.0f;

	tsdf::Parameters2d parameters(
			0.001f, //depth unit ratio
			camera_intrinsic_matrix, //projection matrix
			0.05f, //near clipping distance
			math::Vector2i(94, 804), //offset
			math::Vector2i(16, 16), //shape
			0.004f, //voxel size
			20, //narrow band width
			tsdf::InterpolationMethod::EWA_IMAGE_SPACE,
			1.0f // gaussian covariance scale
			);
	tsdf::Generator2d generator(parameters);
	eig::MatrixXf field = generator.generate(depth_image, eig::Matrix4f::Identity(), 1);

	BOOST_REQUIRE(math::almost_equal_verbose(field, test_data::out_sdf_field, 1e-6f));
}

BOOST_AUTO_TEST_CASE(test_tsdf_generation_EWA_2D_image_space_2) {
	math::MatrixXus depth_image;
	bool image_read = read_image_helper(depth_image, "zigzag2_depth_00108.png");
	BOOST_REQUIRE(image_read);
	bool test_full_image = false;

	eig::Matrix3f camera_intrinsic_matrix;
	camera_intrinsic_matrix <<
			700.0f, 0.0f, 320.0f,
			0.0f, 700.0f, 240.0f,
			0.0f, 0.0f, 1.0f;
	math::Vector2i offset(-256, 0);
	int chunk_x_start = 210, chunk_y_start = 103;
	int chunk_size = 16;
	math::Vector2i offset_chunk_from_image(chunk_x_start, chunk_y_start);
	math::Vector2i offset_chunk = offset + offset_chunk_from_image;

	eig::MatrixXf field_chunk;
	if (test_full_image) {
		tsdf::Parameters2d parameters(
				0.001f, //depth unit ratio
				camera_intrinsic_matrix, //projection matrix
				0.05f, //near clipping distance
				offset,
				math::Vector2i(512, 512), //shape
				0.004f, //voxel size
				20, //narrow band width
				tsdf::InterpolationMethod::EWA_IMAGE_SPACE,
				1.0f // gaussian covariance scale
				);
		tsdf::Generator2d generator(parameters);
		eig::MatrixXf field = generator.generate(depth_image, eig::Matrix4f::Identity(), 200);

		field_chunk = field.block(chunk_y_start, chunk_x_start, chunk_size, chunk_size);
	} else {
		tsdf::Parameters2d parameters(
				0.001f, //depth unit ratio
				camera_intrinsic_matrix, //projection matrix
				0.05f, //near clipping distance
				offset_chunk,
				math::Vector2i(chunk_size, chunk_size), //shape
				0.004f, //voxel size
				20, //narrow band width
				tsdf::InterpolationMethod::EWA_IMAGE_SPACE,
				1.0f // gaussian covariance scale
				);
		tsdf::Generator2d generator(parameters);
		field_chunk = generator.generate(depth_image, eig::Matrix4f::Identity(), 200);
	}
	BOOST_REQUIRE(math::almost_equal_verbose(field_chunk, test_data::out_sdf_chunk, 1e-6f));
}

BOOST_AUTO_TEST_CASE(test_tsdf_generation_EWA_2D_voxel_space_inclusive_1) {
	math::MatrixXus depth_image;
	bool image_read = read_image_helper(depth_image, "zigzag2_depth_00108.png");
	BOOST_REQUIRE(image_read);
	bool test_full_image = false;

	eig::Matrix3f camera_intrinsic_matrix;
	camera_intrinsic_matrix <<
			700.0f, 0.0f, 320.0f,
			0.0f, 700.0f, 240.0f,
			0.0f, 0.0f, 1.0f;
	math::Vector2i offset(-256, 0);
	int chunk_x_start = 210, chunk_y_start = 103;
	int chunk_size = 16;
	math::Vector2i offset_chunk_from_image(chunk_x_start, chunk_y_start);
	math::Vector2i offset_chunk = offset + offset_chunk_from_image;

	eig::MatrixXf field_chunk;
	if (test_full_image) {
		tsdf::Parameters2d parameters(
				0.001f, //depth unit ratio
				camera_intrinsic_matrix, //projection matrix
				0.05f, //near clipping distance
				offset_chunk,
				math::Vector2i(512, 512), //shape
				0.004f, //voxel size
				20, //narrow band width
				tsdf::InterpolationMethod::EWA_VOXEL_SPACE_INCLUSIVE,
				1.0f // gaussian covariance scale
				);
		tsdf::Generator2d generator(parameters);
		eig::MatrixXf field = generator.generate(depth_image, eig::Matrix4f::Identity(), 200);
		field_chunk = field.block(chunk_y_start, chunk_x_start, chunk_size, chunk_size);
	} else {
		tsdf::Parameters2d parameters(
				0.001f, //depth unit ratio
				camera_intrinsic_matrix, //projection matrix
				0.05f, //near clipping distance
				offset_chunk,
				math::Vector2i(chunk_size, chunk_size), //shape
				0.004f, //voxel size
				20, //narrow band width
				tsdf::InterpolationMethod::EWA_VOXEL_SPACE_INCLUSIVE,
				1.0f // gaussian covariance scale
				);
		tsdf::Generator2d generator(parameters);
		field_chunk = generator.generate(depth_image, eig::Matrix4f::Identity(), 200);
	}

	BOOST_REQUIRE(math::almost_equal_verbose(field_chunk, test_data::expected_tsdf_EWA_voxel_space_inclusive_1, 1e-6f));
}

BOOST_AUTO_TEST_CASE(test_tsdf_generation_EWA_2D_voxel_space_inclusive_2) {
	math::MatrixXus depth_image;
	bool image_read = read_image_helper(depth_image, "zigzag_depth_00064.png");
	BOOST_REQUIRE(image_read);
	bool test_full_image = false;

	eig::Matrix3f camera_intrinsic_matrix;
	camera_intrinsic_matrix <<
			700.0f, 0.0f, 320.0f,
			0.0f, 700.0f, 240.0f,
			0.0f, 0.0f, 1.0f;
	math::Vector2i offset(-256, 480);
	int chunk_x_start = 24, chunk_y_start = 10;
	int chunk_size = 16;
	math::Vector2i offset_chunk_from_image(chunk_x_start, chunk_y_start);
	math::Vector2i offset_chunk = offset + offset_chunk_from_image;

	eig::MatrixXf field_chunk;
	if (test_full_image) {
		tsdf::Parameters2d parameters(0.001f, //depth unit ratio
				camera_intrinsic_matrix, //projection matrix
				0.05f, //near clipping distance
				offset_chunk, //offset of scene from world origin
				math::Vector2i(512, 512), //shape
				0.004f, //voxel size
				20, //narrow band width
				tsdf::InterpolationMethod::EWA_VOXEL_SPACE_INCLUSIVE,
				1.0f // gaussian covariance scale
				);
		tsdf::Generator2d generator(parameters);
		eig::MatrixXf field = generator.generate(depth_image, eig::Matrix4f::Identity(), 200);
		field_chunk = field.block(chunk_y_start, chunk_x_start, chunk_size, chunk_size);
	} else {
		tsdf::Parameters2d parameters(
				0.001f, //depth unit ratio
				camera_intrinsic_matrix, //projection matrix
				0.05f, //near clipping distance
				offset_chunk,
				math::Vector2i(chunk_size, chunk_size), //shape
				0.004f, //voxel size
				20, //narrow band width
				tsdf::InterpolationMethod::EWA_VOXEL_SPACE_INCLUSIVE,
				1.0f // gaussian covariance scale
				);
		tsdf::Generator2d generator(parameters);
		field_chunk = generator.generate(depth_image, eig::Matrix4f::Identity(), 200);
	}
	BOOST_REQUIRE(math::almost_equal_verbose(field_chunk, test_data::expected_tsdf_EWA_voxel_space_inclusive_2, 1e-6f));
}

BOOST_AUTO_TEST_CASE(test_tsdf_generation_EWA_3D_image_space_1) {
	math::MatrixXus depth_image;
	bool image_read = read_image_helper(depth_image, "zigzag2_depth_00108.png");
	BOOST_REQUIRE(image_read);

	math::Vector3i offset(-46, -8, 105);
	math::Vector3i field_shape(16, 1, 16);
	eig::Matrix3f camera_intrinsic_matrix;
	camera_intrinsic_matrix <<
			700.0f, 0.0f, 320.0f,
			0.0f, 700.0f, 240.0f,
			0.0f, 0.0f, 1.0f;

	tsdf::Parameters3d parameters(0.001f, //depth unit ratio
						camera_intrinsic_matrix, //projection matrix
						0.05f, //near clipping distance
						offset, //offset of scene from world origin
						field_shape, //dimensions of the voxel grid
						0.004f, //voxel size
						20, //narrow band width
						tsdf::InterpolationMethod::EWA_IMAGE_SPACE,
						1.0f // gaussian covariance scale
						);
	tsdf::Generator3d generator(parameters);
	eig::Tensor<float, 3> field = generator.generate(depth_image, eig::Matrix4f::Identity());

	BOOST_REQUIRE(math::almost_equal_verbose(field, test_data::TSDF_slice01, 1e-6f));
}

BOOST_AUTO_TEST_CASE(test_tsdf_generation_EWA_3D_image_space_2) {
	math::MatrixXus depth_image;
	bool image_read = read_image_helper(depth_image, "zigzag2_depth_00108.png");
	BOOST_REQUIRE(image_read);

	math::Vector3i offset(-46, -8, 105);
	math::Vector3i field_shape(16, 1, 16); //zigzag2-108
	eig::Matrix3f camera_intrinsic_matrix;
	camera_intrinsic_matrix <<
			700.0f, 0.0f, 320.0f,
			0.0f, 700.0f, 240.0f,
			0.0f, 0.0f, 1.0f;
	tsdf::Parameters3d parameters(0.001f, //depth unit ratio
							camera_intrinsic_matrix, //projection matrix
							0.05f, //near clipping distance
							offset, //offset of scene from world origin
							field_shape, //dimensions of the voxel grid
							0.004f, //voxel size
							20, //narrow band width
							tsdf::InterpolationMethod::EWA_IMAGE_SPACE,
							0.5f // gaussian covariance scale
							);
	tsdf::Generator3d generator(parameters);
	eig::Tensor<float, 3> field = generator.generate(depth_image, eig::Matrix4f::Identity());
	BOOST_REQUIRE(math::almost_equal_verbose(field, test_data::TSDF_slice02, 1e-5f));
}

