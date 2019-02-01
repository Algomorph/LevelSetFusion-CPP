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
#include "test_data_tsdf.hpp"

//test targets
#include "../src/tsdf/ewa.hpp"
#include "../src/math/typedefs.hpp"
#include "../src/math/assessment.hpp"

namespace eig = Eigen;



BOOST_AUTO_TEST_CASE(test_EWA_generation01){

	eig::MatrixXus depth_image = eig::MatrixXus::Constant(3,640,USHRT_MAX);
	depth_image.block(0,399,3,18) = test_data::depth_image_region;

	eig::Matrix3f camera_intrinsic_matrix;
	camera_intrinsic_matrix <<
			700.0f, 0.0f, 320.0f,
			0.0f, 700.0f, 240.0f,
			0.0f,0.0f,1.0f;
	eig::Vector3i offset;
		offset << 94, -256, 804;

	eig::MatrixXf field = tsdf::generate_2d_TSDF_field_from_depth_image_EWA(
		1, // y coord
		depth_image,
		0.001f, //depth unit ratio
		camera_intrinsic_matrix,
		eig::MatrixXf::Identity(4,4), //camera pose
		offset,
		16, //field size
		0.004f, //voxel size
		20 // narrow band width
		);

	BOOST_REQUIRE(math::almost_equal_verbose(field,test_data::out_sdf_field,2e-5));
}
