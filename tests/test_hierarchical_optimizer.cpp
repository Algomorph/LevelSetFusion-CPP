/*
 * test_hierarchical_optimizer.cpp
 *
 *  Created on: Dec 20, 2018
 *      Author: Gregory Kramida
 *   Copyright: 2018 Gregory Kramida
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

#define BOOST_TEST_MODULE test_hierarchical_optimizer

// standard library

// libraries
#include <boost/test/unit_test.hpp>
#include <boost/python.hpp>
#include <Eigen/Eigen>

// test data

#include "data/test_data_hierarchical_optimizer.hpp"

// test targets
#include "../src/math/tensors.hpp"
#include "../src/math/assessment.hpp"
#include "../src/nonrigid_optimization/hierarchical/pyramid2d.hpp"
#include "../src/nonrigid_optimization/hierarchical/optimizer2d.hpp"
#include "../src/nonrigid_optimization/field_warping.hpp"


namespace eig = Eigen;

namespace nro_h = nonrigid_optimization::hierarchical;
namespace nro = nonrigid_optimization;

BOOST_AUTO_TEST_CASE(power_of_two_test01){
	BOOST_REQUIRE(nro_h::is_power_of_two(128));
	BOOST_REQUIRE(nro_h::is_power_of_two(2));
	BOOST_REQUIRE(nro_h::is_power_of_two(16));
	BOOST_REQUIRE(!nro_h::is_power_of_two(17));
	BOOST_REQUIRE(!nro_h::is_power_of_two(38));
}

BOOST_AUTO_TEST_CASE(pyramid_test01) {
	// corresponds to test_contstruct_scalar_pyramid for Python

	eig::MatrixXf tile(8, 8);
	tile <<
			1.0f, 2.0f, 5.0f, 6.0f, -1.0f, -2.0f, -5.0f, -6.0f,
			3.0f, 4.0f, 7.0f, 8.0f, -3.0f, -4.0f, -7.0f, -8.0f,
			-1.0f, -2.0f, -5.0f, -6.0f, 1.0f, 2.0f, 5.0f, 6.0f,
			-3.0f, -4.0f, -7.0f, -8.0f, 3.0f, 4.0f, 7.0f, 8.0f,
			1.0f, 2.0f, 5.0f, 6.0f, 5.0f, 5.0f, 5.0f, 5.0f,
			3.0f, 4.0f, 7.0f, 8.0f, 5.0f, 5.0f, 5.0f, 5.0f,
			-1.0f, -2.0f, -5.0f, -6.0f, 5.0f, 5.0f, 5.0f, 5.0f,
			-3.0f, -4.0f, -7.0f, -8.0f, 5.0f, 5.0f, 5.0f, 5.0f;

	// results in shape 128 x 128
	eig::MatrixXf field = tile.replicate(16, 16);

	nro_h::Pyramid2d pyramid(field);
	BOOST_REQUIRE_EQUAL(pyramid.get_level_count(), (unsigned)4);
	BOOST_REQUIRE_EQUAL(pyramid.get_level(0).rows(), 16);
	BOOST_REQUIRE_EQUAL(pyramid.get_level(0).cols(), 16);
	BOOST_REQUIRE_EQUAL(pyramid.get_level(1).rows(), 32);
	BOOST_REQUIRE_EQUAL(pyramid.get_level(2).rows(), 64);
	BOOST_REQUIRE_EQUAL(pyramid.get_level(3).rows(), 128);
	BOOST_REQUIRE_EQUAL(pyramid.get_level(3).cols(), 128);

	float l2_00 = tile.block<2, 2>(0, 0).mean();
	float l2_10 = tile.block<2, 2>(2, 0).mean();
	float l2_01 = tile.block<2, 2>(0, 2).mean();
	float l2_11 = tile.block<2, 2>(2, 2).mean();
	float l2_02 = -l2_00;
	float l2_12 = -l2_10;
	float l2_03 = -l2_01;
	float l2_13 = -l2_11;

	BOOST_REQUIRE_EQUAL(pyramid.get_level(2)(0, 0), l2_00);
	BOOST_REQUIRE_EQUAL(pyramid.get_level(2)(1, 0), l2_10);
	BOOST_REQUIRE_EQUAL(pyramid.get_level(2)(0, 1), l2_01);
	BOOST_REQUIRE_EQUAL(pyramid.get_level(2)(1, 1), l2_11);
	BOOST_REQUIRE_EQUAL(pyramid.get_level(2)(0, 0 + 2), l2_02);
	BOOST_REQUIRE_EQUAL(pyramid.get_level(2)(1, 0 + 2), l2_12);
	BOOST_REQUIRE_EQUAL(pyramid.get_level(2)(0, 1 + 2), l2_03);
	BOOST_REQUIRE_EQUAL(pyramid.get_level(2)(1, 1 + 2), l2_13);
	BOOST_REQUIRE_EQUAL(pyramid.get_level(2)(0 + 4, 0), l2_00);
	BOOST_REQUIRE_EQUAL(pyramid.get_level(2)(1 + 4, 0), l2_10);
	BOOST_REQUIRE_EQUAL(pyramid.get_level(2)(0 + 4, 1), l2_01);
	BOOST_REQUIRE_EQUAL(pyramid.get_level(2)(1 + 4, 1), l2_11);
	BOOST_REQUIRE_EQUAL(pyramid.get_level(2)(0, 0 + 4), l2_00);
	BOOST_REQUIRE_EQUAL(pyramid.get_level(2)(1, 0 + 4), l2_10);
	BOOST_REQUIRE_EQUAL(pyramid.get_level(2)(0, 1 + 4), l2_01);
	BOOST_REQUIRE_EQUAL(pyramid.get_level(2)(1, 1 + 4), l2_11);

	float l1_00 = (l2_00 + l2_10 + l2_01 + l2_11) / 4.0f;
	float l1_01 = (l2_02 + l2_12 + l2_03 + l2_13) / 4.0f;
	float l1_10 = l1_00;
	float l1_11 = 5.0;

	BOOST_REQUIRE_EQUAL(pyramid.get_level(1)(0, 0), l1_00);
	BOOST_REQUIRE_EQUAL(pyramid.get_level(1)(1, 0), l1_10);
	BOOST_REQUIRE_EQUAL(pyramid.get_level(1)(0, 1), l1_01);
	BOOST_REQUIRE_EQUAL(pyramid.get_level(1)(1, 1), l1_11);
	BOOST_REQUIRE_EQUAL(pyramid.get_level(0)(0, 0), 5.0f/4.0f);

	BOOST_REQUIRE(true);
}

BOOST_AUTO_TEST_CASE(resample_field_test01){
	//corresponds to test_resample_field01 in python code
	eig::MatrixXf resampled_field = nro::resample_field(field_A_16x16,warp_field_A_16x16);
	BOOST_REQUIRE(resampled_field.isApprox(fA_resampled_with_wfA));
}

BOOST_AUTO_TEST_CASE(resample_field_test02){
	//corresponds to test_resample_field_replacement01 in python code
	eig::MatrixXf resampled_field = nro::resample_field_replacement(field_B_16x16,warp_field_B_16x16,0);
	BOOST_REQUIRE(resampled_field.isApprox(fB_resampled_with_wfB_replacement));
}

BOOST_AUTO_TEST_CASE(test_hierarchical_optimizer01){
	// corresponds to test_construction-and_operation in python code (test_hns_optimizer2d.py)
	nro_h::Optimizer2d<true> optimizer(
			false, false,
			8,
			0.2,
			100, //max iteration count
			0.001, //max warp update threshold
			1.0,
			0.2,
			eig::VectorXf(0),
			nro_h::Optimizer2d<true>::VerbosityParameters()
			);
	math::MatrixXv2f warp_field_out = optimizer.optimize(canonical_field, live_field);
	eig::MatrixXf final_live_resampled = nro::resample_field(live_field,warp_field_out);

	BOOST_REQUIRE(math::matrix_almost_equal_verbose(warp_field_out,warp_field,10e-6));
	BOOST_REQUIRE(final_live_resampled.isApprox(final_live_field));
}
