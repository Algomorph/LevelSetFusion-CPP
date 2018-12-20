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

// test targets
#include "../src/nonrigid_optimization/pyramid2d.hpp"

namespace eig = Eigen;

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

	nonrigid_optimization::Pyramid2d pyramid(field);
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

