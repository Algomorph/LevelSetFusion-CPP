//  Created by Gregory Kramida on 10/29/18.
//  Copyright (c) 2018 Gregory Kramida
//  Licensed under the Apache License, Version 2.0 (the "License");
//  you may not use this file except in compliance with the License.
//  You may obtain a copy of the License at

//  http://www.apache.org/licenses/LICENSE-2.0

//  Unless required by applicable law or agreed to in writing, software
//  distributed under the License is distributed on an "AS IS" BASIS,
//  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
//  See the License for the specific language governing permissions and
//  limitations under the License.
//  ================================================================

#define BOOST_TEST_MODULE test_math

//stdlib
//_DEBUG
#include <iostream>

//libraries
#include <boost/test/unit_test.hpp>
#include <boost/python.hpp>
#include <Eigen/Eigen>

//test data
#include "data/test_data_math.hpp"

//test targets
#include "../src/math/gradients.hpp"
#include "../src/math/almost_equal.hpp"
#include "../src/math/typedefs.hpp"
#include "../src/math/statistics.hpp"
#include "../src/math/padding.hpp"

namespace eig = Eigen;

BOOST_AUTO_TEST_CASE(scalar_field_gradient_test01) {
	eig::MatrixXf field(2,2);
	field << -0.46612028, -0.8161121,
	0.2427629, -0.79432599;

	eig::MatrixXf expected_gradient_x(2,2), expected_gradient_y(2,2);
	expected_gradient_x << -0.34999183, -0.34999183,
			-1.03708889, -1.03708889;
	expected_gradient_y << 0.70888318, 0.02178612,
			0.70888318, 0.02178612;
	eig::MatrixXf gradient_x, gradient_y;
	math::gradient(gradient_x, gradient_y, field);

	BOOST_REQUIRE(gradient_x.isApprox(expected_gradient_x));
	BOOST_REQUIRE(gradient_y.isApprox(expected_gradient_y));
}

BOOST_AUTO_TEST_CASE(scalar_field_gradient_test02) {
	eig::MatrixXf field(3,3);
	field << 0.11007435, -0.94589225, -0.54835034,
			-0.09617922, 0.15561824, 0.60624432,
			-0.83068796, 0.19262577, -0.21090505;

	eig::MatrixXf expected_gradient_x(3,3), expected_gradient_y(3,3);
	expected_gradient_x << -1.0559666, -0.32921235, 0.39754191,
			0.25179745, 0.35121177, 0.45062608,
			1.02331373, 0.30989146, -0.40353082;
	expected_gradient_y << -0.20625357, 1.10151049, 1.15459466,
			-0.47038115, 0.56925901, 0.16872265,
			-0.73450874, 0.03700753, -0.81714937;

	eig::MatrixXf gradient_x, gradient_y;
	math::gradient(gradient_x, gradient_y, field);

	BOOST_REQUIRE(gradient_x.isApprox(expected_gradient_x));
	BOOST_REQUIRE(gradient_y.isApprox(expected_gradient_y));
}

BOOST_AUTO_TEST_CASE(scalar_field_gradient_test03) {
	eig::MatrixXf gradient_x, gradient_y;
	math::gradient(gradient_x, gradient_y, test_data::field);

	BOOST_REQUIRE(gradient_x.isApprox(test_data::expected_gradient_x));
	BOOST_REQUIRE(gradient_y.isApprox(test_data::expected_gradient_y));
}

BOOST_AUTO_TEST_CASE(scalar_field_gradient_test04) {
	eig::Matrix2f field;
	field << -0.46612028, -0.8161121,
			0.2427629, -0.79432599;

	math::MatrixXv2f expected_gradient(2, 2);
	expected_gradient <<
			// @formatter:off
		math::Vector2f(-0.34999183f, 0.70888318f), math::Vector2f(-0.34999183f, 0.02178612f),
		math::Vector2f(-1.03708889f, 0.70888318f), math::Vector2f(-1.03708889f, 0.02178612f);  // @formatter:on

	math::MatrixXv2f gradient;
	math::gradient(gradient, static_cast<eig::MatrixXf>(field));

	BOOST_REQUIRE(math::almost_equal(gradient, expected_gradient, 1e-6));
}

BOOST_AUTO_TEST_CASE(scalar_field_gradient_test05) {
	namespace eig = Eigen;

	eig::Matrix3f field;
	field << 0.11007435, -0.94589225, -0.54835034,
			-0.09617922, 0.15561824, 0.60624432,
			-0.83068796, 0.19262577, -0.21090505;

	math::MatrixXv2f expected_gradient(3, 3);
	expected_gradient <<
			// @formatter:off
			math::Vector2f(-1.0559666f, -0.20625357f), math::Vector2f(-0.32921235f, 1.10151049f), math::Vector2f(
			0.39754191f, 1.15459466f),
			math::Vector2f(0.25179745f, -0.47038115f), math::Vector2f(0.35121177f, 0.56925901f), math::Vector2f(
					0.45062608f, 0.16872265f),
			math::Vector2f(1.02331373f, -0.73450874f), math::Vector2f(0.30989146f, 0.03700753f), math::Vector2f(
					-0.40353082f, -0.81714937f);
																									// @formatter:on
	math::MatrixXv2f gradient;
	math::gradient(gradient, static_cast<eig::MatrixXf>(field));

	BOOST_REQUIRE(math::almost_equal(gradient, expected_gradient, 1e-6));
}

BOOST_AUTO_TEST_CASE(scalar_field_gradient_test06) {
	namespace eig = Eigen;

	math::MatrixXv2f gradient;
	math::gradient(gradient, test_data::field);

	eig::MatrixXf exp_grad_x = test_data::expected_gradient_x;
	eig::MatrixXf exp_grad_y = test_data::expected_gradient_y;

	math::MatrixXv2f expected_gradient = math::stack_as_xv2f(test_data::expected_gradient_x,
			test_data::expected_gradient_y);
	BOOST_REQUIRE(math::almost_equal(gradient, expected_gradient, 1e-6));
}

BOOST_AUTO_TEST_CASE(vector_field_gradient_test01) {
	math::MatrixXv2f vector_field(2, 2);
	vector_field << //@formatter:off
			math::Vector2f(0.0f, 0.0f), math::Vector2f(1.0f, -1.0f),
			math::Vector2f(-1.0f, 1.0f), math::Vector2f(1.0f, 1.0f);
																									//@formatter:on

	math::MatrixXm2f gradient;
	math::gradient(gradient, vector_field);

	math::MatrixXm2f expected_gradient(2, 2);
	expected_gradient << math::Matrix2f(1.0f, -1.0f, -1.0f, 1.0f), math::Matrix2f(1.0f, 0.0f, -1.0f, 2.0f),
			math::Matrix2f(2.0f, -1.0f, 0.0f, 1.0f), math::Matrix2f(2.0f, 0.0f, 0.0f, 2.0f);

	BOOST_REQUIRE(math::almost_equal(gradient, expected_gradient, 1e-6));
}

BOOST_AUTO_TEST_CASE(vector_field_gradient_test02) {
	math::MatrixXm2f gradient;
	math::gradient(gradient, test_data::vector_field);
	BOOST_REQUIRE(math::almost_equal(gradient, test_data::expected_vector_field_gradient, 1e-6));
}

BOOST_AUTO_TEST_CASE(max_norm_test01) {
	math::MatrixXv2f vector_field(4, 4);
	vector_field <<
			math::Vector2f(0.0f, 0.0f),
			math::Vector2f(-0.02738971f, -0.02738965f),
			math::Vector2f(-0.36405864f, -0.19928734f),
			math::Vector2f(-0.13360168f, -0.18600166f),

			math::Vector2f(0.0f, 0.0f),
			math::Vector2f(-0.44864437f, -0.41914698f),
			math::Vector2f(-0.12387292f, -0.20939942f),
			math::Vector2f(-0.05534932f, -0.20074867f),

			math::Vector2f(0.0f, 0.0f),
			math::Vector2f(-0.7164666f, -0.25778353f),
			math::Vector2f(-0.10596427f, -0.1720703f),
			math::Vector2f(-0.05534932f, -0.24724902f),

			math::Vector2f(0.0f, 0.0f),
			math::Vector2f(-0.57307416f, -0.20139425f),
			math::Vector2f(-0.09964357f, -0.15942998f),
			math::Vector2f(-0.1336017f, -0.27605268f);
	math::Vector2i coordinates;
	float max_norm;
	math::locate_max_norm(max_norm, coordinates, vector_field);

	BOOST_REQUIRE_CLOSE(max_norm, 0.7614307f, 1e-8);
	BOOST_REQUIRE_EQUAL(coordinates.x, 1);
	BOOST_REQUIRE_EQUAL(coordinates.y, 2);
}

BOOST_AUTO_TEST_CASE(mean_and_std_test01) {
	float mean, std;

	float mean_length = math::mean_vector_length(test_data::vector_field2);
	BOOST_REQUIRE_CLOSE(mean_length, 0.7339459657669067f, 1e-8);

	math::mean_and_std_vector_length(mean, std, test_data::vector_field);
	BOOST_REQUIRE_CLOSE(mean, 0.7791687f, 1e-8);
	BOOST_REQUIRE_CLOSE(std, 0.26306444f, 1e-8);
}

BOOST_AUTO_TEST_CASE(test_pad_replicate01) {
	math::Tensor3f input(3, 3, 3);
	input.setValues(  // @formatter:off
				{{{1.0f, 10.0f, 19.0f},
				  {4.0f, 13.0f, 22.0f},
				  {7.0f, 16.0f, 25.0f}},
				 {{2.0f, 11.0f, 20.0f},
				  {5.0f, 14.0f, 23.0f},
				  {8.0f, 17.0f, 26.0f}},
				 {{3.0f, 12.0f, 21.0f},
				  {6.0f, 15.0f, 24.0f},
				  {9.0f, 18.0f, 27.0f}}}
	);  // @formatter:on

	math::Tensor3f expected_output(5, 5, 5);
	expected_output.setValues(  // @formatter:off
			{{{ 1.0f,  1.0f, 10.0f, 19.0f, 19.0f},
			  { 1.0f,  1.0f, 10.0f, 19.0f, 19.0f},
			  { 4.0f,  4.0f, 13.0f, 22.0f, 22.0f},
			  { 7.0f,  7.0f, 16.0f, 25.0f, 25.0f},
			  { 7.0f,  7.0f, 16.0f, 25.0f, 25.0f}},

			 {{ 1.0f,  1.0f, 10.0f, 19.0f, 19.0f},
			  { 1.0f,  1.0f, 10.0f, 19.0f, 19.0f},
			  { 4.0f,  4.0f, 13.0f, 22.0f, 22.0f},
			  { 7.0f,  7.0f, 16.0f, 25.0f, 25.0f},
			  { 7.0f,  7.0f, 16.0f, 25.0f, 25.0f}},

			 {{ 2.0f,  2.0f, 11.0f, 20.0f, 20.0f},
			  { 2.0f,  2.0f, 11.0f, 20.0f, 20.0f},
			  { 5.0f,  5.0f, 14.0f, 23.0f, 23.0f},
			  { 8.0f,  8.0f, 17.0f, 26.0f, 26.0f},
			  { 8.0f,  8.0f, 17.0f, 26.0f, 26.0f}},

			 {{ 3.0f,  3.0f, 12.0f, 21.0f, 21.0f},
			  { 3.0f,  3.0f, 12.0f, 21.0f, 21.0f},
			  { 6.0f,  6.0f, 15.0f, 24.0f, 24.0f},
			  { 9.0f,  9.0f, 18.0f, 27.0f, 27.0f},
			  { 9.0f,  9.0f, 18.0f, 27.0f, 27.0f}},

			 {{ 3.0f,  3.0f, 12.0f, 21.0f, 21.0f},
			  { 3.0f,  3.0f, 12.0f, 21.0f, 21.0f},
			  { 6.0f,  6.0f, 15.0f, 24.0f, 24.0f},
			  { 9.0f,  9.0f, 18.0f, 27.0f, 27.0f},
			  { 9.0f,  9.0f, 18.0f, 27.0f, 27.0f}}});  // @formatter:on

	math::Tensor3f output = math::pad_replicate(input, 1);

	BOOST_REQUIRE(math::almost_equal_verbose(output, expected_output, 1e-6));
}

