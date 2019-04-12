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
#include <iostream>

//libraries
#include <boost/test/unit_test.hpp>
#include <boost/python.hpp>
#include <Eigen/Eigen>

//utils
#include "../src/console/pretty_printers.hpp"

//test data
#include "data/test_data_math.hpp"

//test targets
#include "../src/math/checks.hpp"
#include "../src/math/gradients.hpp"
#include "../src/math/almost_equal.hpp"
#include "../src/math/typedefs.hpp"
#include "../src/math/convolution.hpp"
#include "../src/math/statistics.hpp"
#include "../src/math/resampling.hpp"
#include "../src/math/padding.hpp"

namespace eig = Eigen;

BOOST_AUTO_TEST_CASE(power_of_two_test01) {
	BOOST_REQUIRE(math::is_power_of_two(128));
	BOOST_REQUIRE(math::is_power_of_two(2));
	BOOST_REQUIRE(math::is_power_of_two(16));
	BOOST_REQUIRE(!math::is_power_of_two(17));
	BOOST_REQUIRE(!math::is_power_of_two(38));
}

BOOST_AUTO_TEST_CASE(scalar_field_gradient_test01) {
	eig::Matrix2f field;
	field << -0.46612028, -0.8161121,
	0.2427629, -0.79432599;

	eig::Matrix2f expected_gradient_x, expected_gradient_y;
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
	eig::Matrix3f field;
	field << 0.11007435, -0.94589225, -0.54835034,
			-0.09617922, 0.15561824, 0.60624432,
			-0.83068796, 0.19262577, -0.21090505;

	eig::Matrix3f expected_gradient_x, expected_gradient_y;
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
	math::gradient(gradient, field);

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
	math::gradient(gradient, field);

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

BOOST_AUTO_TEST_CASE(convolution_test01) {
	eig::MatrixXf field(3, 3);
	field << 1.f, 4.f, 7.f, 2.f, 5.f, 8.f, 3.f, 6.f, 9.f;
	math::MatrixXv2f vector_field = math::stack_as_xv2f(field, field);
	eig::VectorXf kernel(3);
	kernel << 3.f, 2.f, 1.f;
	field << 85.f, 168.f, 99.f, 124.f, 228.f, 132.f, 67.f, 120.f, 69.f;
	math::MatrixXv2f expected_output = math::stack_as_xv2f(field, field);
	math::convolve_with_kernel_preserve_zeros(vector_field, kernel);
	BOOST_REQUIRE(math::almost_equal(vector_field, vector_field, 1e-10));
}

BOOST_AUTO_TEST_CASE(convolution_test02) {
	math::MatrixXv2f vector_field(4, 4);
	vector_field <<
			math::Vector2f(0.f, 0.f),
			math::Vector2f(0.f, 0.f),
			math::Vector2f(-0.35937524f, -0.18750024f),
			math::Vector2f(-0.13125f, -0.17500037f),

			math::Vector2f(0.f, 0.f),
			math::Vector2f(-0.4062504f, -0.4062496f),
			math::Vector2f(-0.09375f, -0.1874992f),
			math::Vector2f(-0.04375001f, -0.17499907f),

			math::Vector2f(0.f, 0.f),
			math::Vector2f(-0.65624946f, -0.21874908f),
			math::Vector2f(-0.09375f, -0.1499992f),
			math::Vector2f(-0.04375001f, -0.21874908f),

			math::Vector2f(0.f, 0.f),
			math::Vector2f(-0.5312497f, -0.18750025f),
			math::Vector2f(-0.09374999f, -0.15000032f),
			math::Vector2f(-0.13125001f, -0.2625004f);

	eig::VectorXf kernel(3);
	kernel << 0.06742075, 0.99544406, 0.06742075;
	math::MatrixXv2f expected_output(4, 4);
	expected_output <<

	math::Vector2f(0.0f, 0.0f),
			math::Vector2f(0.0f, 0.0f),
			math::Vector2f(-0.37140754f, -0.21091977f),
			math::Vector2f(-0.1575381f, -0.19859035f),

			math::Vector2f(0.0f, 0.0f),
			math::Vector2f(-0.45495197f, -0.43135524f),
			math::Vector2f(-0.1572882f, -0.25023922f),
			math::Vector2f(-0.06344876f, -0.21395193f),

			math::Vector2f(0.0f, 0.0f),
			math::Vector2f(-0.7203466f, -0.2682102f),
			math::Vector2f(-0.15751791f, -0.20533603f),
			math::Vector2f(-0.06224134f, -0.2577237f),

			math::Vector2f(0.0f, 0.0f),
			math::Vector2f(-0.57718134f, -0.2112256f),
			math::Vector2f(-0.14683421f, -0.19089346f),
			math::Vector2f(-0.13971105f, -0.2855439f);

	math::convolve_with_kernel_preserve_zeros(vector_field, kernel);
	BOOST_REQUIRE(math::almost_equal(vector_field, expected_output, 1e-6));
}

BOOST_AUTO_TEST_CASE(convolution_test03) {
	math::MatrixXv2f vector_field(4, 4);
	vector_field <<
			math::Vector2f(0.f, 0.f),
			math::Vector2f(0.f, 0.f),
			math::Vector2f(-0.35937524f, -0.18750024f),
			math::Vector2f(-0.13125f, -0.17500037f),

			math::Vector2f(0.f, 0.f),
			math::Vector2f(-0.4062504f, -0.4062496f),
			math::Vector2f(-0.09375f, -0.1874992f),
			math::Vector2f(-0.04375001f, -0.17499907f),

			math::Vector2f(0.f, 0.f),
			math::Vector2f(-0.65624946f, -0.21874908f),
			math::Vector2f(-0.09375f, -0.1499992f),
			math::Vector2f(-0.04375001f, -0.21874908f),

			math::Vector2f(0.f, 0.f),
			math::Vector2f(-0.5312497f, -0.18750025f),
			math::Vector2f(-0.09374999f, -0.15000032f),
			math::Vector2f(-0.13125001f, -0.2625004f);

	eig::VectorXf kernel(3);
	kernel << 0.06742075, 0.99544406, 0.06742075;
	math::MatrixXv2f expected_output(4, 4);
	expected_output <<

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

	math::convolve_with_kernel_y(vector_field, kernel);

	BOOST_REQUIRE(math::almost_equal_verbose(vector_field, expected_output, 1e-6));
}

BOOST_AUTO_TEST_CASE(convolution_test04) {
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

	eig::VectorXf kernel(3);
	kernel << 0.06742075, 0.99544406, 0.06742075;
	math::MatrixXv2f expected_output(4, 4);
	expected_output <<

	math::Vector2f(-0.00184663f, -0.00184663f),
			math::Vector2f(-0.05181003f, -0.04070096f),
			math::Vector2f(-0.37325418f, -0.2127664f),
			math::Vector2f(-0.1575381f, -0.19859035f),

			math::Vector2f(-0.03024794f, -0.0282592f),
			math::Vector2f(-0.45495197f, -0.43135524f),
			math::Vector2f(-0.1572882f, -0.25023922f),
			math::Vector2f(-0.06344876f, -0.21395193f),

			math::Vector2f(-0.04830472f, -0.01737996f),
			math::Vector2f(-0.7203466f, -0.2682102f),
			math::Vector2f(-0.15751792f, -0.20533603f),
			math::Vector2f(-0.06224134f, -0.2577237f),

			math::Vector2f(-0.03863709f, -0.01357815f),
			math::Vector2f(-0.57718134f, -0.2112256f),
			math::Vector2f(-0.14683422f, -0.19089346f),
			math::Vector2f(-0.13971105f, -0.2855439f);

	math::convolve_with_kernel_x(vector_field, kernel);

	BOOST_REQUIRE(math::almost_equal(vector_field, expected_output, 1e-6));
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

BOOST_AUTO_TEST_CASE(test_upsampling_linear_matrix01) {
	eig::MatrixXf input(2, 3);
	input << 1.0f, 2.0f, 3.0f,
			4.0f, 5.0f, 6.0f;

	eig::MatrixXf expected_output(4, 6);
	expected_output <<
			1.00f, 1.25f, 1.75f, 2.25f, 2.75f, 3.00f,
			1.75f, 2.00f, 2.50f, 3.00f, 3.50f, 3.75f,
			3.25f, 3.50f, 4.00f, 4.50f, 5.00f, 5.25f,
			4.00f, 4.25f, 4.75f, 5.25f, 5.75f, 6.00f;
	eig::MatrixXf output = math::upsampleX2_linear(input);
	BOOST_REQUIRE(math::almost_equal_verbose(output, expected_output, 1e-6));

	eig::MatrixXf input2(3, 2);
	input2 <<
			1.0f, 2.0f,
			3.0f, 4.0f,
			5.0f, 6.0f;

	eig::MatrixXf expected_output2(6, 4);
	expected_output2 <<
			1.00f, 1.25f, 1.75f, 2.00f,
			1.50f, 1.75f, 2.25f, 2.50f,
			2.50f, 2.75f, 3.25f, 3.50f,
			3.50f, 3.75f, 4.25f, 4.50f,
			4.50f, 4.75f, 5.25f, 5.50f,
			5.00f, 5.25f, 5.75f, 6.00f;

	eig::MatrixXf output2 = math::upsampleX2_linear(input2);
	BOOST_REQUIRE(math::almost_equal_verbose(output2, expected_output2, 1e-6));

	eig::MatrixXf input3(3, 3);
	input3 <<
			1.0f, 2.0f, 3.0f,
			4.0f, 5.0f, 6.0f,
			7.0f, 8.0f, 9.0f;
	eig::MatrixXf expected_output3(6, 6);
	expected_output3 <<
			1.00f, 1.25f, 1.75f, 2.25f, 2.75f, 3.00f,
			1.75f, 2.00f, 2.50f, 3.00f, 3.50f, 3.75f,
			3.25f, 3.50f, 4.00f, 4.50f, 5.00f, 5.25f,
			4.75f, 5.00f, 5.50f, 6.00f, 6.50f, 6.75f,
			6.25f, 6.50f, 7.00f, 7.50f, 8.00f, 8.25f,
			7.00f, 7.25f, 7.75f, 8.25f, 8.75f, 9.00f;
	eig::MatrixXf output3 = math::upsampleX2_linear(input3);
	BOOST_REQUIRE(math::almost_equal_verbose(output3, expected_output3, 1e-6));
}

BOOST_AUTO_TEST_CASE(test_upsampling_nearest_tensor01) {
	math::Tensor3f input(2, 2, 2);
	input.setValues( //@formatter:off
		{{{ 1.0f, 2.0f },
		  { 3.0f, 4.0f }},
		 {{ 5.0f, 6.0f },
		  { 7.0f, 8.0f }}}
	);   //@formatter:on

	math::Tensor3f expected_output(4, 4, 4);
	expected_output.setValues( //@formatter:off
		{{{1.0f, 1.0f, 2.0f, 2.0f},
		  {1.0f, 1.0f, 2.0f, 2.0f},
		  {3.0f, 3.0f, 4.0f, 4.0f},
		  {3.0f, 3.0f, 4.0f, 4.0f}},
		 {{1.0f, 1.0f, 2.0f, 2.0f},
		  {1.0f, 1.0f, 2.0f, 2.0f},
		  {3.0f, 3.0f, 4.0f, 4.0f},
		  {3.0f, 3.0f, 4.0f, 4.0f}},
		 {{5.0f, 5.0f, 6.0f, 6.0f},
		  {5.0f, 5.0f, 6.0f, 6.0f},
		  {7.0f, 7.0f, 8.0f, 8.0f},
		  {7.0f, 7.0f, 8.0f, 8.0f}},
		 {{5.0f, 5.0f, 6.0f, 6.0f},
		  {5.0f, 5.0f, 6.0f, 6.0f},
		  {7.0f, 7.0f, 8.0f, 8.0f},
		  {7.0f, 7.0f, 8.0f, 8.0f}}}
	);   //@formatter:on

	math::Tensor3f output = math::upsampleX2_nearest(input);
	BOOST_REQUIRE(math::almost_equal_verbose(output, expected_output, 1e-6));
}

BOOST_AUTO_TEST_CASE(test_upsampling_linear_tensor01) {
	math::Tensor3f input(2, 2, 2);
	input.setValues( //@formatter:off
		{{{ 1.0f, 5.0f },
		  { 3.0f, 7.0f }},
		 {{ 2.0f, 6.0f },
		  { 4.0f, 8.0f }}}
	);   //@formatter:on

	math::Tensor3f expected_output(4, 4, 4);
	expected_output.setValues( //@formatter:off
		{{{1.00f, 2.00f, 4.00f, 5.00f},
		  {1.50f, 2.50f, 4.50f, 5.50f},
		  {2.50f, 3.50f, 5.50f, 6.50f},
		  {3.00f, 4.00f, 6.00f, 7.00f}},
		 {{1.25f, 2.25f, 4.25f, 5.25f},
		  {1.75f, 2.75f, 4.75f, 5.75f},
		  {2.75f, 3.75f, 5.75f, 6.75f},
		  {3.25f, 4.25f, 6.25f, 7.25f}},
		 {{1.75f, 2.75f, 4.75f, 5.75f},
		  {2.25f, 3.25f, 5.25f, 6.25f},
		  {3.25f, 4.25f, 6.25f, 7.25f},
		  {3.75f, 4.75f, 6.75f, 7.75f}},
		 {{2.00f, 3.00f, 5.00f, 6.00f},
		  {2.50f, 3.50f, 5.50f, 6.50f},
		  {3.50f, 4.50f, 6.50f, 7.50f},
		  {4.00f, 5.00f, 7.00f, 8.00f}}}
	);   //@formatter:on

	math::Tensor3f output = math::upsampleX2_linear(input);
	BOOST_REQUIRE(math::almost_equal_verbose(output, expected_output, 1e-6));
	math::Tensor3f input2(3, 3, 3);
	input2.setValues( //@formatter:off
			{{{1.0f, 10.0f, 19.0f},
			  {4.0f, 13.0f, 22.0f},
			  {7.0f, 16.0f, 25.0f}},
			 {{2.0f, 11.0f, 20.0f},
			  {5.0f, 14.0f, 23.0f},
			  {8.0f, 17.0f, 26.0f}},
			 {{3.0f, 12.0f, 21.0f},
			  {6.0f, 15.0f, 24.0f},
			  {9.0f, 18.0f, 27.0f}}}
	);  //@formatter:on

	math::Tensor3f expected_output2(6, 6, 6);
	expected_output2.setValues( //@formatter:off
			{{{1.00f, 3.25f, 7.75f, 12.25f, 16.75f, 19.00f},
			  {1.75f, 4.00f, 8.50f, 13.00f, 17.50f, 19.75f},
			  {3.25f, 5.50f, 10.00f, 14.50f, 19.00f, 21.25f},
			  {4.75f, 7.00f, 11.50f, 16.00f, 20.50f, 22.75f},
			  {6.25f, 8.50f, 13.00f, 17.50f, 22.00f, 24.25f},
			  {7.00f, 9.25f, 13.75f, 18.25f, 22.75f, 25.00f}},
			 {{1.25f, 3.50f, 8.00f, 12.50f, 17.00f, 19.25f},
			  {2.00f, 4.25f, 8.75f, 13.25f, 17.75f, 20.00f},
			  {3.50f, 5.75f, 10.25f, 14.75f, 19.25f, 21.50f},
			  {5.00f, 7.25f, 11.75f, 16.25f, 20.75f, 23.00f},
			  {6.50f, 8.75f, 13.25f, 17.75f, 22.25f, 24.50f},
			  {7.25f, 9.50f, 14.00f, 18.50f, 23.00f, 25.25f}},
			 {{1.75f, 4.00f, 8.50f, 13.00f, 17.50f, 19.75f},
			  {2.50f, 4.75f, 9.25f, 13.75f, 18.25f, 20.50f},
			  {4.00f, 6.25f, 10.75f, 15.25f, 19.75f, 22.00f},
			  {5.50f, 7.75f, 12.25f, 16.75f, 21.25f, 23.50f},
			  {7.00f, 9.25f, 13.75f, 18.25f, 22.75f, 25.00f},
			  {7.75f, 10.00f, 14.50f, 19.00f, 23.50f, 25.75f}},
			 {{2.25f, 4.50f, 9.00f, 13.50f, 18.00f, 20.25f},
			  {3.00f, 5.25f, 9.75f, 14.25f, 18.75f, 21.00f},
			  {4.50f, 6.75f, 11.25f, 15.75f, 20.25f, 22.50f},
			  {6.00f, 8.25f, 12.75f, 17.25f, 21.75f, 24.00f},
			  {7.50f, 9.75f, 14.25f, 18.75f, 23.25f, 25.50f},
			  {8.25f, 10.50f, 15.00f, 19.50f, 24.00f, 26.25f}},
			 {{2.75f, 5.00f, 9.50f, 14.00f, 18.50f, 20.75f},
			  {3.50f, 5.75f, 10.25f, 14.75f, 19.25f, 21.50f},
			  {5.00f, 7.25f, 11.75f, 16.25f, 20.75f, 23.00f},
			  {6.50f, 8.75f, 13.25f, 17.75f, 22.25f, 24.50f},
			  {8.00f, 10.25f, 14.75f, 19.25f, 23.75f, 26.00f},
			  {8.75f, 11.00f, 15.50f, 20.00f, 24.50f, 26.75f}},
			 {{3.00f, 5.25f, 9.75f, 14.25f, 18.75f, 21.00f},
			  {3.75f, 6.00f, 10.50f, 15.00f, 19.50f, 21.75f},
			  {5.25f, 7.50f, 12.00f, 16.50f, 21.00f, 23.25f},
			  {6.75f, 9.00f, 13.50f, 18.00f, 22.50f, 24.75f},
			  {8.25f, 10.50f, 15.00f, 19.50f, 24.00f, 26.25f},
			  {9.00f, 11.25f, 15.75f, 20.25f, 24.75f, 27.00f}}}
	);  //@formatter:on
	math::Tensor3f output2 = math::upsampleX2_linear(input2);
	BOOST_REQUIRE(math::almost_equal_verbose(output2, expected_output2, 1e-6));
}
BOOST_AUTO_TEST_CASE(test_upsampling_linear_tensor02){
	math::Tensor3f input(4,3,3);
	input.setValues(//@formatter:off
			{{{1.0f,1.0f,1.0f},
			  {1.0f,1.0f,1.0f},
			  {1.0f,1.0f,1.0f}},
			 {{2.0f,2.0f,2.0f},
			  {2.0f,2.0f,2.0f},
			  {2.0f,2.0f,2.0f}},
			 {{3.0f,3.0f,3.0f},
			  {3.0f,3.0f,3.0f},
			  {3.0f,3.0f,3.0f}},
			 {{4.0f,4.0f,4.0f},
			  {4.0f,4.0f,4.0f},
			  {4.0f,4.0f,4.0f}}
	});//@formatter:on

	math::Tensor3f expected_output(8,6,6);
	expected_output.setValues(//@formatter:off
			{{{1.00f, 1.00f, 1.00f, 1.00f, 1.00f, 1.00f},
			  {1.00f, 1.00f, 1.00f, 1.00f, 1.00f, 1.00f},
			  {1.00f, 1.00f, 1.00f, 1.00f, 1.00f, 1.00f},
			  {1.00f, 1.00f, 1.00f, 1.00f, 1.00f, 1.00f},
			  {1.00f, 1.00f, 1.00f, 1.00f, 1.00f, 1.00f},
			  {1.00f, 1.00f, 1.00f, 1.00f, 1.00f, 1.00f}},
			 {{1.25f, 1.25f, 1.25f, 1.25f, 1.25f, 1.25f},
			  {1.25f, 1.25f, 1.25f, 1.25f, 1.25f, 1.25f},
			  {1.25f, 1.25f, 1.25f, 1.25f, 1.25f, 1.25f},
			  {1.25f, 1.25f, 1.25f, 1.25f, 1.25f, 1.25f},
			  {1.25f, 1.25f, 1.25f, 1.25f, 1.25f, 1.25f},
			  {1.25f, 1.25f, 1.25f, 1.25f, 1.25f, 1.25f}},
			 {{1.75f, 1.75f, 1.75f, 1.75f, 1.75f, 1.75f},
			  {1.75f, 1.75f, 1.75f, 1.75f, 1.75f, 1.75f},
			  {1.75f, 1.75f, 1.75f, 1.75f, 1.75f, 1.75f},
			  {1.75f, 1.75f, 1.75f, 1.75f, 1.75f, 1.75f},
			  {1.75f, 1.75f, 1.75f, 1.75f, 1.75f, 1.75f},
			  {1.75f, 1.75f, 1.75f, 1.75f, 1.75f, 1.75f}},
			 {{2.25f, 2.25f, 2.25f, 2.25f, 2.25f, 2.25f},
			  {2.25f, 2.25f, 2.25f, 2.25f, 2.25f, 2.25f},
			  {2.25f, 2.25f, 2.25f, 2.25f, 2.25f, 2.25f},
			  {2.25f, 2.25f, 2.25f, 2.25f, 2.25f, 2.25f},
			  {2.25f, 2.25f, 2.25f, 2.25f, 2.25f, 2.25f},
			  {2.25f, 2.25f, 2.25f, 2.25f, 2.25f, 2.25f}},
			 {{2.75f, 2.75f, 2.75f, 2.75f, 2.75f, 2.75f},
			  {2.75f, 2.75f, 2.75f, 2.75f, 2.75f, 2.75f},
			  {2.75f, 2.75f, 2.75f, 2.75f, 2.75f, 2.75f},
			  {2.75f, 2.75f, 2.75f, 2.75f, 2.75f, 2.75f},
			  {2.75f, 2.75f, 2.75f, 2.75f, 2.75f, 2.75f},
			  {2.75f, 2.75f, 2.75f, 2.75f, 2.75f, 2.75f}},
			 {{3.25f, 3.25f, 3.25f, 3.25f, 3.25f, 3.25f},
			  {3.25f, 3.25f, 3.25f, 3.25f, 3.25f, 3.25f},
			  {3.25f, 3.25f, 3.25f, 3.25f, 3.25f, 3.25f},
			  {3.25f, 3.25f, 3.25f, 3.25f, 3.25f, 3.25f},
			  {3.25f, 3.25f, 3.25f, 3.25f, 3.25f, 3.25f},
			  {3.25f, 3.25f, 3.25f, 3.25f, 3.25f, 3.25f}},
			 {{3.75f, 3.75f, 3.75f, 3.75f, 3.75f, 3.75f},
			  {3.75f, 3.75f, 3.75f, 3.75f, 3.75f, 3.75f},
			  {3.75f, 3.75f, 3.75f, 3.75f, 3.75f, 3.75f},
			  {3.75f, 3.75f, 3.75f, 3.75f, 3.75f, 3.75f},
			  {3.75f, 3.75f, 3.75f, 3.75f, 3.75f, 3.75f},
			  {3.75f, 3.75f, 3.75f, 3.75f, 3.75f, 3.75f}},
			 {{4.00f, 4.00f, 4.00f, 4.00f, 4.00f, 4.00f},
			  {4.00f, 4.00f, 4.00f, 4.00f, 4.00f, 4.00f},
			  {4.00f, 4.00f, 4.00f, 4.00f, 4.00f, 4.00f},
			  {4.00f, 4.00f, 4.00f, 4.00f, 4.00f, 4.00f},
			  {4.00f, 4.00f, 4.00f, 4.00f, 4.00f, 4.00f},
			  {4.00f, 4.00f, 4.00f, 4.00f, 4.00f, 4.00f}}}
	);//@formatter:on

	math::Tensor3f output = math::upsampleX2_linear(input);
	BOOST_REQUIRE(math::almost_equal_verbose(output, expected_output, 1e-6));
}


BOOST_AUTO_TEST_CASE(test_downsampling_linear_matrix01) {
	eig::MatrixXf input(6, 4);
	input <<
			0.0f, 1.0f, 2.0f, 3.0f,
			4.0f, 5.0f, 6.0f, 7.0f,
			8.0f, 9.0f, 10.0f, 11.0f,
			12.0f, 13.0f, 14.0f, 15.0f,
			16.0f, 17.0f, 18.0f, 19.0f,
			20.0f, 21.0f, 22.0f, 23.0f;
	eig::MatrixXf expected_output(3, 2);
	expected_output <<
			3.125f, 4.875f,
			10.625f, 12.375f,
			18.125f, 19.875f;
	eig::MatrixXf output = math::downsampleX2_linear(input);
	BOOST_REQUIRE(math::almost_equal_verbose(output, expected_output, 1e-6));

	eig::MatrixXf input2(8, 8);
	input2 <<
			0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f,
			8.0f, 9.0f, 10.0f, 11.0f, 12.0f, 13.0f, 14.0f, 15.0f,
			16.0f, 17.0f, 18.0f, 19.0f, 20.0f, 21.0f, 22.0f, 23.0f,
			24.0f, 25.0f, 26.0f, 27.0f, 28.0f, 29.0f, 30.0f, 31.0f,
			32.0f, 33.0f, 34.0f, 35.0f, 36.0f, 37.0f, 38.0f, 39.0f,
			40.0f, 41.0f, 42.0f, 43.0f, 44.0f, 45.0f, 46.0f, 47.0f,
			48.0f, 49.0f, 50.0f, 51.0f, 52.0f, 53.0f, 54.0f, 55.0f,
			56.0f, 57.0f, 58.0f, 59.0f, 60.0f, 61.0f, 62.0f, 63.0f;
	eig::MatrixXf expected_output2(4, 4);
	// NB: output is different from OpenCV at the border because OpenCV resize w/ INTER_LINEAR simply averages
	// each coarse border value from the 4 corresponding fine values, whereas this code treats the border values
	// as repeating and applies the same set of coefficients at the borders as any other place.
	expected_output2 <<
			5.625f, 7.5f, 9.5f, 11.375f,
			20.625f, 22.5f, 24.5f, 26.375f,
			36.625f, 38.5f, 40.5f, 42.375f,
			51.625f, 53.5f, 55.5f, 57.375f;

	eig::MatrixXf output2 = math::downsampleX2_linear(input2);
	BOOST_REQUIRE(math::almost_equal_verbose(output2, expected_output2, 1e-6));
}

BOOST_AUTO_TEST_CASE(test_downsampling_linear_tensor01) {
	math::Tensor3f input(4, 4, 4);
	input.setValues(	//@formatter:off
		{{{1.0000, 2.0000, 4.0000, 5.0000},
		  {1.5000, 2.5000, 4.5000, 5.5000},
		  {2.5000, 3.5000, 5.5000, 6.5000},
		  {3.0000, 4.0000, 6.0000, 7.0000}},
		 {{1.2500, 2.2500, 4.2500, 5.2500},
		  {1.7500, 2.7500, 4.7500, 5.7500},
		  {2.7500, 3.7500, 5.7500, 6.7500},
		  {3.2500, 4.2500, 6.2500, 7.2500}},
		 {{1.7500, 2.7500, 4.7500, 5.7500},
		  {2.2500, 3.2500, 5.2500, 6.2500},
		  {3.2500, 4.2500, 6.2500, 7.2500},
		  {3.7500, 4.7500, 6.7500, 7.7500}},
		 {{2.0000, 3.0000, 5.0000, 6.0000},
		  {2.5000, 3.5000, 5.5000, 6.5000},
		  {3.5000, 4.5000, 6.5000, 7.5000},
		  {4.0000, 5.0000, 7.0000, 8.0000}}}
	); 		//@formatter:on
	math::Tensor3f expected_output(2,2,2);
	expected_output.setValues(//@formatter:off
			{{{2.3125f, 4.8125f},
			  {3.5625f, 6.0625f}},
			 {{2.9375f, 5.4375f},
			  {4.1875f, 6.6875f}}}
	);//@formatter:on
	math::Tensor3f output = math::downsampleX2_linear(input);
	BOOST_REQUIRE(math::almost_equal_verbose(output, expected_output, 1e-6));

	math::Tensor3f input2(6, 6, 6);
	input2.setValues( //@formatter:off
			{{{1.00f, 3.25f, 7.75f, 12.25f, 16.75f, 19.00f},
			  {1.75f, 4.00f, 8.50f, 13.00f, 17.50f, 19.75f},
			  {3.25f, 5.50f, 10.00f, 14.50f, 19.00f, 21.25f},
			  {4.75f, 7.00f, 11.50f, 16.00f, 20.50f, 22.75f},
			  {6.25f, 8.50f, 13.00f, 17.50f, 22.00f, 24.25f},
			  {7.00f, 9.25f, 13.75f, 18.25f, 22.75f, 25.00f}},
			 {{1.25f, 3.50f, 8.00f, 12.50f, 17.00f, 19.25f},
			  {2.00f, 4.25f, 8.75f, 13.25f, 17.75f, 20.00f},
			  {3.50f, 5.75f, 10.25f, 14.75f, 19.25f, 21.50f},
			  {5.00f, 7.25f, 11.75f, 16.25f, 20.75f, 23.00f},
			  {6.50f, 8.75f, 13.25f, 17.75f, 22.25f, 24.50f},
			  {7.25f, 9.50f, 14.00f, 18.50f, 23.00f, 25.25f}},
			 {{1.75f, 4.00f, 8.50f, 13.00f, 17.50f, 19.75f},
			  {2.50f, 4.75f, 9.25f, 13.75f, 18.25f, 20.50f},
			  {4.00f, 6.25f, 10.75f, 15.25f, 19.75f, 22.00f},
			  {5.50f, 7.75f, 12.25f, 16.75f, 21.25f, 23.50f},
			  {7.00f, 9.25f, 13.75f, 18.25f, 22.75f, 25.00f},
			  {7.75f, 10.00f, 14.50f, 19.00f, 23.50f, 25.75f}},
			 {{2.25f, 4.50f, 9.00f, 13.50f, 18.00f, 20.25f},
			  {3.00f, 5.25f, 9.75f, 14.25f, 18.75f, 21.00f},
			  {4.50f, 6.75f, 11.25f, 15.75f, 20.25f, 22.50f},
			  {6.00f, 8.25f, 12.75f, 17.25f, 21.75f, 24.00f},
			  {7.50f, 9.75f, 14.25f, 18.75f, 23.25f, 25.50f},
			  {8.25f, 10.50f, 15.00f, 19.50f, 24.00f, 26.25f}},
			 {{2.75f, 5.00f, 9.50f, 14.00f, 18.50f, 20.75f},
			  {3.50f, 5.75f, 10.25f, 14.75f, 19.25f, 21.50f},
			  {5.00f, 7.25f, 11.75f, 16.25f, 20.75f, 23.00f},
			  {6.50f, 8.75f, 13.25f, 17.75f, 22.25f, 24.50f},
			  {8.00f, 10.25f, 14.75f, 19.25f, 23.75f, 26.00f},
			  {8.75f, 11.00f, 15.50f, 20.00f, 24.50f, 26.75f}},
			 {{3.00f, 5.25f, 9.75f, 14.25f, 18.75f, 21.00f},
			  {3.75f, 6.00f, 10.50f, 15.00f, 19.50f, 21.75f},
			  {5.25f, 7.50f, 12.00f, 16.50f, 21.00f, 23.25f},
			  {6.75f, 9.00f, 13.50f, 18.00f, 22.50f, 24.75f},
			  {8.25f, 10.50f, 15.00f, 19.50f, 24.00f, 26.25f},
			  {9.00f, 11.25f, 15.75f, 20.25f, 24.75f, 27.00f}}}
	);  //@formatter:on
	math::Tensor3f expected_output2(3,3,3);
	expected_output2.setValues(//@formatter:off
		{{{3.4375f, 10.7500f, 18.0625f},
		  {5.8750f, 13.1875f, 20.5000f},
		  {8.3125f, 15.6250f, 22.9375f}},
		 {{4.2500f, 11.5625f, 18.8750f},
		  {6.6875f, 14.0000f, 21.3125f},
		  {9.1250f, 16.4375f, 23.7500f}},
		 {{5.0625f, 12.3750f, 19.6875f},
		  {7.5000f, 14.8125f, 22.1250f},
		  {9.9375f, 17.2500f, 24.5625f}}}
	);//@formatter:on
	math::Tensor3f output2 = math::downsampleX2_linear(input2);
	BOOST_REQUIRE(math::almost_equal_verbose(output2, expected_output2, 1e-6));

	math::Tensor3f input3_up(4,3,3);
	input3_up.setValues(//@formatter:off
			{{{1.0f,1.0f,1.0f},
			  {1.0f,1.0f,1.0f},
			  {1.0f,1.0f,1.0f}},
			 {{2.0f,2.0f,2.0f},
			  {2.0f,2.0f,2.0f},
			  {2.0f,2.0f,2.0f}},
			 {{3.0f,3.0f,3.0f},
			  {3.0f,3.0f,3.0f},
			  {3.0f,3.0f,3.0f}},
			 {{4.0f,4.0f,4.0f},
			  {4.0f,4.0f,4.0f},
			  {4.0f,4.0f,4.0f}}
	});//@formatter:on
	math::Tensor3f input3_down = math::upsampleX2_linear(input3_up);
	math::Tensor3f expected_output3(4,3,3);
	expected_output3.setValues(//@formatter:off
			{{{1.1875f, 1.1875f, 1.1875f},
			  {1.1875f, 1.1875f, 1.1875f},
			  {1.1875f, 1.1875f, 1.1875f}},
			 {{2.0000f, 2.0000f, 2.0000f},
			  {2.0000f, 2.0000f, 2.0000f},
			  {2.0000f, 2.0000f, 2.0000f}},
			 {{3.0000f, 3.0000f, 3.0000f},
			  {3.0000f, 3.0000f, 3.0000f},
			  {3.0000f, 3.0000f, 3.0000f}},
			 {{3.8125f, 3.8125f, 3.8125f},
			  {3.8125f, 3.8125f, 3.8125f},
			  {3.8125f, 3.8125f, 3.8125f}}}
	);//@formatter:on
	math::Tensor3f output3 = math::downsampleX2_linear(input3_down);
	BOOST_REQUIRE(math::almost_equal_verbose(output3, expected_output3, 1e-6));
}
