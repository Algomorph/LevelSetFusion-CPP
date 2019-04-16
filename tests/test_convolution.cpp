/*
 * test_convolution.cpp
 *
 *  Created on: Apr 16, 2019
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

#define BOOST_TEST_MODULE test_convolution

//libraries
#include <boost/test/unit_test.hpp>
#include <boost/python.hpp>
#include <Eigen/Eigen>

//test targets
#include "../src/math/convolution.hpp"
#include "../src/math/almost_equal.hpp"
#include "../src/math/typedefs.hpp"


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


