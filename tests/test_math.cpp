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
#include "../src/math/almost_equal.hpp"
#include "../src/math/typedefs.hpp"
#include "../src/math/statistics.hpp"
#include "../src/math/padding.hpp"
#include "../src/math/cwise_binary.hpp"
#include "../src/math/cwise_unary.hpp"
#include "../src/math/transformation.hpp"

namespace eig = Eigen;


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
	);        // @formatter:on

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
			  { 9.0f,  9.0f, 18.0f, 27.0f, 27.0f}}});        // @formatter:on

	math::Tensor3f output = math::pad_replicate(input, 1);

	BOOST_REQUIRE(math::almost_equal_verbose(output, expected_output, 1e-6));
}

BOOST_AUTO_TEST_CASE(test_scale_matrix) {
	eig::MatrixXf a(2, 2);
	a << 1.0f, 2.0f, 3.0f, 4.0f;
	eig::MatrixXf a_scaled = math::scale(a, 2.0f);
	eig::MatrixXf expected_output(2, 2);
	expected_output << 2.0f, 4.0f, 6.0f, 8.0f;
	BOOST_REQUIRE(math::almost_equal_verbose(a_scaled, expected_output, 1e-6));

	eig::MatrixXd b(1, 2);
	b << 1.0, 2.0;
	eig::MatrixXd b_scaled = math::scale(b, 2.0);
	eig::MatrixXd expected_output2(1, 2);
	expected_output2 << 2.0, 4.0;
	BOOST_REQUIRE(math::almost_equal_verbose(b_scaled, expected_output2, 1e-6));

	math::MatrixXv2f c(1, 2);
	c << math::Vector2f(1.0f, 2.0f), math::Vector2f(3.0f, 4.0f);
	math::MatrixXv2f c_scaled = math::scale(c, -2.0f);
	math::MatrixXv2f expected_output3(1, 2);
	expected_output3 << math::Vector2f(-2.0f, -4.0f), math::Vector2f(-6.0f, -8.0f);
	BOOST_REQUIRE(math::almost_equal_verbose(c_scaled, expected_output3, 1e-6));
}

BOOST_AUTO_TEST_CASE(test_scale_tensor) {
	math::Tensor3f a(1, 2, 2);
	a.setValues( { { { 1.0f, 2.0f },
			{ 3.0f, 4.0f } } });
	math::Tensor3f a_scaled = math::scale(a, 3.0f);
	math::Tensor3f expected_output(1, 2, 2);
	expected_output.setValues(
			{ { { 3.0f, 6.0f },
					{ 9.0f, 12.0f } } }
			);
	BOOST_REQUIRE(math::almost_equal_verbose(a_scaled, expected_output, 1e-6));

	math::Tensor3v3f b(1, 2, 2);
	b.setValues( { { { math::Vector3f(1.0f, 2.0f, 3.0f), math::Vector3f(4.0f, 5.0f, 6.0f) },
			{ math::Vector3f(7.0f, 8.0f, 9.0f), math::Vector3f(10.0f, 11.0f, 12.0f) } } });
	math::Tensor3v3f b_scaled = math::scale(b, -0.5f);
	math::Tensor3v3f expected_output2(1, 2, 2);
	expected_output2.setValues( { { { math::Vector3f(-0.5f, -1.0f, -1.5f), math::Vector3f(-2.0f, -2.5f, -3.0f) },
			{ math::Vector3f(-3.5f, -4.0f, -4.5f), math::Vector3f(-5.0f, -5.5f, -6.0f) } } });
	BOOST_REQUIRE(math::almost_equal_verbose(b_scaled, expected_output2, 1e-6));
}

BOOST_AUTO_TEST_CASE(test_cwise_product_matrix) {
	math::MatrixXv2f a(1, 2);
	a << math::Vector2f(1.0f, 2.0f), math::Vector2f(8.0f, 4.0f);
	eig::MatrixXf b(1, 2);
	b << 0.5, -0.25f;
	math::MatrixXv2f c = math::cwise_product(a, b);
	math::MatrixXv2f expected_c(1, 2);
	expected_c << math::Vector2f(0.5f, 1.0f), math::Vector2f(-2.0f, -1.0f);
	BOOST_REQUIRE(math::almost_equal_verbose(c, expected_c, 1e-6));
}

BOOST_AUTO_TEST_CASE(test_cwise_product_tensor) {
	math::Tensor3v3f a(1, 2, 2);
	a.setValues( { { { math::Vector3f(1.0f, 2.0f, 3.0f), math::Vector3f(4.0f, 5.0f, 6.0f) },
			{ math::Vector3f(7.0f, 8.0f, 9.0f), math::Vector3f(10.0f, 11.0f, 12.0f) } } });
	math::Tensor3f b(1, 2, 2);
	b.setValues( { { { 2.0f, 0.5f },
			{ -0.5f, 1.0f } } });
	math::Tensor3v3f c = math::cwise_product(a, b);
	math::Tensor3v3f expected_c(1, 2, 2);
	expected_c.setValues( { { { math::Vector3f(2.0f, 4.0f, 6.0f), math::Vector3f(2.0f, 2.5f, 3.0f) },
			{ math::Vector3f(-3.5f, -4.0f, -4.5f), math::Vector3f(10.0f, 11.0f, 12.0f) } } });
	BOOST_REQUIRE(math::almost_equal_verbose(c, expected_c, 1e-6));
}

BOOST_AUTO_TEST_CASE(test_cwise_add_constant_matrix) {
	math::MatrixXv2f a(1, 2);
	a << math::Vector2f(1.0f, 2.0f), math::Vector2f(3.0f, 4.0f);
	math::MatrixXv2f a_plus = math::cwise_add_constant(a, math::Vector2f(1.0f, 2.0f));
	math::MatrixXv2f expected_output(1, 2);
	expected_output << math::Vector2f(2.0f, 4.0f), math::Vector2f(4.0f, 6.0f);
	BOOST_REQUIRE(math::almost_equal_verbose(a_plus, expected_output, 1e-6));

	eig::MatrixXf b(2, 2);
	b << 1.0f, 2.0f, 3.0f, 4.0f;
	eig::MatrixXf b_minus_one = math::cwise_add_constant(b, -1.0f);
	eig::MatrixXf expected_output2(2, 2);
	expected_output2 << 0.0f, 1.0f, 2.0f, 3.0f;
	BOOST_REQUIRE(math::almost_equal_verbose(b_minus_one, expected_output2, 1e-6));
}

BOOST_AUTO_TEST_CASE(test_cwise_add_constant_tensor) {
	math::Tensor3f a(1, 2, 2);
	a.setValues( { { { 1.0f, 2.0f },
			{ 3.0f, 4.0f } } });
	math::Tensor3f a_plus = math::cwise_add_constant(a, 1.5f);
	math::Tensor3f expected_a_plus(1,2,2);
	expected_a_plus.setValues( { { { 2.5f, 3.5f },
		{ 4.5f, 5.5f } } });
	BOOST_REQUIRE(math::almost_equal_verbose(a_plus, expected_a_plus, 1e-6));

	math::Tensor3v3f b(1, 2, 2);
	b.setValues( { { { math::Vector3f(1.0f, 2.0f, 3.0f), math::Vector3f(4.0f, 5.0f, 6.0f) },
			{ math::Vector3f(7.0f, 8.0f, 9.0f), math::Vector3f(10.0f, 11.0f, 12.0f) } } });
	math::Tensor3v3f b_plus = math::cwise_add_constant(b, math::Vector3f(2.0f, 1.0f, -1.0f));
	math::Tensor3v3f expected_b_plus(1,2,2);
	expected_b_plus.setValues( { { { math::Vector3f(3.0f, 3.0f, 2.0f), math::Vector3f(6.0f, 6.0f, 5.0f) },
		{ math::Vector3f(9.0f, 9.0f, 8.0f), math::Vector3f(12.0, 12.0f, 11.0f) } } });

	BOOST_REQUIRE(math::almost_equal_verbose(b_plus, expected_b_plus, 1e-6));
}

BOOST_AUTO_TEST_CASE(transformation_vector_to_matrix3d_test01) {
	using namespace Eigen;

	Matrix<float, 6, 1> vector;
	vector << 0.f, 0.f, 0.f, 0.f, 0.f, 0.f;
	MatrixXf matrix(4, 4);
	matrix = math::transformation_vector_to_matrix3d(vector);
	MatrixXf expected_matrix(4, 4);
	expected_matrix << 1.f, 0.f, 0.f, 0.f,
			           0.f, 1.f, 0.f, 0.f,
			           0.f, 0.f, 1.f, 0.f,
			           0.f, 0.f, 0.f, 1.f;
	BOOST_REQUIRE(math::almost_equal(matrix, expected_matrix, 1e-8));
}

BOOST_AUTO_TEST_CASE(transformation_vector_to_matrix3d_test02) {
	using namespace Eigen;

	Matrix<float, 6, 1> vector;
	vector << 0.f, 0.f, 0.f, 0.f, M_PI/3, 0.f;
	MatrixXf matrix(4, 4);
	matrix = math::transformation_vector_to_matrix3d(vector);
	MatrixXf expected_matrix(4, 4);
	expected_matrix << 0.5f, 0.f, 0.8660254f, 0.f,
			           0.f, 1.f, 0.f, 0.f,
			           -0.8660254f, 0.f, 0.5f, 0.f,
			           0.f, 0.f, 0.f, 1.f;
	BOOST_REQUIRE(math::almost_equal(matrix, expected_matrix, 1e-6));
}

BOOST_AUTO_TEST_CASE(transformation_vector_to_matrix3d_test03) {
    using namespace Eigen;

	Matrix<float, 6, 1> vector;
	vector << -0.03258679, 0.f, 0.00103423f, 0.f, 0.06314822f, 0.f;
	MatrixXf matrix(4, 4);
	matrix = math::transformation_vector_to_matrix3d(vector);
	MatrixXf expected_matrix(4, 4);
	expected_matrix << 0.99800682f, 0.f, 0.06310626f, -0.03258679f,
	                   0.f, 1.f, 0.f, 0.f,
	                   -0.06310626f, 0.f, 0.99800682f, 0.00103423f,
	                   0.f, 0.f, 0.f, 1.f;
	BOOST_REQUIRE(math::almost_equal(matrix, expected_matrix, 1e-6));
}
