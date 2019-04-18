/*
 * test_gradient.cpp
 *
 *  Created on: Apr 17, 2019
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

#define BOOST_TEST_MODULE test_math

//stdlib
//_DEBUG
#include <iostream>
#include "../src/console/pretty_printers.hpp"

//libraries
#include <boost/test/unit_test.hpp>
#include <boost/python.hpp>
#include <Eigen/Eigen>

//test data
#include "data/test_data_gradients.hpp"

//test targets
#include "../src/math/checks.hpp"
#include "../src/math/gradients.hpp"
#include "../src/math/almost_equal.hpp"
#include "../src/math/typedefs.hpp"
#include "../src/math/stacking.hpp"

BOOST_AUTO_TEST_CASE(scalar_field_gradient_test01) {
	eig::MatrixXf field(2, 2);
	field << -0.46612028, -0.8161121,
			0.2427629, -0.79432599;

	eig::MatrixXf expected_gradient_x(2, 2), expected_gradient_y(2, 2);
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
	eig::MatrixXf field(3, 3);
	field << 0.11007435, -0.94589225, -0.54835034,
			-0.09617922, 0.15561824, 0.60624432,
			-0.83068796, 0.19262577, -0.21090505;

	eig::MatrixXf expected_gradient_x(3, 3), expected_gradient_y(3, 3);
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

BOOST_AUTO_TEST_CASE(test_vector_field_gradient01) {
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

BOOST_AUTO_TEST_CASE(test_vector_field_gradient02) {
	math::MatrixXm2f gradient;
	math::gradient(gradient, test_data::vector_field);
	BOOST_REQUIRE(math::almost_equal(gradient, test_data::expected_vector_field_gradient, 1e-6));
}

BOOST_AUTO_TEST_CASE(test_scalar_field_graident_tensor){
	math::Tensor3f scalar_field(4,3,2);
	scalar_field.setValues(  // @formatter:off
		{{{1.0f, 13.0f},
		  {5.0f, 17.0f},
		  {9.0f, 21.0f}},
		 {{2.0f, 14.0f},
		  {6.0f, 18.0f},
		  {10.0f, 22.0f}},
		 {{3.0f, 15.0f},
		  {7.0f, 19.0f},
		  {11.0f, 23.0f}},
		 {{4.0f, 16.0f},
		  {8.0f, 20.0f},
		  {12.0f, 24.0f}}}
	); // @formatter:on
	math::Tensor3v3f gradient;
	math::gradient(gradient, scalar_field);
	BOOST_REQUIRE(math::are_dimensions_equal(scalar_field, gradient));
	math::Vector3f exp_out_elem(1.0f, 4.0f, 12.0f);
	math::Tensor3v3f expected_gradient = gradient.constant(exp_out_elem);
	BOOST_REQUIRE(math::almost_equal(gradient, expected_gradient, 1e-6));
}

BOOST_AUTO_TEST_CASE(test_gradient_matrix_matrix) {
	math::MatrixXv2f vector_field(3, 3);
	vector_field <<
			math::Vector2f(1.0f, 2.0f),
			math::Vector2f(3.0f, 4.0f),
			math::Vector2f(5.0f, 6.0f),

			math::Vector2f(7.0f, 8.0f),
			math::Vector2f(9.0f, 10.0f),
			math::Vector2f(11.0f, 12.0f),

			math::Vector2f(13.0f, 14.0f),
			math::Vector2f(15.0f, 16.0f),
			math::Vector2f(17.0f, 18.0f);
	math::MatrixXm2f vector_field_gradient;
	math::gradient(vector_field_gradient, vector_field);
	math::MatrixXm2f expected_vector_field_gradient(3,3);
	expected_vector_field_gradient <<
		math::Matrix2f(2.0f, 6.0f, 2.0f, 6.0f), math::Matrix2f(2.0f, 6.0f, 2.0f, 6.0f), math::Matrix2f(2.0f, 6.0f, 2.0f, 6.0f),
		math::Matrix2f(2.0f, 6.0f, 2.0f, 6.0f), math::Matrix2f(2.0f, 6.0f, 2.0f, 6.0f), math::Matrix2f(2.0f, 6.0f, 2.0f, 6.0f),
		math::Matrix2f(2.0f, 6.0f, 2.0f, 6.0f), math::Matrix2f(2.0f, 6.0f, 2.0f, 6.0f), math::Matrix2f(2.0f, 6.0f, 2.0f, 6.0f);
	BOOST_REQUIRE(math::almost_equal(vector_field_gradient, expected_vector_field_gradient, 1e-6));
}

static inline
void gen_arange_tensor3v3f(math::Tensor3v3f& vector_field){
	float counter = 1.0f;
	for (int z = 0; z < vector_field.dimension(2); z++) {
		for (int y = 0; y < vector_field.dimension(1); y++) {
			for (int x = 0; x < vector_field.dimension(0); x++) {
				vector_field(x, y, z) = math::Vector3f(counter, counter + 1.0f, counter + 2.0f);
				counter += 3.0f;
			}
		}
	}
}


BOOST_AUTO_TEST_CASE(test_gradient_matrix_tensor) {
	math::Tensor3v3f vector_field(4, 4, 5);
	gen_arange_tensor3v3f(vector_field);
	math::Tensor3m3f vector_field_gradient;
	math::gradient(vector_field_gradient, vector_field);
	math::Matrix3f exp_out_elem(3.0f, 3.0f, 3.0f,  12.0f, 12.0f, 12.0f, 48.0f, 48.0f,48.0f);
	BOOST_REQUIRE(math::are_dimensions_equal(vector_field,vector_field_gradient));
	math::Tensor3m3f expected_vector_field_gradient = vector_field_gradient.constant(exp_out_elem);
	BOOST_REQUIRE(math::almost_equal_verbose(vector_field_gradient, expected_vector_field_gradient, 1e-6));
}

BOOST_AUTO_TEST_CASE(test_laplacian_matrix) {
	eig::MatrixXf a(4,4);
	//fill matrix arange-style and flipping value signs for even/odd entries
	float value = 1.0f;
	bool neg = false;
	for (int i_row = 0; i_row < a.rows(); i_row++){
		for (int i_col = 0; i_col < a.cols(); i_col++){
			if (neg){
				a(i_row, i_col) = -value;
			}else{
				a(i_row, i_col) = value;
			}
			value += 1.0f;
			neg = !neg;
		}
	}

	math::MatrixXv2f b = math::stack_as_xv2f(a,a);
	math::MatrixXv2f b_laplacian;
	math::laplacian(b_laplacian, b);
	eig::MatrixXf expected_b_laplacian_layer(4,4);
	expected_b_laplacian_layer <<
			1.0, 4.0, -8.0, 3.0,
			-11.0, 24.0, -28.0, 15.0,
			-19.0, 40.0, -44.0, 23.0,
			-31.0, 60.0, -64.0, 35.0;
	math::MatrixXv2f expected_b_laplacian = math::stack_as_xv2f(expected_b_laplacian_layer,expected_b_laplacian_layer);
	BOOST_REQUIRE(math::almost_equal_verbose(b_laplacian, expected_b_laplacian, 1e-6));
}
