/*
 * test_checks.cpp
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

#define BOOST_TEST_MODULE test_checks

//_DEBUG
#include <iostream>

//libraries
#include <boost/test/unit_test.hpp>
#include <boost/python.hpp>
#include <Eigen/Dense>
#include <unsupported/Eigen/CXX11/Tensor>

//test targets
#include "../src/math/almost_equal.hpp"
#include "../src/math/checks.hpp"
#include "../src/math/typedefs.hpp"

namespace eig = Eigen;

BOOST_AUTO_TEST_CASE(test_almost_equal_reals) {
	float a = 10.2093f;
	float b = 10.2394f;
	BOOST_REQUIRE(math::almost_equal(a, b, 0.04));
	BOOST_REQUIRE(!math::almost_equal(a, b, 0.03));
	double c = -291.299912;
	double d = -291.299936;
	BOOST_REQUIRE(math::almost_equal(c, d, 1e-4));
	BOOST_REQUIRE(!math::almost_equal(c, d, 1e-5));
	float e = 10.23935;
	BOOST_REQUIRE(math::almost_equal(b, e, 1e-4));
	BOOST_REQUIRE(!math::almost_equal(b, e, 1e-5));
}

BOOST_AUTO_TEST_CASE(test_almost_equal_real_matrices) {
	eig::MatrixXf a(2, 3);
	a << 1.0f, -2.0f, 3.0f,
			4.0f, 5.0f, 6.0f;
	eig::MatrixXf b(3, 2);
	b << 1.0f, -2.0f,
			3.0f, 4.0f,
			5.0f, 6.0f;
	BOOST_REQUIRE(!math::almost_equal(a, b, 1e-5));
	eig::MatrixXf c(2, 3);
	c << 1.0f, -2.0f, 3.0f,
			4.0f, 5.0f, 6.0000004f;
	BOOST_REQUIRE(math::almost_equal(a, c, 1e-5));
	BOOST_REQUIRE(!math::almost_equal(a, c, 1e-7));

	eig::MatrixXd e(2, 1);
	e << -10.003, 9.9997;
	eig::MatrixXd f(2, 1);
	f << -10.00, 10.00;
	BOOST_REQUIRE(math::almost_equal(e, f, 1e-2));
	BOOST_REQUIRE(!math::almost_equal(e, f, 1e-3));
}

BOOST_AUTO_TEST_CASE(test_almost_equal_nested_matrices) {
	math::MatrixXv2f a(1, 2);
	a << math::Vector2f(1.0f, 2.0f), math::Vector2f(-3.0f, 4.0f);
	math::MatrixXv2f b(1, 2);
	b << math::Vector2f(1.0f, 2.0f), math::Vector2f(-3.0f, 4.000002f);
	BOOST_REQUIRE(math::almost_equal(a, b, 1e-5));
	BOOST_REQUIRE(!math::almost_equal(a, b, 1e-6));

	math::MatrixXm2f c(2, 1);
	c << math::Matrix2f(1.0f, 2.0f, 3.0f, 4.0f), math::Matrix2f(4.0f, 0.0f, -12.0f, -230.129f);
	math::MatrixXm2f d(2, 1);
	d << math::Matrix2f(1.0f, 2.0f, 3.0f, 4.0f), math::Matrix2f(4.0f, 0.0f, -12.0f, -230.1291f);
	BOOST_REQUIRE(math::almost_equal(c, d, 1e-3));
	BOOST_REQUIRE(!math::almost_equal(c, d, 1e-4));

	math::MatrixXv2f e(2, 1);
	e << math::Vector2f(1.0f, 2.0f), math::Vector2f(-3.0f, 4.000002f);

	BOOST_REQUIRE(!math::almost_equal(b, e, 1e-4));
}

BOOST_AUTO_TEST_CASE(test_almost_equal_real_tensors) {
	math::Tensor3f a(1, 3, 2);
	a.setValues(
			{ { { 1.0f, 2.0f },
					{ 3.0f, -4.0f },
					{ 5.0f, 6.0f } } }
			);
	math::Tensor3f b(1, 3, 2);
	b.setValues(
			{ { { 1.0f, 2.0f },
					{ 3.0f, -3.9998f },
					{ 5.0f, 6.0f } } }
			);

	BOOST_REQUIRE(math::almost_equal(a, b, 3e-4));
	BOOST_REQUIRE(!math::almost_equal(a, b, 1e-4));

	math::Tensor3f c(1, 2, 3);
	c.setValues(
			{ { { 1.0f, 2.0f, 3.0f },
					{ -3.9998f, 5.0f, 6.0f } } }
			);
	BOOST_REQUIRE(!math::almost_equal(b, c, 1e-4));
}

typedef math::Vector3f t3;

BOOST_AUTO_TEST_CASE(test_almost_equal_nested_tensors) {
	math::Tensor3v3f a(1, 3, 2);

	a.setValues(
			{ { { t3(1.0f), t3(2.0f) },
					{ t3(3.0f), t3(-4.0f) },
					{ t3(5.0f), t3(6.0f) } } }
			);
	math::Tensor3v3f b(1, 3, 2);
	b.setValues(
			{ { { t3(1.0f), t3(2.0f) },
					{ t3(3.0f), t3(-4.0f, -3.9998f, -4.0f) },
					{ t3(5.0f), t3(6.0f) } } }
			);
	BOOST_REQUIRE(math::almost_equal(a, b, 3e-4));
	BOOST_REQUIRE(!math::almost_equal(a, b, 1e-4));

	math::Tensor3v3f c(1, 2, 3);
	c.setValues(
			{ { { t3(1.0f), t3(2.0f), t3(3.0f) },
					{ t3(-4.0f, -3.9998f, -4.0f), t3(5.0f), t3(6.0f) } } }
			);
	BOOST_REQUIRE(!math::almost_equal(b, c, 1e-4));

}

BOOST_AUTO_TEST_CASE(test_is_power_of_two) {
	BOOST_REQUIRE(math::is_power_of_two(128));
	BOOST_REQUIRE(math::is_power_of_two(2));
	BOOST_REQUIRE(math::is_power_of_two(16));
	BOOST_REQUIRE(!math::is_power_of_two(17));
	BOOST_REQUIRE(!math::is_power_of_two(38));
}

BOOST_AUTO_TEST_CASE(test_dimension_checks_matrix) {
	eig::MatrixXf a(2,3);
	eig::MatrixXf b(2,3);
	eig::MatrixXf c(2,3);
	eig::MatrixXf d(3,2);
	eig::MatrixXf e(2,4);
	BOOST_REQUIRE(math::are_dimensions_equal(a,b));
	BOOST_REQUIRE(!math::are_dimensions_equal(a,d));
	BOOST_REQUIRE(!math::are_dimensions_equal(a,e));
	BOOST_REQUIRE(math::are_dimensions_equal(a,b,c));
	BOOST_REQUIRE(!math::are_dimensions_equal(a,e,c));

	eig::MatrixXf f(2,3);
	math::MatrixXv2f g(2,3);
	math::MatrixXv2f h(2,4);
	eig::MatrixXd i(2,3);
	BOOST_REQUIRE(math::are_dimensions_equal(f,g));
	BOOST_REQUIRE(!math::are_dimensions_equal(f,h));
	BOOST_REQUIRE(!math::are_dimensions_equal(g,h));
	BOOST_REQUIRE(!math::are_dimensions_equal(g,h,i));
	BOOST_REQUIRE(math::are_dimensions_equal(f,g,i));
}

BOOST_AUTO_TEST_CASE(test_dimension_checks_tensor) {
	math::Tensor3f a(2,3,2);
	math::Tensor3f b(2,3,2);
	math::Tensor3f c(2,3,2);
	math::Tensor3f d(3,2,2);
	math::Tensor3f e(2,4,2);
	BOOST_REQUIRE(math::are_dimensions_equal(a,b));
	BOOST_REQUIRE(!math::are_dimensions_equal(a,d));
	BOOST_REQUIRE(!math::are_dimensions_equal(a,e));
	BOOST_REQUIRE(math::are_dimensions_equal(a,b,c));
	BOOST_REQUIRE(!math::are_dimensions_equal(a,e,c));

	math::Tensor3f f(1,2,3);
	math::Tensor3v3f g(1,2,3);
	math::Tensor3v3f h(1,2,4);
	eig::Tensor<double, 3, eig::ColMajor> i(1,2,3);
	BOOST_REQUIRE(math::are_dimensions_equal(f,g));
	BOOST_REQUIRE(!math::are_dimensions_equal(f,h));
	BOOST_REQUIRE(!math::are_dimensions_equal(g,h));
	BOOST_REQUIRE(!math::are_dimensions_equal(g,h,i));
	BOOST_REQUIRE(math::are_dimensions_equal(f,g,i));
}
