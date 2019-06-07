/*
 * test_sdf_2_sdf_optimizer.cpp
 *
 *  Created on: Jun 07, 2019
 *      Author: Fei Shan
 */

#define BOOST_TEST_MODULE test_sdf_2_sdf_optimizer

//stdlib
//_DEBUG
#include <iostream>

//libraries
#include <boost/test/unit_test.hpp>
#include <boost/python.hpp>
#include <Eigen/Eigen>

//test data

//test targets
#include "../src/math/almost_equal.hpp"
#include "../src/rigid_optimization/sdf_weight.hpp"
#include "../src/rigid_optimization/sdf_gradient_wrt_transformation.hpp"
#include "../src/rigid_optimization/sdf_2_sdf_optimizer.hpp"

namespace eig = Eigen;
namespace ro = rigid_optimization;

BOOST_AUTO_TEST_CASE(test_sdf_weight01) {
    eig::MatrixXf field(4, 4);
    field.setZero();
    float eta = -1.f;
    eig::MatrixXf expected_weight = field;
    eig::MatrixXf weight = ro::sdf_weight(field, eta);

    BOOST_REQUIRE(math::almost_equal_verbose(weight, expected_weight, 1e-14));
}