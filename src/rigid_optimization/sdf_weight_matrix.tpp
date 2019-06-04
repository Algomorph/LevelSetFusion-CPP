/*
 * sdf_weight.tpp
 *
 *  Created on: Jun 04, 2019
 *      Author: Fei Shan
 */

#pragma once

//local
#include "sdf_weight.hpp"

namespace eig = Eigen;

namespace rigid_optimization {

    template<typename Scalar>
    eig::Matrix<Scalar, eig::Dynamic, eig::Dynamic, eig::ColMajor> sdf_weight(
            const eig::Matrix<Scalar, eig::Dynamic, eig::Dynamic, eig::ColMajor>& field,
            const Scalar& eta){

        eig::Matrix<Scalar, eig::Dynamic, eig::Dynamic, eig::ColMajor> weight = field.replicate(1, 1);

        int matrix_size = static_cast<int>(weight.size());

        int x_size = weight.cols();

#pragma omp parallel for
        for (int i_element = 0; i_element < matrix_size; i_element++) {
            // Any MatrixXf in Eigen is column-major
            // i_element = x * column_count + y
            int x_field = i_element / x_size;
            int y_field = i_element % x_size;
            weight(y_field, x_field) = (field(y_field, x_field) <= -eta) ? Scalar(0.0f) : Scalar(1.0f);
        }

//        for (int i = 0; i < sdf_weight.rows(); ++i) { // Determine weight based on thickness
//            for (int j = 0; j < sdf_weight.cols(); ++j) {
//                weight(i, j) = (field(i, j) <= -eta) ? 0.0f : 1.0f;
//            }
//        }

        return weight;
}
;

}