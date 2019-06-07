/*
 * sdf_weight_tensor.tpp
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
eig::Tensor<Scalar, 3, eig::ColMajor> sdf_weight(const eig::Tensor<Scalar, 3, eig::ColMajor>& field, Scalar eta){

    int x_size = field.dimension(0);
    int y_size = field.dimension(1);
    int z_size = field.dimension(2);

    eig::Tensor<Scalar, 3, eig::ColMajor> weight(x_size, y_size, z_size);
    weight.setZero();

    int y_stride = x_size;
    int z_stride = y_stride * y_size;

    int voxel_count = static_cast<int>(field.size());

#pragma omp parallel for
    for (int i_element = 0; i_element < voxel_count; i_element++) {
        int z_field = i_element / z_stride;
        int remainder = i_element % z_stride;
        int y_field = remainder / y_stride;
        int x_field = remainder % y_stride;

        weight(x_field, y_field, z_field) = (field(x_field, y_field, z_field) <= -eta) ? Scalar(0.0f) : Scalar(1.0f);
    }

    return weight;
}
;

}
