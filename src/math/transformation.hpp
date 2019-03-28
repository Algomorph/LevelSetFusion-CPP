//  ================================================================
//  Created by Fei Shan on 03/05/19.
//  ================================================================

#pragma once

//libraries
#include <Eigen/Eigen>

namespace eig = Eigen;

namespace math{

    eig::Matrix3f transformation_vector_to_matrix2d(const eig::Vector3f &twist);
    eig::Matrix4f transformation_vector_to_matrix3d(const eig::Matrix<float, 6, 1>& twist);

}
