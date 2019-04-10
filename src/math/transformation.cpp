//  ================================================================
//  Created by Fei Shan on 03/05/19.
//  ================================================================

#include "transformation.hpp"
#include "math.h"

namespace eig = Eigen;

namespace math {
    eig::Matrix3f transformation_vector_to_matrix2d(const eig::Vector3f& twist) {
        double(theta) = twist(2);
        eig::Matrix3f twist_matrix_homo;
        twist_matrix_homo << cos(theta), -sin(theta), twist(0),
                             sin(theta), cos(theta),  twist(1),
                             0.f,        0.f,         1.f;
        return twist_matrix_homo;
    }

    eig::Matrix4f transformation_vector_to_matrix3d(const eig::Matrix<float, 6, 1>& twist){
        eig::Vector3f translation = twist.head<3>();
        eig::Vector3f rotation = twist.tail<3>();
        float theta = rotation.norm();

        if (std::abs(theta) > 1e-14) {
            rotation /= theta;
        }

        eig::Quaternionf q;
        q = eig::Quaternionf(cos(theta/2),
                             sin(theta/2) * rotation(0),
                             sin(theta/2) * rotation(1),
                             sin(theta/2) * rotation(2));

        eig::Matrix3f rotation_matrix = q.toRotationMatrix();

        eig::Matrix4f twist_matrix_homo;

        twist_matrix_homo << rotation_matrix(0, 0), rotation_matrix(0, 1), rotation_matrix(0, 2), translation(0),
                             rotation_matrix(1, 0), rotation_matrix(1, 1), rotation_matrix(1, 2), translation(1),
                             rotation_matrix(2, 0), rotation_matrix(2, 1), rotation_matrix(2, 2), translation(2),
                             0.0f,                  0.0f,                  0.0f,                  1.0f;
        return twist_matrix_homo;
    }
}