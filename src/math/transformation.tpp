//  ================================================================
//  Created by Fei Shan on 03/05/19.
//  Converting transformation vector to matrix in 2D and 3D homogeneous coordinates.
//  ================================================================

#include "transformation.hpp"
#include "math.h"

namespace eig = Eigen;

namespace math {

template<typename Scalar>
eig::Matrix<Scalar, 3, 3> transformation_vector_to_matrix(const eig::Matrix<Scalar, 3, 1>& twist) {
    double(theta) = twist(2);
    eig::Matrix<Scalar, 3, 3> twist_matrix;
    twist_matrix << cos(theta), -sin(theta), twist(0),
                    sin(theta), cos(theta),  twist(1),
                    0.f,        0.f,         1.f;
    return twist_matrix;
}

template<typename Scalar>
eig::Matrix<Scalar, 4, 4> transformation_vector_to_matrix(const eig::Matrix<Scalar, 6, 1>& twist){
    eig::Matrix<Scalar, 3, 1> translation = twist.head(3);
    eig::Matrix<Scalar, 3, 1> rotation = twist.tail(3);
    float theta = rotation.norm();

    if (std::abs(theta) > 1e-14) {
        rotation /= theta;
    }

    eig::Quaternionf q;
    q = eig::Quaternionf(cos(theta/2),
                         sin(theta/2) * rotation(0),
                         sin(theta/2) * rotation(1),
                         sin(theta/2) * rotation(2));

    eig::Matrix<Scalar, 3, 3> rotation_matrix = q.toRotationMatrix();

    eig::Matrix<Scalar, 4, 4> twist_matrix;

    twist_matrix << rotation_matrix(0, 0), rotation_matrix(0, 1), rotation_matrix(0, 2), translation(0),
                    rotation_matrix(1, 0), rotation_matrix(1, 1), rotation_matrix(1, 2), translation(1),
                    rotation_matrix(2, 0), rotation_matrix(2, 1), rotation_matrix(2, 2), translation(2),
                    0.0f,                  0.0f,                  0.0f,                  1.0f;
    return twist_matrix;
}

template<typename Scalar>
eig::Matrix<Scalar, 4, 4> inverse_transformation_matrix(const eig::Matrix<Scalar, 4, 4>& twist_matrix){
    eig::Matrix<Scalar, 3, 1> inv_translation;
    inv_translation << -twist_matrix(0, 0)*twist_matrix(0, 3)-twist_matrix(1, 0)*twist_matrix(1, 3)-twist_matrix(2, 0)*twist_matrix(2, 3),
                       -twist_matrix(0, 1)*twist_matrix(0, 3)-twist_matrix(1, 1)*twist_matrix(1, 3)-twist_matrix(2, 1)*twist_matrix(2, 3),
                       -twist_matrix(0, 2)*twist_matrix(0, 3)-twist_matrix(1, 2)*twist_matrix(1, 3)-twist_matrix(2, 2)*twist_matrix(2, 3);

    eig::Matrix<Scalar, 4, 4> inv_twist_matrix;
    inv_twist_matrix << twist_matrix(0, 0), twist_matrix(1, 0), twist_matrix(2, 0), inv_translation(0),
                        twist_matrix(0, 1), twist_matrix(1, 1), twist_matrix(2, 1), inv_translation(1),
                        twist_matrix(0, 2), twist_matrix(1, 2), twist_matrix(2, 2), inv_translation(2),
                        twist_matrix(3, 0), twist_matrix(3, 1), twist_matrix(3, 2), twist_matrix(3, 3);
    return inv_twist_matrix;
}

template <typename Scalar>
eig::Matrix<Scalar, 3, 3> init_transformation_matrix(const eig::Matrix<Scalar, eig::Dynamic, eig::Dynamic>& field){
    eig::Matrix<Scalar, 3, 3> matrix;
    matrix.setZero();
    return matrix;
};

template <typename Scalar>
eig::Matrix<Scalar, 4, 4> init_transformation_matrix(const eig::Tensor<Scalar, 3>& field){
    eig::Matrix<Scalar, 4, 4> matrix;
    matrix.setZero();
    return matrix;
};

template <typename Scalar>
eig::Matrix<Scalar, 3, 1> init_transformation_vector(const eig::Matrix<Scalar, eig::Dynamic, eig::Dynamic>& field){
    eig::Matrix<Scalar, 3, 1> vector;
    vector.setZero();
    return vector;
};

template <typename Scalar>
eig::Matrix<Scalar, 6, 1> init_transformation_vector(const eig::Tensor<Scalar, 3>& field){
    eig::Matrix<Scalar, 6, 1> vector;
    vector.setZero();
    return vector;
};

template <typename Scalar>
eig::Matrix<Scalar, 6, 1> to_3d_transformation_vector(const eig::Matrix<Scalar, 3, 1>& twist){
    eig::Matrix<Scalar, 6, 1> twist3d;
    twist3d << twist(0), Scalar(0.f), twist(1), Scalar(0.f), twist(2), Scalar(0.f);
    return twist3d;
};

template <typename Scalar>
eig::Matrix<Scalar, 6, 1> to_3d_transformation_vector(const eig::Matrix<Scalar, 6, 1>& twist){
    eig::Matrix<Scalar, 6, 1> twist3d = eig::Matrix<Scalar, 6, 1>(twist);
    return twist3d;
};

template <typename Scalar>
eig::Matrix<Scalar, 3, 1> to_2d_transformation_vector(const eig::Matrix<Scalar, 6, 1>& twist){
    eig::Matrix<Scalar, 3, 1> twist2d;
    twist2d << twist(0), twist(2), twist(4);
    return twist2d;
};

template <typename Scalar>
eig::Matrix<Scalar, 3, 1> to_2d_transformation_vector(const eig::Matrix<Scalar, 3, 1>& twist){
    eig::Matrix<Scalar, 3, 1> twist2d = eig::Matrix<Scalar, 3, 1>(twist);
    return twist2d;
};



}