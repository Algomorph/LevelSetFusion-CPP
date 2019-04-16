/*
 * sdf_gradient_wrt_transformation2d.h
 *
 *  Created on: Mar 05, 2019
 *      Author: Fei Shan
 */

//libraries
#include "Eigen/Eigen"

namespace eig = Eigen;

namespace rigid_optimization {
    void gradient_wrt_twist(const eig::MatrixXf& live_field,
                            const eig::Vector3f& twist2d,
                            const eig::Vector3i& array_offset,
                            const float& voxel_size,
                            eig::Matrix<eig::Vector3f, eig::Dynamic, eig::Dynamic>& gradient_field);
}