/*
 * sdf2sdf_optimizer.hpp
 *
 *  Created on: May 31, 2019
 *      Author: Fei Shan
 */

#pragma once

//libraries
#include <Eigen/Eigen>

//local
#include "../math/typedefs.hpp"
#include "../tsdf/generator.hpp"

namespace eig = Eigen;

namespace rigid_optimization {

template<typename ScalarContainer, typename VectorContainer, typename TsdfGenerationParameters, typename TsdfGenerator>
class Sdf2SdfOptimizer {
public:
    struct VerbosityParameters {
        VerbosityParameters(bool print_iteration_max_warp_update = false,
                            bool print_iteration_energy = false);
        //per-iteration parameters
        const bool print_iteration_max_warp_update = false;
        const bool print_iteration_energy = false;
        const bool print_per_iteration_info = false;
    };

    Sdf2SdfOptimizer(
            float rate = 0.5f,
            int maximum_iteration_count = 60,
            TsdfGenerationParameters tsdf_generation_parameters = TsdfGenerationParameters(),
            VerbosityParameters verbosity_parameters = VerbosityParameters()
    );

    virtual ~Sdf2SdfOptimizer() = default;

//    eig::Matrix3f optimize(int image_y_coordinate,
//                           const ScalarContainer canonical_field,
//                           const eig::Matrix<unsigned short, eig::Dynamic, eig::Dynamic>& live_depth_image,
//                           float eta = 0.01f,
//                           const eig::Matrix4f& initial_camera_pose = eig::Matrix4f::Identity());
//
//
//    eig::Matrix4f optimize(const ScalarContainer canonical_field,
//                           const eig::Matrix<unsigned short, eig::Dynamic, eig::Dynamic>& live_depth_image,
//                           float eta = 0.01f,
//                           const eig::Matrix4f& initial_camera_pose = eig::Matrix4f::Identity());
private:
    const float rate = 0.5f;
    const int maximum_iteration_count = 60;
    const TsdfGenerator tsdf_generator;
    const Sdf2SdfOptimizer::VerbosityParameters verbosity_parameters;

};

template<>
class Sdf2SdfOptimizer<eig::MatrixXf, math::MatrixXv3f, tsdf::Parameters2d, tsdf::Generator2d> {
public:
    eig::Matrix3f optimize(int image_y_coordinate,
                           const eig::MatrixXf canonical_field,
                           const eig::Matrix<unsigned short, eig::Dynamic, eig::Dynamic>& live_depth_image,
                           float eta = 0.01f,
                           const eig::Matrix4f& initial_camera_pose = eig::Matrix4f::Identity());
};


template<>
class Sdf2SdfOptimizer<math::Tensor3f, math::Tensor3v6f, tsdf::Parameters3d, tsdf::Generator3d> {
public:
    eig::Matrix4f optimize(const math::Tensor3f canonical_field,
                           const eig::Matrix<unsigned short, eig::Dynamic, eig::Dynamic>& live_depth_image,
                           float eta = 0.01f,
                           const eig::Matrix4f& initial_camera_pose = eig::Matrix4f::Identity());
};



}
