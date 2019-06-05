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

template<typename Scalar, typename ScalarContainer, typename TsdfGenerationParameters, typename TsdfGenerator, typename Transformation>
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

    Transformation optimize(const ScalarContainer& canonical_field,
                            const eig::Matrix<unsigned short, eig::Dynamic, eig::Dynamic>& live_depth_image,
                            float eta = 0.01f,
                            const eig::Matrix4f& initial_camera_pose = eig::Matrix4f::Identity(),
                            int image_y_coordinate = 0) {
        // Overload.
        return optimizeImpl(canonical_field,
                            live_depth_image,
                            eta,
                            initial_camera_pose,
                            image_y_coordinate);
    };

private:
    const float rate = 0.5f;
    const int maximum_iteration_count = 60;
    const TsdfGenerator tsdf_generator;
    const Sdf2SdfOptimizer::VerbosityParameters verbosity_parameters;

    // 2D
    eig::Matrix3f optimizeImpl(const eig::Matrix<Scalar, eig::Dynamic, eig::Dynamic>& canonical_field,
                               const eig::Matrix<unsigned short, eig::Dynamic, eig::Dynamic>& live_depth_image,
                               float eta,
                               const eig::Matrix4f& initial_camera_pose,
                               int image_y_coordinate);
    // 3D
    eig::Matrix4f optimizeImpl(const eig::Tensor<Scalar, 3>& canonical_field,
                               const eig::Matrix<unsigned short, eig::Dynamic, eig::Dynamic>& live_depth_image,
                               float eta,
                               const eig::Matrix4f& initial_camera_pose,
                               int image_y_coordinate);

};

}
