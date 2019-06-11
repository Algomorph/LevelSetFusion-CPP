/*
 * sdf2sdf_optimizer.hpp
 *
 *  Created on: May 31, 2019
 *      Author: Fei Shan
 */

#pragma once

//libraries
#include <Eigen/Eigen>
#include <unsupported/Eigen/CXX11/Tensor>

//local
#include "../math/typedefs.hpp"
#include "../tsdf/generator.hpp"

namespace eig = Eigen;

namespace rigid_optimization {

template<typename ScalarContainer, typename VectorContainer>
class Sdf2SdfOptimizer {

typedef typename ScalarContainer::Scalar Scalar;
typedef typename math::ContainerWrapper<ScalarContainer>::Coordinates Coordinates;
typedef typename math::ContainerWrapper<ScalarContainer>::TransformationVector TransformationVector;
typedef typename math::ContainerWrapper<ScalarContainer>::TransformationMatrix TransformationMatrix;
typedef typename math::ContainerWrapper<ScalarContainer>::Sdf2SdfOptimizerCoefficientA OptimizerCoefficient;
typedef typename tsdf::Generator<ScalarContainer> TsdfGenerator;
typedef typename tsdf::Parameters<ScalarContainer> TsdfGenerationParameters;

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

    TransformationMatrix optimize(const ScalarContainer& canonical_field,
                                  const eig::Matrix<unsigned short, eig::Dynamic, eig::Dynamic>& live_depth_image,
                                  float eta = 0.01f,
                                  const eig::Matrix4f& initial_camera_pose = eig::Matrix4f::Identity(),
                                  int image_y_coordinate = 0);

private:
    const float rate = 0.5f;
    const int maximum_iteration_count = 60;
    const TsdfGenerator tsdf_generator;
    const Sdf2SdfOptimizer::VerbosityParameters verbosity_parameters;

};

}
