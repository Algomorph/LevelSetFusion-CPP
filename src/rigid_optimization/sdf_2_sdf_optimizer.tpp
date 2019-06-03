/*
 * sdf2sdf_optimizer.cpp
 *
 *  Created on: May 31, 2019
 *      Author: Fei Shan
 */

//libraries
#include <boost/python.hpp>

//_DEBUG
#include <iostream>

// local
#include "../math/transformation.hpp"
#include "sdf_2_sdf_optimizer.hpp"
#include "sdf_gradient_wrt_transformation.hpp"

namespace rigid_optimization {

template<typename ScalarContainer, typename VectorContainer, typename TsdfGenerationParameters, typename TsdfGenerator>
Sdf2SdfOptimizer<ScalarContainer, VectorContainer, TsdfGenerationParameters, TsdfGenerator>::Sdf2SdfOptimizer(
        float rate,
        int maximum_iteration_count,
        TsdfGenerationParameters tsdf_generation_parameters,
        VerbosityParameters verbosity_parameters) :
            rate(rate),
            maximum_iteration_count(maximum_iteration_count),
            tsdf_generator(tsdf_generation_parameters),
            verbosity_parameters(verbosity_parameters)
{
};

template<typename ScalarContainer, typename VectorContainer, typename TsdfGenerationParameters, typename TsdfGenerator>
Sdf2SdfOptimizer<ScalarContainer, VectorContainer, TsdfGenerationParameters, TsdfGenerator>::VerbosityParameters::VerbosityParameters(
        bool print_iteration_max_warp_update,
        bool print_iteration_energy) :
        print_iteration_max_warp_update(print_iteration_max_warp_update),
        print_iteration_energy(print_iteration_energy),
        print_per_iteration_info(
                print_iteration_max_warp_update ||
                print_iteration_energy
        )
{
};

template<>
eig::Matrix3f Sdf2SdfOptimizer<eig::MatrixXf, math::MatrixXv3f, tsdf::Parameters2d, tsdf::Generator2d>::optimize(
        int image_y_coordinate,
        const eig::MatrixXf canonical_field,
        const const eig::Matrix<unsigned short, eig::Dynamic, eig::Dynamic>& live_depth_image,
        float eta,
        const eig::Matrix4f& initial_camera_pose
        ) {
};

template<>
eig::Matrix4f Sdf2SdfOptimizer<math::Tensor3f, math::Tensor3v6f, tsdf::Parameters3d, tsdf::Generator3d>::optimize(
        const math::Tensor3f canonical_field,
        const eig::Matrix<unsigned short, eig::Dynamic, eig::Dynamic>& live_depth_image,
        float eta,
        const eig::Matrix4f& initial_camera_pose
        ) {

};

}
