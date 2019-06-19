/*
 * sdf2sdf_optimizer.tpp
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
#include "sdf_weight.hpp"

namespace rigid_optimization {

template<typename ScalarContainer, typename VectorContainer>
Sdf2SdfOptimizer<ScalarContainer, VectorContainer>::Sdf2SdfOptimizer(
        float rate,
        int maximum_iteration_count,
        TsdfGenerationParameters tsdf_generation_parameters,
        VerbosityParameters verbosity_parameters) :
            rate(rate),
            maximum_iteration_count(maximum_iteration_count),
            tsdf_generator(tsdf_generation_parameters),
            verbosity_parameters(verbosity_parameters)
{
}
;

template<typename ScalarContainer, typename VectorContainer>
Sdf2SdfOptimizer<ScalarContainer, VectorContainer>::VerbosityParameters::VerbosityParameters(
        bool print_iteration_max_warp_update,
        bool print_iteration_energy) :
        print_iteration_max_warp_update(print_iteration_max_warp_update),
        print_iteration_energy(print_iteration_energy),
        print_per_iteration_info(
                print_iteration_max_warp_update ||
                print_iteration_energy
        )
{
}
;

template<typename ScalarContainer, typename VectorContainer>
typename math::ContainerWrapper<ScalarContainer>::TransformationMatrix
Sdf2SdfOptimizer<ScalarContainer, VectorContainer>::optimize(
        const ScalarContainer& canonical_field,
        const eig::Matrix<unsigned short, eig::Dynamic, eig::Dynamic>& live_depth_image,
        float eta,
        const eig::Matrix4f& initial_camera_pose,
        int image_y_coordinate) {

    ScalarContainer canonical_weight = sdf_weight(canonical_field, eta);

    TransformationVector twist;
    twist.setZero();

    for (int iteration_count = 0; iteration_count < maximum_iteration_count; ++iteration_count) {
        OptimizerCoefficient matrix_A;
        matrix_A.setZero();
        TransformationVector vector_b;
        vector_b.setZero();

        // Expand in 2D case.
        eig::Matrix<Scalar, 6, 1> twist3d = math::to_3d_transformation_vector(twist);

        eig::Matrix<Scalar, 4, 4> twist_matrix3d = math::transformation_vector_to_matrix(twist3d);

        ScalarContainer live_field = tsdf_generator.generate(live_depth_image,
                                                             twist_matrix3d,
                                                             image_y_coordinate);
        ScalarContainer live_weight = sdf_weight(live_field, eta);

        VectorContainer live_gradient = init_gradient_wrt_twist(live_field);

        float energy = gradient_wrt_twist(live_field,
                                          twist,
                                          tsdf_generator.parameters.array_offset,
                                          tsdf_generator.parameters.voxel_size,
                                          canonical_field,
                                          live_gradient,
                                          matrix_A,
                                          vector_b);

        eig::Matrix<Scalar, eig::Dynamic, eig::Dynamic> optimal_twist(twist.size(), 1);
        optimal_twist = matrix_A.inverse() * vector_b;
        twist += this->rate * (optimal_twist - twist);

        if (this->verbosity_parameters.print_per_iteration_info) {
            std::cout << "[ITERATION " << iteration_count << " COMPLETED]\n";
            if (this->verbosity_parameters.print_iteration_max_warp_update) {
                std::cout << " [optimize twist:" << optimal_twist.transpose() << "]\n";
                std::cout << " [twist:" << twist.transpose() << "]\n";
            }
            if (this->verbosity_parameters.print_iteration_energy) {
                std::cout << " [energy: " << energy << "]\n\n" << std::endl;
            }

        }

    }

    TransformationMatrix twist_matrix = math::transformation_vector_to_matrix(twist);
    return twist_matrix;
}
;

}
