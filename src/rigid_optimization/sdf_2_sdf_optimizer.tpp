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

template<typename Scalar, typename ScalarContainer, typename TsdfGenerationParameters, typename TsdfGenerator, typename Transformation>
Sdf2SdfOptimizer<Scalar, ScalarContainer, TsdfGenerationParameters, TsdfGenerator, Transformation>::Sdf2SdfOptimizer(
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

template<typename Scalar, typename ScalarContainer, typename TsdfGenerationParameters, typename TsdfGenerator, typename Transformation>
Sdf2SdfOptimizer<Scalar, ScalarContainer, TsdfGenerationParameters, TsdfGenerator, Transformation>::VerbosityParameters::VerbosityParameters(
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

template<typename Scalar, typename ScalarContainer, typename TsdfGenerationParameters, typename TsdfGenerator, typename Transformation>
eig::Matrix3f Sdf2SdfOptimizer<Scalar, ScalarContainer, TsdfGenerationParameters, TsdfGenerator, Transformation>::optimizeImpl(
        const eig::Matrix<Scalar, eig::Dynamic, eig::Dynamic>& canonical_field,
        const eig::Matrix<unsigned short, eig::Dynamic, eig::Dynamic>& live_depth_image,
        float eta,
        const eig::Matrix4f& initial_camera_pose,
        int image_y_coordinate) {

    eig::Matrix<Scalar, eig::Dynamic, eig::Dynamic> canonical_weight = sdf_weight(canonical_field, eta);

    eig::Vector3f twist;
    std::fill_n(twist.data(), twist.size(), (Scalar) 0.0);

    for (int iteration_count = 0; iteration_count < maximum_iteration_count; ++iteration_count) {
        eig::Matrix3f matrix_A;
        std::fill_n(matrix_A.data(), matrix_A.size(), (Scalar) 0.0);
        eig::Vector3f vector_b;
        std::fill_n(vector_b.data(), vector_b.size(), (Scalar) 0.0);

        // 2D case, add constrain.
        eig::Matrix<float, 6, 1> twist3d;
        twist3d << twist(0), 0.0f, twist(1), 0.0f, twist(2), 0.0f;

        eig::Matrix4f twist_matrix3d = math::transformation_vector_to_matrix(twist3d);

        eig::MatrixXf live_field = this->tsdf_generator.generate(live_depth_image,
                                                                 twist_matrix3d,
                                                                 image_y_coordinate);
        eig::Matrix<Scalar, eig::Dynamic, eig::Dynamic> live_weight = sdf_weight(live_field, eta);

        eig::Matrix<eig::Vector3f, eig::Dynamic, eig::Dynamic> live_gradient(live_field.rows(),
                                                                             live_field.cols());
        eig::Vector3i offset(tsdf_generator.parameters.array_offset.x,
                             0,
                             tsdf_generator.parameters.array_offset.y);

        gradient_wrt_twist(live_field,
                           twist,
                           offset,
                           tsdf_generator.parameters.voxel_size,
                           canonical_field,
                           live_gradient,
                           matrix_A,
                           vector_b);

        float energy = .5f * (canonical_field.cwiseProduct(canonical_weight) -
                              live_field.cwiseProduct(live_weight)).array().pow(2.f).sum();
        eig::Vector3f optimal_twist = matrix_A.inverse() * vector_b;
        twist = twist + this->rate * (optimal_twist - twist);

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

    eig::Matrix3f twist_matrix = math::transformation_vector_to_matrix(twist);
    return twist_matrix;
}
;

template<typename Scalar, typename ScalarContainer, typename TsdfGenerationParameters, typename TsdfGenerator, typename Transformation>
eig::Matrix4f Sdf2SdfOptimizer<Scalar, ScalarContainer, TsdfGenerationParameters, TsdfGenerator, Transformation>::optimizeImpl(
        const eig::Tensor<Scalar, 3>& canonical_field,
        const eig::Matrix<unsigned short, eig::Dynamic, eig::Dynamic>& live_depth_image,
        float eta,
        const eig::Matrix4f& initial_camera_pose,
        int image_y_coordinate) {

    eig::Tensor<Scalar, 3> canonical_weight = sdf_weight(canonical_field, eta);

    eig::Matrix<float, 6, 1> twist;
    std::fill_n(twist.data(), twist.size(), (Scalar) 0.0);

    for (int iteration_count = 0; iteration_count < maximum_iteration_count; ++iteration_count) {
        eig::Matrix<float, 6, 6> matrix_A;
        std::fill_n(matrix_A.data(), matrix_A.size(), (Scalar) 0.0);
        eig::Matrix<float, 6, 1> vector_b;
        std::fill_n(vector_b.data(), vector_b.size(), (Scalar) 0.0);

        eig::Matrix4f twist_matrix3d = math::transformation_vector_to_matrix(twist);

        eig::Tensor<Scalar, 3> live_field = this->tsdf_generator.generate(live_depth_image,
                                                                          twist_matrix3d,
                                                                          image_y_coordinate);
        eig::Tensor<Scalar, 3> live_weight = sdf_weight(live_field, eta);

        eig::Tensor<eig::Matrix<float, 6, 1>, 3> live_gradient(live_field.dimentions(0),
                                                               live_field.dimentions(1),
                                                               live_field.dimentions(2));
        eig::Vector3i offset(tsdf_generator.parameters.array_offset.x,
                             tsdf_generator.parameters.array_offset.y,
                             tsdf_generator.parameters.array_offset.z);

        gradient_wrt_twist(live_field,
                           twist,
                           offset,
                           tsdf_generator.parameters.voxel_size,
                           canonical_field,
                           live_gradient,
                           matrix_A,
                           vector_b);

        float energy = .5f * (canonical_field.cwiseProduct(canonical_weight) -
                              live_field.cwiseProduct(live_weight)).array().pow(2.f).sum();
        eig::Matrix<float, 6, 1> optimal_twist = matrix_A.inverse() * vector_b;
        twist = twist + this->rate * (optimal_twist - twist);

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

    eig::Matrix4f twist_matrix = math::transformation_vector_to_matrix(twist);
    return twist_matrix;
}
;

}
