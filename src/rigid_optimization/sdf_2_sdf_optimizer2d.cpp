/*
 * sdf2sdf_optimizer2d.cpp
 *
 *  Created on: Mar 05, 2019
 *      Author: Fei Shan
 */

//libraries
#include <boost/python.hpp>

// local
#include "../math/transformation.hpp"
#include "sdf_2_sdf_optimizer2d.hpp"
#include "sdf_gradient_wrt_transformation2d.hpp"
#include "../tsdf/tsdf.hpp"

namespace ropt = rigid_optimization;
namespace eig = Eigen;

namespace rigid_optimization {

Sdf2SdfOptimizer2d::Sdf2SdfOptimizer2d(
        float rate,
        int maximum_iteration_count,
        VerbosityParameters verbosity_parameters) :
                rate(rate),
                maximum_iteration_count(maximum_iteration_count),
                verbosity_parameters(verbosity_parameters)
{
};

Sdf2SdfOptimizer2d::~Sdf2SdfOptimizer2d()
{
};

Sdf2SdfOptimizer2d::VerbosityParameters::VerbosityParameters(
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

Sdf2SdfOptimizer2d::TSDFGenerationParameters::TSDFGenerationParameters(
        float depth_unit_ratio,
        eig::Matrix3f camera_intrinsic_matrix,
        eig::Matrix4f camera_pose,
        eig::Vector3i array_offset,
        int field_size,
        float voxel_size,
        int narrow_band_width_voxels) :
            depth_unit_ratio(depth_unit_ratio),
            camera_intrinsic_matrix(camera_intrinsic_matrix),
            camera_pose(camera_pose),
            array_offset(array_offset),
            field_size(field_size),
            voxel_size(voxel_size),
            narrow_band_width_voxels(narrow_band_width_voxels)
{
};

eig::Vector3f Sdf2SdfOptimizer2d::optimize(int image_y_coordinate,
        const eig::Matrix<unsigned short, eig::Dynamic, eig::Dynamic>& canonical_depth_image,
        const eig::Matrix<unsigned short, eig::Dynamic, eig::Dynamic>& live_depth_image,
        const Sdf2SdfOptimizer2d::TSDFGenerationParameters tsdf_generation_parameters,
        float eta = 0.01f) {

    eig::MatrixXf canonical_field = tsdf::generate_TSDF_2D(image_y_coordinate,
                                                           canonical_depth_image,
                                                           tsdf_generation_parameters.depth_unit_ratio,
                                                           tsdf_generation_parameters.camera_intrinsic_matrix,
                                                           tsdf_generation_parameters.camera_pose,
                                                           tsdf_generation_parameters.array_offset,
                                                           tsdf_generation_parameters.field_size,
                                                           tsdf_generation_parameters.voxel_size,
                                                           tsdf_generation_parameters.narrow_band_width_voxels);

    eig::MatrixXf canonical_weight = canonical_field.replicate(1, 1);
    for (int i=0; i < canonical_weight.rows(); ++i) { // Determine weight based on thickness
        for (int j=0; j < canonical_weight.cols(); ++j){
            canonical_weight(i, j) = (canonical_field(i, j) <= -eta) ? 0.0f : 1.0f;
        }
    }

    eig::Vector3f twist = eig::Vector3f(0.0f, 0.0f, 0.0f);

    for (int iteration_count=0; iteration_count<maximum_iteration_count; ++iteration_count) {
        eig::Matrix3f matrix_A = eig::Matrix3f::Zero();
        eig::Vector3f vector_b = eig::Vector3f::Zero();

        eig::Matrix<float, 6, 1> twist3d;
        twist3d << twist(0), 0.0f, twist(1), 0.0f, twist(2), 0.0f;
        eig::Matrix4f twist_matrix3d = math::transformation_vector_to_matrix3d(twist3d);
        std::cout << twist_matrix3d << std::endl;
        eig::MatrixXf live_field = tsdf::generate_TSDF_2D(image_y_coordinate,
                                                          live_depth_image,
                                                          tsdf_generation_parameters.depth_unit_ratio,
                                                          tsdf_generation_parameters.camera_intrinsic_matrix,
                                                          tsdf_generation_parameters.camera_pose*twist_matrix3d,
                                                          tsdf_generation_parameters.array_offset,
                                                          tsdf_generation_parameters.field_size,
                                                          tsdf_generation_parameters.voxel_size,
                                                          tsdf_generation_parameters.narrow_band_width_voxels);
        if (iteration_count == 1) std::cout << live_field << std::endl;
        eig::MatrixXf live_weight = live_field.replicate(1, 1);
        for (int i=0; i < live_weight.rows(); ++i) { // Determine weight based on thickness
            for (int j=0; j < live_weight.cols(); ++j){
                live_weight(i, j) = (live_field(i, j) <= -eta) ? 0.0f : 1.0f;
            }
        }
        eig::Matrix<eig::Vector3f, eig::Dynamic, eig::Dynamic> live_gradient(canonical_field.rows(),
                                                                             canonical_field.cols());
        ropt::gradient_wrt_twist(live_field,
                                 twist,
                                 tsdf_generation_parameters.array_offset,
                                 tsdf_generation_parameters.voxel_size,
                                 live_gradient);

        for (int i=0; i < live_field.rows(); ++i) {
            for (int j=0; j < live_field.cols(); ++j) {
//                if (i==0 && j==0){
//                if (std::abs(live_gradient(i, j)(1)) > 0.0000001f) {
//                    std::cout << "i: " << i << " j: " << j << std::endl;
//                    std::cout << live_gradient(i, j) << "\n" << std::endl;
//                }
                matrix_A += live_gradient(i, j) * live_gradient(i, j).transpose();
                vector_b += (canonical_field(i, j) - live_field(i, j) + live_gradient(i, j).transpose() * twist)
                            * live_gradient(i, j);
//                if (i==0 && j==0) {
//                    std::cout << "c++ i: " << i << " j: " << j << std::endl;
//                    std::cout << "A: " << matrix_A << "\n"  << "b: " << vector_b << "\n"<< std::endl;
//                }
            }
        }

//        std::cout << "A: \n" << matrix_A << std::endl;
//        std::cout << "A.inverse: \n" << matrix_A.inverse() << std::endl;
//        std::cout << "b: \n" << vector_b << std::endl;

        float energy = .5f * (canonical_field.cwiseProduct(canonical_weight) -
                              live_field.cwiseProduct(live_weight)).array().pow(2.f).sum();
        eig::Vector3f twist_star = matrix_A.inverse() * vector_b;
        twist = twist + this->rate * (twist_star - twist);
        if (this->verbosity_parameters.print_per_iteration_info) {
            std::cout << "[ITERATION " << iteration_count << " COMPLETED]\n";
            if (this->verbosity_parameters.print_iteration_max_warp_update) {
                std::cout << " [optimize twist:" << twist_star.transpose() << "]\n";
                std::cout << " [twist:" << twist.transpose() << "]\n";
            }
            if (this->verbosity_parameters.print_iteration_energy) {
                std::cout << " [energy: " << energy << "]\n\n" << std::endl;
            }

        }
    }

    return twist;
}


}
