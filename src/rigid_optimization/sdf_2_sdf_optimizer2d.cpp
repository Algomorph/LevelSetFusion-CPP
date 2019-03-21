/*
 * sdf2sdf_optimizer2d.cpp
 *
 *  Created on: Mar 05, 2019
 *      Author: Fei Shan
 */

//libraries
#include <boost/python.hpp>

// local
#include "sdf2sdf_optimizer2d.hpp"
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

eig::Vector3f Sdf2SdfOptimizer2d::optimize(int image_y_coordinate,
        const eig::Matrix<unsigned short, eig::Dynamic, eig::Dynamic>& canonical_depth_image,
        const eig::Matrix<unsigned short, eig::Dynamic, eig::Dynamic>& live_depth_image,
        float depth_unit_ratio,
        const eig::Matrix3f& camera_intrinsic_matrix,
        const eig::Matrix4f& camera_pose = eig::Matrix4f::Identity(4, 4),
        const eig::Vector3i& array_offset =
            [] {eig::Vector3i default_offset; default_offset << -64, -64, 64; return default_offset;}(),
        int field_size = 128,
        float voxel_size = 0.004,
        int narrow_band_width_voxels = 20,
        float eta = 0.01f) {

    eig::MatrixXf canonical_field = tsdf::generate_TSDF_2D(image_y_coordinate,
                                                           canonical_depth_image,
                                                           depth_unit_ratio,
                                                           camera_intrinsic_matrix,
                                                           camera_pose,
                                                           array_offset,
                                                           field_size,
                                                           voxel_size,
                                                           narrow_band_width_voxels);

    eig::MatrixXf canonical_weight = canonical_field.replicate(1, 1);
    for (int i=0; i < canonical_weight.rows(); ++i) { // Determine weight based on thickness
        for (int j=0; j < canonical_weight.cols(); ++j){
            canonical_weight(i, j) = (canonical_weight(i, j) <= -eta) ? 0.0f : 1.0f;
        }
    }




    eig::Matrix3f matrix_A;
    eig::Vector3f vector_b;
    eig::Matrix<eig::Vector3f, eig::Dynamic, eig::Dynamic> live_gradient;
    eig::Vector3f twist = eig::Vector3f(0.0f, 0.0f, 0.0f);

    for (int iteration_count=0; iteration_count<maximum_iteration_count; ++iteration_count) {
        eig::MatrixXf live_field = tsdf::generate_TSDF_2D(image_y_coordinate,
                                                          live_depth_image,
                                                          depth_unit_ratio,
                                                          camera_intrinsic_matrix,
                                                          camera_pose,
                                                          array_offset,
                                                          field_size,
                                                          voxel_size,
                                                          narrow_band_width_voxels);

        eig::MatrixXf live_weight = live_field.replicate(1, 1);
        for (int i=0; i < live_weight.rows(); ++i) { // Determine weight based on thickness
            for (int j=0; j < live_weight.cols(); ++j){
                live_weight(i, j) = (live_weight(i, j) <= -eta) ? 0.0f : 1.0f;
            }
        }

        ropt::gradient_wrt_twist(live_field, twist, array_offset, voxel_size, live_gradient);

        for (int i=0; i < live_field.rows(); ++i) {
            for (int j=0; j < live_field.cols(); ++j) {
                matrix_A += live_gradient(i, j) * live_gradient(i, j).transpose();
                vector_b += (canonical_field(i, j) - live_field(i, j) + live_gradient(i, j).transpose() * twist)
                            * live_gradient(i, j);
            }
        }
        float energy = .5f * (canonical_field.cwiseProduct(canonical_weight) -
                              live_field.cwiseProduct(live_weight)).array().pow(2.f).sum();
        eig::Vector3f twist_star = matrix_A.inverse() * vector_b;
        twist = this->rate * (twist_star - twist);
        if (this->verbosity_parameters.print_per_iteration_info) {
            std::cout << "[ITERATION " << iteration_count << " COMPLETED]";
            if (this->verbosity_parameters.print_iteration_max_warp_update) {
                std::cout << " [optimize twist: " << twist_star << "]";
            }
            if (this->verbosity_parameters.print_iteration_energy) {
                std::cout << " [energy: " << energy << "]";
            }

        }
    }

    return twist;
}


}
