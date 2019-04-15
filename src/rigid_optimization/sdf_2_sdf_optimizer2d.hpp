/*
 * sdf2sdf_optimizer2d.cpp
 *
 *  Created on: Mar 05, 2019
 *      Author: Fei Shan
 */


#pragma once

//libraries
#include <Eigen/Eigen>

//local
#include "../math/tensor_operations.hpp"

namespace eig = Eigen;

namespace rigid_optimization {

class Sdf2SdfOptimizer2d {
    public:
        struct VerbosityParameters {
            VerbosityParameters(bool print_iteration_max_warp_update = false,
                                bool print_iteration_energy = false);
            //per-iteration parameters
            const bool print_iteration_max_warp_update = false;
            const bool print_iteration_energy = false;
            const bool print_per_iteration_info = false;
        };

        struct TSDFGenerationParameters {
            TSDFGenerationParameters(float depth_unit_ratio = 0.001f,
                                     eig::Matrix3f camera_intrinsic_matrix = []
                                        {eig::Matrix3f camera_intrinsic_matrix;
                                            camera_intrinsic_matrix <<
                                            570.3999633789062f, 0.f, 320.f,
                                            0.f, 570.3999633789062f, 240.f,
                                            0.f, 0.f, 1.f;
                                            return camera_intrinsic_matrix;}(),
                                     eig::Matrix4f camera_pose = eig::MatrixXf::Identity(4, 4),
                                     eig::Vector3i array_offset = eig::Vector3i(-16, -16, 93), // 93.4375
                                     int field_size = 32,
                                     float voxel_size = 0.004f,
                                     int narrow_band_width_voxels = 2);
            const float depth_unit_ratio;
            const eig::Matrix3f camera_intrinsic_matrix;
            const eig::Matrix4f camera_pose;
            const eig::Vector3i array_offset;
            int field_size;
            float voxel_size;
            int narrow_band_width_voxels;
        };

        Sdf2SdfOptimizer2d(
                float rate = 0.5f,
                int maximum_iteration_count = 60,
                VerbosityParameters verbosity_parameters = VerbosityParameters()
        );

        virtual ~Sdf2SdfOptimizer2d();

    eig::Vector3f optimize(int image_y_coordinate,
                           const eig::Matrix<unsigned short, eig::Dynamic, eig::Dynamic>& canonical_depth_image,
                           const eig::Matrix<unsigned short, eig::Dynamic, eig::Dynamic>& live_depth_image,
                           const TSDFGenerationParameters tsdf_generation_parameters,
                           float eta);

    private:
        const float rate = 0.5f;
        const int maximum_iteration_count = 60;
        Sdf2SdfOptimizer2d::VerbosityParameters verbosity_parameters;

};

}