/*
 * sdf2sdf_optimizer2d.hpp
 *
 *  Created on: Mar 05, 2019
 *      Author: Fei Shan
 */


#pragma once

//libraries
#include <Eigen/Eigen>

//local
#include "../tsdf/parameters.hpp"
#include "../tsdf/generator.hpp"

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

        Sdf2SdfOptimizer2d(
                float rate = 0.5f,
                int maximum_iteration_count = 60,
				tsdf::Parameters2d tsdf_generation_parameters = tsdf::Parameters2d(),
                VerbosityParameters verbosity_parameters = VerbosityParameters()
        );

        virtual ~Sdf2SdfOptimizer2d();

    eig::Matrix3f optimize(int image_y_coordinate,
                           const eig::MatrixXf canonical_field,
						   const eig::Matrix<unsigned short, eig::Dynamic, eig::Dynamic>& live_depth_image,
                           float eta = 0.01f,
						   const eig::Matrix4f& initial_camera_pose = eig::Matrix4f::Identity());

    private:
        const float rate = 0.5f;
        const int maximum_iteration_count = 60;
        const tsdf::Generator2d tsdf_generator;
        const Sdf2SdfOptimizer2d::VerbosityParameters verbosity_parameters;

};

}
