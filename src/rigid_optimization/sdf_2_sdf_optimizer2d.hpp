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

    //TODO: (1) remove the canonical_depth_image -- instead accept a ready canonical field and alter usages
	//TODO: (2) change output to eig::Matrix4f
    //    in theory, strictly speaking, output (as well as initial_camera_pose) should be eig::Matrix3f for 2d case
    //    and eig::Matrix4f for 3d case
    // 2D:
	// |r1 r2 t_x|
	// |r3 r4 t_y|
	// | 0  0  1 |
    // 3D:
	// |r1 r2 r3 t_x|
	// |r4 r5 r6 t_y|
	// |r7 r8 r9 t_z|
	// | 0  0  0  0 |
    // see defn' of rotation matrices on Wikipedia if needed.
    // To simplify usage of tsdf generator (which always takes the 4x4 matrix), here we can also dumb down the
    // 3D matrix down (use r9 = 1, r3 = r6 = r7 = r8 = 0) and simply homogenize the (x,y) coordinates to (x,y,1,1)
    // whereever needed
    eig::Vector3f optimize(int image_y_coordinate,
    					   const eig::Matrix<unsigned short, eig::Dynamic, eig::Dynamic>& canonical_depth_image,
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
