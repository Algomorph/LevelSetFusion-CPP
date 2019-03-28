//  ================================================================
//  Created by Fei Shan on 03/15/19.
//  ================================================================

//stdlib
#include <iostream>

#define BOOST_TEST_MODULE test_rigid_optimization // NB:has to appear before the boost include

//libraries
#include <boost/test/unit_test.hpp>
#include <boost/python.hpp>
#include <Eigen/Eigen>

//local
#include "../src/math/assessment.hpp"

//test data
#include "data/test_data_sdf_2_sdf_optimizer.hpp"

//test targets
#include "../src/rigid_optimization/sdf_2_sdf_optimizer2d.hpp"
#include "../src/rigid_optimization/sdf_gradient_wrt_transformation2d.hpp"


namespace tt = boost::test_tools;
namespace bp = boost::python;
namespace eig = Eigen;
namespace ropt = rigid_optimization;

BOOST_AUTO_TEST_CASE(test_sdf2sdf_optimizer01) {
    // corresponds to test_construction-and_operation in python code (test_sdf_2_sdf_optimizer2d.py)
    using namespace math;

    int image_y_coordinate = 240;
    float depth_unit_ratio = 0.001f;
    eig::Matrix3f camera_intrinsic_matrix;
    camera_intrinsic_matrix <<
        570.3999633789062f, 0.f, 320.f,
        0.f, 570.3999633789062f, 240.f,
        0.f, 0.f, 1.f;
    eig::Matrix4f camera_pose = eig::MatrixXf::Identity(4, 4);
    eig::Vector3i array_offset = eig::Vector3i(-16, -16, 93); // 93.4375
    int field_size = 32;
    float voxel_size = 0.004f;
    int narrow_band_width_voxels = 2;

    float gaussian_covariance_scale = 1.0f;

    ropt::Sdf2SdfOptimizer2d optimizer(0.5f, // eta
                                       60, //max iteration count
                                       ropt::Sdf2SdfOptimizer2d::VerbosityParameters(false, false));

    eig::Vector3f twist_out = optimizer.optimize(image_y_coordinate,
                                                 canonical_depth_image,
                                                 live_depth_image,
                                                 depth_unit_ratio,
                                                 camera_intrinsic_matrix,
                                                 camera_pose,
                                                 array_offset,
                                                 field_size,
                                                 voxel_size,
                                                 narrow_band_width_voxels,
                                                 gaussian_covariance_scale);
    BOOST_REQUIRE((twist - twist_out).isZero(1e-5));
}



