/*
 * test_sdf_2_sdf_optimizer.cpp
 *
 *  Created on: Jun 07, 2019
 *      Author: Fei Shan
 */

#define BOOST_TEST_MODULE test_sdf_2_sdf_optimizer

//stdlib
//_DEBUG
#include <iostream>

//libraries
#include <boost/test/unit_test.hpp>
#include <boost/python.hpp>
#include <Eigen/Eigen>

//local
#include "../src/tsdf/generator.hpp"
#include "../src/math/typedefs.hpp"
#include "../src/math/almost_equal.hpp"
#define TINYEXR_IMPLEMENTATION
#include "../src/image_io/tinyexr.h"

//test targets
#include "../src/rigid_optimization/sdf_weight.hpp"
#include "../src/rigid_optimization/sdf_gradient_wrt_transformation.hpp"
#include "../src/rigid_optimization/sdf_2_sdf_optimizer.hpp"

namespace eig = Eigen;
namespace ro = rigid_optimization;

eig::Matrix<unsigned short, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor> load_exr_helper(const char* frame_path) {
    float* out; // width * height * RGBA
    int width;
    int height;

    const char* err = nullptr; // or nullptr in C++11

    int ret = LoadEXR(&out, &width, &height, frame_path, &err);

    eig::Matrix<unsigned short, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> frame(height, width);

    if (ret != TINYEXR_SUCCESS) {
        if (err) {
            fprintf(stderr, "ERR : %s\n", err);
            FreeEXRErrorMessage(err); // release memory of error message.
        }
    } else {

        for (int i = 0; i < width * height; ++i) {
            unsigned short pixel = static_cast<unsigned short>(*(out+4*i));
            if (pixel == 0) { pixel = USHRT_MAX; }
            std::fill(frame.data()+i, frame.data()+i+1, pixel);
        }
        free(out); // relase memory of image data
    }
    return frame;
}

BOOST_AUTO_TEST_CASE(test_sdf_weight01) {
    eig::MatrixXf field(4, 4);
    field.setZero();
    float eta = -1.f;
    eig::MatrixXf expected_weight = field;
    eig::MatrixXf weight = ro::sdf_weight(field, eta);

    BOOST_REQUIRE(math::almost_equal_verbose(weight, expected_weight, 1e-14));
}

BOOST_AUTO_TEST_CASE(test_sdf_2_sdf_optimizer01) {
    const char* canonical_frame_path = "../tests/data/depth_000000.exr";
    const char* live_frame_path = "../tests/data/depth_000003.exr";

    eig::Matrix<unsigned short, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor> canonical_frame = load_exr_helper(canonical_frame_path);
    eig::Matrix<unsigned short, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor> live_frame = load_exr_helper(live_frame_path);

    eig::Matrix3f camera_intrinsic_matrix;
    camera_intrinsic_matrix << 570.3999633789062, 0., 320.,  // FX = 570.3999633789062 CX = 320.0
                               0., 570.3999633789062, 240.,  // FY = 570.3999633789062 CY = 240.0
                               0., 0., 1.;
    float voxel_size = 0.004f;
    int narrow_band_width_voxels = 2;
    math::Vector3i offset(-16, -16, 110);
    math::Vector3i field_shape(32, 32, 32);
    float eta = 0.01f;
    eig::Matrix4f camera_position;
    camera_position.setIdentity();

    ro::Sdf2SdfOptimizer<eig::Tensor<float, 3>, eig::Tensor<eig::Matrix<float, 6, 1>, 3>>::VerbosityParameters verbosity_parameters(true, true);

    tsdf::Parameters3d parameters(0.001f, //depth unit ratio
                                  camera_intrinsic_matrix, //projection matrix
                                  0.05f, //near clipping distance
                                  offset, //offset of scene from world origin
                                  field_shape, //dimensions of the voxel grid
                                  voxel_size, //voxel size
                                  narrow_band_width_voxels, //narrow band width
                                  tsdf::FilteringMethod::NONE
                                  );
    tsdf::Generator3d generator(parameters);
    eig::Tensor<float, 3> canonical_field = generator.generate(canonical_frame, camera_position);

    ro::Sdf2SdfOptimizer<eig::Tensor<float, 3>, eig::Tensor<eig::Matrix<float, 6, 1>, 3>> optimizer(0.5f,
                                                                                                    60,
                                                                                                    parameters,
                                                                                                    verbosity_parameters);
    optimizer.optimize(canonical_field,
                       live_frame,
                       eta,
                       camera_position);
}