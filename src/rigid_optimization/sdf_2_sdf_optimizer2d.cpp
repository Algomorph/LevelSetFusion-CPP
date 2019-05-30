/*
 * sdf2sdf_optimizer2d.cpp
 *
 *  Created on: Mar 05, 2019
 *      Author: Fei Shan
 */

//libraries
#include <boost/python.hpp>
#include <iostream>

// local
#include "../math/transformation.hpp"
#include "sdf_2_sdf_optimizer2d.hpp"
#include "sdf_gradient_wrt_transformation2d.hpp"

namespace ropt = rigid_optimization;
namespace eig = Eigen;

namespace rigid_optimization {

Sdf2SdfOptimizer2d::Sdf2SdfOptimizer2d(
		float rate,
		int maximum_iteration_count,
		tsdf::Parameters2d tsdf_generation_parameters,
		VerbosityParameters verbosity_parameters) :
		rate(rate),
				maximum_iteration_count(maximum_iteration_count),
				tsdf_generator(tsdf_generation_parameters),
				verbosity_parameters(verbosity_parameters)
{
}
;

Sdf2SdfOptimizer2d::~Sdf2SdfOptimizer2d()
{
}
;

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
}
;

eig::Matrix3f Sdf2SdfOptimizer2d::optimize(int image_y_coordinate,
		const eig::MatrixXf canonical_field,
		const eig::Matrix<unsigned short, eig::Dynamic, eig::Dynamic>& live_depth_image,
		float eta,
		const eig::Matrix4f& initial_camera_pose) {

	eig::MatrixXf canonical_weight = canonical_field.replicate(1, 1);
	for (int i = 0; i < canonical_weight.rows(); ++i) { // Determine weight based on thickness
		for (int j = 0; j < canonical_weight.cols(); ++j) {
			canonical_weight(i, j) = (canonical_field(i, j) <= -eta) ? 0.0f : 1.0f;
		}
	}

	eig::Vector3f twist = eig::Vector3f(0.0f, 0.0f, 0.0f);

	for (int iteration_count = 0; iteration_count < maximum_iteration_count; ++iteration_count) {
		eig::Matrix3f matrix_A = eig::Matrix3f::Zero(); // from sdf2sdf paper,
														// used for calculating the optimal transformation.
		eig::Vector3f vector_b = eig::Vector3f::Zero(); // from sdf2sdf paper,
														// used to calculating the optimal transformation.

		eig::Matrix<float, 6, 1> twist3d;
		twist3d << twist(0), 0.0f, twist(1), 0.0f, twist(2), 0.0f;
		eig::Matrix4f twist_matrix3d = math::transformation_vector_to_matrix3d(twist3d);
		eig::MatrixXf live_field = this->tsdf_generator.generate(live_depth_image,
				twist_matrix3d,
				image_y_coordinate);
		eig::MatrixXf live_weight = live_field.replicate(1, 1);
		for (int j = 0; j < live_weight.cols(); ++j) { // Determine weight based on thickness
			for (int i = 0; i < live_weight.rows(); ++i) {
				live_weight(i, j) = (live_field(i, j) <= -eta) ? 0.0f : 1.0f;
			}
		}
		eig::Matrix<eig::Vector3f, eig::Dynamic, eig::Dynamic> live_gradient(canonical_field.rows(),
				canonical_field.cols());
		eig::Vector3i offset(tsdf_generator.parameters.array_offset.x,
				0,
				tsdf_generator.parameters.array_offset.y);
		ropt::gradient_wrt_twist(live_field,
				twist,
				offset,
				tsdf_generator.parameters.voxel_size,
				live_gradient);

		for (int j = 0; j < live_field.cols(); ++j) {
			for (int i = 0; i < live_field.rows(); ++i) {
				matrix_A += live_gradient(i, j) * live_gradient(i, j).transpose();
				vector_b += (canonical_field(i, j) - live_field(i, j) + live_gradient(i, j).transpose() * twist)
						* live_gradient(i, j);
			}
		}

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


    eig::Matrix3f twist_matrix2d = math::transformation_vector_to_matrix2d(twist);
	return twist_matrix2d;
}

}
