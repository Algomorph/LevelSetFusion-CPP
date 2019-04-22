/*
 * generator.hpp
 *
 *  Created on: Apr 19, 2019
 *      Author: Gregory Kramida
 *   Copyright: 2019 Gregory Kramida
 *
 *   Licensed under the Apache License, Version 2.0 (the "License");
 *   you may not use this file except in compliance with the License.
 *   You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 *   Unless required by applicable law or agreed to in writing, software
 *   distributed under the License is distributed on an "AS IS" BASIS,
 *   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *   See the License for the specific language governing permissions and
 *   limitations under the License.
 */

#pragma once

//libraries
#include <Eigen/Dense>
#include <unsupported/Eigen/CXX11/Tensor>

//local
#include "../math/container_traits.hpp"
#include "../math/typedefs.hpp"



namespace tsdf {

enum class InterpolationMethod {
	NONE = 0,
//	BILINEAR_IMAGE_SPACE = 1,
//	BILINEAR_VOXEL_SPACE = 2,
	EWA_IMAGE_SPACE = 3,
	EWA_VOXEL_SPACE = 4,
	EWA_VOXEL_SPACE_INCLUSIVE = 5
};

template<typename ScalarContainer>
struct Parameters {
	typedef typename math::ContainerWrapper<ScalarContainer>::Coordinates Coordinates;
	typedef typename ScalarContainer::Scalar Scalar;
	typedef Eigen::Matrix<Scalar, 3, 3, Eigen::ColMajor> Mat3;
	Parameters(Scalar depth_unit_ratio = (Scalar)0.001,
			Mat3 projection_matrix = Mat3::Identity(),
			Scalar near_clipping_distance = (Scalar)0.05,
			Coordinates array_offset = Coordinates(-64),
			Coordinates field_shape = Coordinates(128),
			Scalar voxel_size = (Scalar)0.004,
			int narrow_band_width_voxels = 20,
			InterpolationMethod interpolation_method = InterpolationMethod::NONE,
			Scalar smoothing_factor = (Scalar)1.0
			);
	Scalar depth_unit_ratio = (Scalar)0.001; //meters
	Mat3 projection_matrix;
	Scalar near_clipping_distance = 0.05; //meters
	Coordinates array_offset = Coordinates(-64); //voxels
	Coordinates field_shape = Coordinates(128); //voxels
	Scalar voxel_size = 0.004; //meters
	int narrow_band_width_voxels = 20; //voxels
	InterpolationMethod interpolation_method = InterpolationMethod::NONE;
	Scalar smoothing_factor = (Scalar)1.0; // gaussian covariance scale for EWA
};

template<typename Generator, typename ScalarContainer>
class GeneratorCRTP {
public:
	GeneratorCRTP(const Parameters<ScalarContainer>& parameters);

	const Parameters<ScalarContainer> parameters;
	typedef typename ScalarContainer::Scalar ContainerScalar;

	ScalarContainer generate(
			const Eigen::Matrix<unsigned short, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>& depth_image,
			const Eigen::Matrix<ContainerScalar, 4, 4, Eigen::ColMajor>& camera_pose =
					Eigen::Matrix<ContainerScalar, 4, 4, Eigen::ColMajor>::Identity(),
			int image_y_coordinate = 0) const;
};

template<typename ScalarContainer>
class Generator {
};

template<typename Scalar>
class Generator<Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor> > :
		public GeneratorCRTP<Generator<Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor> >,
				Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>> {

public:
	typedef Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor> Mat;
	using GeneratorCRTP<Generator<Mat>, Mat>::GeneratorCRTP;
	typedef eig::Matrix<Scalar,4,1> Vec4;
	typedef eig::Matrix<Scalar,3,1> Vec3;
	typedef eig::Matrix<Scalar,2,1> Vec2;
	typedef eig::Matrix<Scalar,3,3> Mat3;
	typedef eig::Matrix<Scalar,2,2> Mat2;

	Mat generate__none(
			const Eigen::Matrix<unsigned short, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>& depth_image,
			const Eigen::Matrix<Scalar, 4, 4, Eigen::ColMajor>& camera_pose,
			int image_y_coordinate = 0) const;

	Mat generate__ewa_image_space(
					const Eigen::Matrix<unsigned short, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>& depth_image,
					const Eigen::Matrix<Scalar, 4, 4, Eigen::ColMajor>& camera_pose,
					int image_y_coordinate = 0) const;

	Mat generate__ewa_voxel_space(
				const Eigen::Matrix<unsigned short, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>& depth_image,
				const Eigen::Matrix<Scalar, 4, 4, Eigen::ColMajor>& camera_pose,
				int image_y_coordinate = 0) const;

	Mat generate__ewa_voxel_space_inclusive(
			const Eigen::Matrix<unsigned short, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>& depth_image,
			const Eigen::Matrix<Scalar, 4, 4, Eigen::ColMajor>& camera_pose,
			int image_y_coordinate = 0) const;
private:
	template <typename SamplingBoundsFunction, typename VoxelValueFunction>
	Mat generate__ewa_aux(SamplingBoundsFunction&& compute_sampling_bounds, VoxelValueFunction&& compute_voxel_value,
					const Eigen::Matrix<unsigned short, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>& depth_image,
					const Eigen::Matrix<Scalar, 4, 4, Eigen::ColMajor>& camera_pose,
					int image_y_coordinate) const;
};

template<typename Scalar>
class Generator<Eigen::Tensor<Scalar, 3, Eigen::ColMajor> > :
		public GeneratorCRTP<Generator<Eigen::Tensor<Scalar, 3, Eigen::ColMajor> >,
			Eigen::Tensor<Scalar, 3, Eigen::ColMajor> > {
public:
	typedef Eigen::Tensor<Scalar, 3, Eigen::ColMajor> Ts;
	using GeneratorCRTP<Generator<Ts>, Ts>::GeneratorCRTP;
	Generator(const Parameters<Eigen::Tensor<Scalar, 3, Eigen::ColMajor> >& parameters);
	typedef eig::Matrix<Scalar,4,1> Vec4;
	typedef eig::Matrix<Scalar,3,1> Vec3;
	typedef eig::Matrix<Scalar,2,1> Vec2;
	typedef eig::Matrix<Scalar,3,3> Mat3;
	typedef eig::Matrix<Scalar,2,2> Mat2;

	Ts generate__none(
			const Eigen::Matrix<unsigned short, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>& depth_image,
			const Eigen::Matrix<Scalar, 4, 4, Eigen::ColMajor>& camera_pose,
			int image_y_coordinate = 0) const;

	Ts generate__ewa_image_space(
				const Eigen::Matrix<unsigned short, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>& depth_image,
				const Eigen::Matrix<Scalar, 4, 4, Eigen::ColMajor>& camera_pose,
				int image_y_coordinate = 0) const;

	Ts generate__ewa_voxel_space(
				const Eigen::Matrix<unsigned short, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>& depth_image,
				const Eigen::Matrix<Scalar, 4, 4, Eigen::ColMajor>& camera_pose,
				int image_y_coordinate = 0) const;

	Ts generate__ewa_voxel_space_inclusive(
			const Eigen::Matrix<unsigned short, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>& depth_image,
			const Eigen::Matrix<Scalar, 4, 4, Eigen::ColMajor>& camera_pose,
			int image_y_coordinate = 0) const;
private:
template <typename SamplingBoundsFunction, typename VoxelValueFunction>
Ts generate__ewa_aux(SamplingBoundsFunction&& compute_sampling_bounds, VoxelValueFunction&& compute_voxel_value,
				const Eigen::Matrix<unsigned short, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>& depth_image,
				const Eigen::Matrix<Scalar, 4, 4, Eigen::ColMajor>& camera_pose,
				int image_y_coordinate) const;
};

typedef Parameters<eig::MatrixXf> Parameters2d;
typedef Parameters<math::Tensor3f> Parameters3d;
typedef Generator<eig::MatrixXf> Generator2d;
typedef Generator<math::Tensor3f> Generator3d;

}  // namespace tsdf
