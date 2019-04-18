/*
 * pretty_printers.hpp
 *
 *  Created on: Apr 5, 2019
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

//stdlib
#include <iostream>
#include <utility>

//libraries
#include <Eigen/Dense>
#include <unsupported/Eigen/CXX11/Tensor>

//local
#include "../math/typedefs.hpp"

namespace eig = Eigen;

template<typename Scalar>
std::ostream& operator<<(std::ostream& os, const eig::Tensor<Scalar, 3, eig::ColMajor>& tensor)
		{
	std::ios_base::fmtflags original_flags(os.flags());
	os.precision(6);
	os << std::fixed;
	for (int z = 0; z < tensor.dimension(2); z++) {
		for (int y = 0; y < tensor.dimension(1); y++) {
			for (int x = 0; x < tensor.dimension(0); x++) {
				os << tensor(x, y, z) << "| ";
			}
			os << std::endl;
		}
		os << std::endl;
	}
	os.flags(original_flags);
	return os;
}

namespace console {

template<typename PrintElementInitializerFunction, typename Scalar>
static inline
void print_initializer_list_aux(std::ostream& os, const eig::Tensor<Scalar, 3, eig::ColMajor>& tensor,
		PrintElementInitializerFunction&& print_element,
		bool use_scientific = false, int precision = 4, bool elements_on_new_lines = false,
		char bracket_open = '{', char bracket_close = '}'){
	std::ios_base::fmtflags original_flags(os.flags());
	if (!use_scientific) {
		os.precision(precision);
		os << std::fixed;
	}
	os << bracket_open;
	for (int x = 0; x < tensor.dimension(0); x++) {
		if (x > 0)
			os << " ";
		os << bracket_open;
		for (int y = 0; y < tensor.dimension(1); y++) {
			if (y > 0)
				os << "  ";
			os << bracket_open;
			for (int z = 0; z < tensor.dimension(2) - 1; z++) {
				if (elements_on_new_lines && z > 0){
					os << "   ";
				}
				std::forward<PrintElementInitializerFunction>(print_element)(os, tensor(x, y, z));
				os << ", ";
				if (elements_on_new_lines) {
					os << std::endl;
				}
			}
			if (elements_on_new_lines){
				os << "   ";
			}
			std::forward<PrintElementInitializerFunction>(print_element)(os, tensor(x, y, tensor.dimension(2) - 1));
			os << bracket_close;
			if (y == tensor.dimension(1) - 1) {
				os << "";
			} else {
				os << "," << std::endl;
			}
		}
		if (x == tensor.dimension(0) - 1) {
			os << bracket_close;
		} else {
			os << bracket_close << "," << std::endl;
		}
	}
	os << bracket_close << std::endl;
	os.flags(original_flags);
}

template<typename PrintElementInitializerFunction, typename Scalar>
static inline
void print_initializer_list_py_aux(std::ostream& os, const eig::Tensor<Scalar, 3, eig::ColMajor>& tensor,
		PrintElementInitializerFunction&& print_element,
		bool use_scientific = false, int precision = 4, bool elements_on_new_lines = false) {
	print_initializer_list_aux(
			os, tensor, print_element, use_scientific, precision, elements_on_new_lines, '[',']');
}

void print_initializer_list(std::ostream& os, const math::Tensor3f& tensor,
		bool use_scientific = false, int precision = 4) {
	print_initializer_list_aux(os, tensor, [](std::ostream& _os, float v) -> void {
		_os << v << "f";
	}, use_scientific, precision);
}

void print_initializer_list(std::ostream& os, const math::Tensor3v3f& tensor,
		bool use_scientific = false, int precision = 4) {
	print_initializer_list_aux(os, tensor, [](std::ostream& _os, const math::Vector3f& v) -> void {
		_os << "math::Vector3f(" << v.x << "f, " << v.y << "f, " << v.z << "f)";
	}, use_scientific, precision);
}

//TODO: flip dimensions for py printing, i.e equiv. of np.moveaxis(x, (0,1,2),(2,1,0))
void print_initializer_list_py(std::ostream& os, const math::Tensor3v3f& tensor,
		bool use_scientific = false, int precision = 4) {
	print_initializer_list_py_aux(os, tensor, [](std::ostream& _os, const math::Vector3f& v) -> void {
		_os << "[" << v.x << ", " << v.y << ", " << v.z << "]";
	}, use_scientific, precision, true);
}

template<typename PrintElementInitializerFunction, typename Scalar>
void print_initializer_list_aux(std::ostream& os,
		const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>& matrix,
		PrintElementInitializerFunction&& print_function,
		bool use_scientific = false, int precision = 4) {
	std::ios_base::fmtflags original_flags(os.flags());
	if (!use_scientific) {
		os.precision(precision);
		os << std::fixed;
	}
	int last_x = matrix.cols() - 1;
	for (int y = 0; y < matrix.rows(); y++) {
		for (int x = 0; x < last_x; x++) {
			std::forward<PrintElementInitializerFunction>(print_function)(os, matrix(y, x));
			os << ", ";
		}
		std::forward<PrintElementInitializerFunction>(print_function)(os, matrix(y, last_x));
		if (y == matrix.rows() - 1) {
			os << ";";
		} else {
			os << "," << std::endl;
		}
	}
	os.flags(original_flags);
}

void print_initializer_list(std::ostream& os, const eig::MatrixXf& matrix, bool use_scientific = false, int precision =
		4)
		{
	print_initializer_list_aux(os, matrix, [](std::ostream& _os, float s)-> void {
		_os << s;
	}, use_scientific, precision);
}

void print_initializer_list(std::ostream& os, const math::MatrixXv2f& matrix, bool use_scientific = false,
		int precision = 4)
		{
	print_initializer_list_aux(os, matrix, [](std::ostream& _os, const math::Vector2f& v)-> void {
		_os << "math::Vector2f(" << v.x << "f, " << v.y << "f)";
	}, use_scientific, precision);
}

void print_initializer_list(std::ostream& os, const math::MatrixXm2f& matrix, bool use_scientific = false,
		int precision = 4)
		{
	print_initializer_list_aux(os, matrix, [](std::ostream& _os, const math::Matrix2f& m)-> void {
		_os << "math::Matrix2f(" << m.xy00 << "f, " << m.xy01 << "f, " << m.xy10 << "f, " << m.xy11 << "f)";
	}, use_scientific, precision);
}

} //end namespace console

