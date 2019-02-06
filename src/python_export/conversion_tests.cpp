/*
 * converter_tests.cpp
 *
 *  Created on: Feb 6, 2019
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

//stdlib
#include <iostream>
#include <vector>

//librarires
#include <boost/python.hpp>

//local
#include "conversion_tests.hpp"
#include <unsupported/Eigen/CXX11/Tensor>
namespace eig = Eigen;


namespace bp = boost::python;

namespace python_export{

template<typename T>
std::vector<T> arange(T start, T stop, T step = 1) {
    std::vector<T> values;
    for (T value = start; value < stop; value += step)
        values.push_back(value);
    return values;
}

eig::Tensor<float,3> return_input_F3(eig::Tensor<float,3> input){
	return input;
}

eig::Tensor<float,4> return_input_F4(eig::Tensor<float,4> input){
	return input;
}

eig::Tensor<float,4, eig::RowMajor> return_tensor_F4RM(){
	std::vector<float> v = arange(1.0f,49.0f);
	return Eigen::TensorMap<Eigen::Tensor<float, 4, Eigen::RowMajor>>(&v[0], 2, 4, 2, 3);
}

eig::Tensor<float,3> scale(eig::Tensor<float,3> a, float factor){
	return a * factor;
}

eig::Tensor<float,3> add_constant(eig::Tensor<float,3> a, float constant){
	return a + constant;
}

eig::Tensor<float,3> add_tensors(eig::Tensor<float,3> a, eig::Tensor<float,3> b){
	return a + b;
}

void export_conversion_tests(){
	bp::def("return_input_f3", &return_input_F3,"Returns the input 3-dimensional tensor", bp::args("input"));
	bp::def("return_input_f4", &return_input_F4,"Returns the input 4-dimensional tensor", bp::args("input"));
	bp::def("return_tensor_f4rm", &return_tensor_F4RM,"Returns a 4-dimensional tensor of dimensions (2, 4, 2, 3) "
			"containing in range [1,48].");
	bp::def("scale", &scale,"Scales the input 3-dimensional tensor by the given factor",
			bp::args("a","factor"));
	bp::def("add_constant", &add_constant,
			"Adds the given constant to each element of the provided 3-dimensional tensor", bp::args("a"),"constant");
	bp::def("add_tensors", &add_tensors,"Adds the two provided 3-dimensional tensors "
			"and returns the result.", bp::args("a","b"));
}

}//namespace python_export

