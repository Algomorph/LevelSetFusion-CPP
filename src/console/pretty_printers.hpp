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

#include <iostream>
#include "../math/typedefs.hpp"

std::ostream& operator<<(std::ostream& os, const math::Tensor3f& tensor)
{
	std::ios_base::fmtflags original_flags(os.flags());
	os.precision(6);
	os << std::fixed;
	for(int z = 0; z < tensor.dimension(2); z++){
		for(int y = 0; y < tensor.dimension(1); y++){
			for(int x =0; x < tensor.dimension(0); x++){
				os << tensor(x,y,z) << " ";
			}
			os << std::endl;
		}
		os << std::endl;
	}
	os.flags(original_flags);
    return os;
}
namespace console{
void print_initializer_list(std::ostream& os, const math::Tensor3f& tensor, int precision = 4)
{
	std::ios_base::fmtflags original_flags(os.flags());
	os.precision(precision);
	os << std::fixed;
	os << "{";
	for(int x = 0; x < tensor.dimension(0); x++){
		if(x > 0)
			os << " ";
		os << "{";
		for(int y = 0; y < tensor.dimension(1); y++){
			if(y > 0)
				os << "  ";
			os << "{";
			for(int z =0; z < tensor.dimension(2)-1; z++){
				os << tensor(x,y,z) << "f, ";
			}
			os << tensor(x,y,tensor.dimension(2)-1) << "f}";
			if(y == tensor.dimension(1)-1){
				os << "";
			}else{
				os << "," << std::endl;
			}
		}
		if(x == tensor.dimension(0)-1){
			os << "}";
		}else{
			os << "}," << std::endl;
		}
	}
	os << "}" << std::endl;
	os.flags(original_flags);
}
} //end namespace console

