/*
 * extent.hpp
 *
 *  Created on: Apr 22, 2019
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

namespace math{
struct Extent2d{
	int x_start = 0;
	int x_end = 0;
	int y_start = 0;
	int y_end = 0;

	int area(){
		return (y_end - y_start) * (x_end - x_start);
	}

	bool empty(){
		return this->area() == 0;
	}
};

}//namespace math


