/*
 * common.cpp
 *
 *  Created on: Mar 26, 2019
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

#include "common.hpp"

bool read_image_helper(math::MatrixXus& depth_image, std::string filename) {
	std::string full_path = "test_data/" + filename;
	bool image_read = image_io::png::read_GRAY16(full_path.c_str(), depth_image);
	if (!image_read) {
		//are we running from the project root dir, maybe?
		std::string full_path = "tests/data/" + filename;
		image_read = image_io::png::read_GRAY16(full_path.c_str(), depth_image);
	}
	return image_read;
}



