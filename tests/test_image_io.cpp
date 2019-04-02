
/*
 * test_image_io.cpp
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
#define BOOST_TEST_MODULE test_image_io

//libraries
#include <boost/test/unit_test.hpp>

//test data
#include "data/test_data_image_io.hpp"

//test targets
#include "../src/math/typedefs.hpp"
#include "../src/image_io/png_eigen.hpp"
#include "common.hpp"

BOOST_AUTO_TEST_CASE(test_image_read01) {
	math::MatrixXus depth_image;
	bool image_read = read_image_helper(depth_image, "zigzag_depth_00064.png");
	BOOST_REQUIRE(image_read);
	BOOST_REQUIRE_EQUAL(depth_image.rows(), 480);
	BOOST_REQUIRE_EQUAL(depth_image.cols(), 640);
	BOOST_REQUIRE_EQUAL(depth_image(0, 0), (unsigned short )1997);
	BOOST_REQUIRE_EQUAL(depth_image(479, 0), (unsigned short )1997);
	BOOST_REQUIRE_EQUAL(depth_image(479, 639), (unsigned short ) 5154);
	math::MatrixXus sample = depth_image.block(40, 60, 1, 20);
	BOOST_REQUIRE(sample.isApprox(test_data::depth_00064_sample));
}

