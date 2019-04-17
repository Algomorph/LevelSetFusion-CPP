/*
 * test_data_convolution.hpp
 *
 *  Created on: Apr 16, 2019
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

#include <Eigen/Dense>
#include <unsupported/Eigen/CXX11/Tensor>

#include "../../src/math/typedefs.hpp"
namespace test_data {


static math::Tensor3v3f convolved_3d_vector_field = []{
		math::Tensor3v3f convolved_3d_vector_field(4,4,5);
		convolved_3d_vector_field.setValues(  // @formatter:off
		{{{math::Vector3f(4850.0f,4975.0f,5100.0f), math::Vector3f(11100.0f,11250.0f,11400.0f), math::Vector3f(18300.0f,18450.0f,18600.0f), math::Vector3f(25500.0f,25650.0f,25800.0f), math::Vector3f(13950.0f,14025.0f,14100.0f)},
		  {math::Vector3f(7140.0f,7290.0f,7440.0f), math::Vector3f(14904.0f,15084.0f,15264.0f), math::Vector3f(23544.0f,23724.0f,23904.0f), math::Vector3f(32184.0f,32364.0f,32544.0f), math::Vector3f(17532.0f,17622.0f,17712.0f)},
		  {math::Vector3f(8940.0f,9090.0f,9240.0f), math::Vector3f(17064.0f,17244.0f,17424.0f), math::Vector3f(25704.0f,25884.0f,26064.0f), math::Vector3f(34344.0f,34524.0f,34704.0f), math::Vector3f(18612.0f,18702.0f,18792.0f)},
		  {math::Vector3f(4770.0f,4845.0f,4920.0f), math::Vector3f(8892.0f,8982.0f,9072.0f), math::Vector3f(13212.0f,13302.0f,13392.0f), math::Vector3f(17532.0f,17622.0f,17712.0f), math::Vector3f(9486.0f,9531.0f,9576.0f)}},
		 {{math::Vector3f(6150.0f,6300.0f,6450.0f), math::Vector3f(13716.0f,13896.0f,14076.0f), math::Vector3f(22356.0f,22536.0f,22716.0f), math::Vector3f(30996.0f,31176.0f,31356.0f), math::Vector3f(16938.0f,17028.0f,17118.0f)},
		  {math::Vector3f(8964.0f,9144.0f,9324.0f), math::Vector3f(18360.0f,18576.0f,18792.0f), math::Vector3f(28728.0f,28944.0f,29160.0f), math::Vector3f(39096.0f,39312.0f,39528.0f), math::Vector3f(21276.0f,21384.0f,21492.0f)},
		  {math::Vector3f(11124.0f,11304.0f,11484.0f), math::Vector3f(20952.0f,21168.0f,21384.0f), math::Vector3f(31320.0f,31536.0f,31752.0f), math::Vector3f(41688.0f,41904.0f,42120.0f), math::Vector3f(22572.0f,22680.0f,22788.0f)},
		  {math::Vector3f(5922.0f,6012.0f,6102.0f), math::Vector3f(10908.0f,11016.0f,11124.0f), math::Vector3f(16092.0f,16200.0f,16308.0f), math::Vector3f(21276.0f,21384.0f,21492.0f), math::Vector3f(11502.0f,11556.0f,11610.0f)}},
		 {{math::Vector3f(6600.0f,6750.0f,6900.0f), math::Vector3f(14256.0f,14436.0f,14616.0f), math::Vector3f(22896.0f,23076.0f,23256.0f), math::Vector3f(31536.0f,31716.0f,31896.0f), math::Vector3f(17208.0f,17298.0f,17388.0f)},
		  {math::Vector3f(9504.0f,9684.0f,9864.0f), math::Vector3f(19008.0f,19224.0f,19440.0f), math::Vector3f(29376.0f,29592.0f,29808.0f), math::Vector3f(39744.0f,39960.0f,40176.0f), math::Vector3f(21600.0f,21708.0f,21816.0f)},
		  {math::Vector3f(11664.0f,11844.0f,12024.0f), math::Vector3f(21600.0f,21816.0f,22032.0f), math::Vector3f(31968.0f,32184.0f,32400.0f), math::Vector3f(42336.0f,42552.0f,42768.0f), math::Vector3f(22896.0f,23004.0f,23112.0f)},
		  {math::Vector3f(6192.0f,6282.0f,6372.0f), math::Vector3f(11232.0f,11340.0f,11448.0f), math::Vector3f(16416.0f,16524.0f,16632.0f), math::Vector3f(21600.0f,21708.0f,21816.0f), math::Vector3f(11664.0f,11718.0f,11772.0f)}},
		 {{math::Vector3f(3375.0f,3450.0f,3525.0f), math::Vector3f(7218.0f,7308.0f,7398.0f), math::Vector3f(11538.0f,11628.0f,11718.0f), math::Vector3f(15858.0f,15948.0f,16038.0f), math::Vector3f(8649.0f,8694.0f,8739.0f)},
		  {math::Vector3f(4842.0f,4932.0f,5022.0f), math::Vector3f(9612.0f,9720.0f,9828.0f), math::Vector3f(14796.0f,14904.0f,15012.0f), math::Vector3f(19980.0f,20088.0f,20196.0f), math::Vector3f(10854.0f,10908.0f,10962.0f)},
		  {math::Vector3f(5922.0f,6012.0f,6102.0f), math::Vector3f(10908.0f,11016.0f,11124.0f), math::Vector3f(16092.0f,16200.0f,16308.0f), math::Vector3f(21276.0f,21384.0f,21492.0f), math::Vector3f(11502.0f,11556.0f,11610.0f)},
		  {math::Vector3f(3141.0f,3186.0f,3231.0f), math::Vector3f(5670.0f,5724.0f,5778.0f), math::Vector3f(8262.0f,8316.0f,8370.0f), math::Vector3f(10854.0f,10908.0f,10962.0f), math::Vector3f(5859.0f,5886.0f,5913.0f)}}}); // @formatter:on
		return convolved_3d_vector_field;
}();

static math::Tensor3v3f convolved_x_3d_vector_field = []{
		math::Tensor3v3f convolved_x_3d_vector_field(4,4,5);
		convolved_x_3d_vector_field.setValues(  // @formatter:off
		{{{math::Vector3f(14.0f,19.0f,24.0f), math::Vector3f(254.0f,259.0f,264.0f), math::Vector3f(494.0f,499.0f,504.0f), math::Vector3f(734.0f,739.0f,744.0f), math::Vector3f(974.0f,979.0f,984.0f)},
		  {math::Vector3f(74.0f,79.0f,84.0f), math::Vector3f(314.0f,319.0f,324.0f), math::Vector3f(554.0f,559.0f,564.0f), math::Vector3f(794.0f,799.0f,804.0f), math::Vector3f(1034.0f,1039.0f,1044.0f)},
		  {math::Vector3f(134.0f,139.0f,144.0f), math::Vector3f(374.0f,379.0f,384.0f), math::Vector3f(614.0f,619.0f,624.0f), math::Vector3f(854.0f,859.0f,864.0f), math::Vector3f(1094.0f,1099.0f,1104.0f)},
		  {math::Vector3f(194.0f,199.0f,204.0f), math::Vector3f(434.0f,439.0f,444.0f), math::Vector3f(674.0f,679.0f,684.0f), math::Vector3f(914.0f,919.0f,924.0f), math::Vector3f(1154.0f,1159.0f,1164.0f)}},
		 {{math::Vector3f(30.0f,36.0f,42.0f), math::Vector3f(318.0f,324.0f,330.0f), math::Vector3f(606.0f,612.0f,618.0f), math::Vector3f(894.0f,900.0f,906.0f), math::Vector3f(1182.0f,1188.0f,1194.0f)},
		  {math::Vector3f(102.0f,108.0f,114.0f), math::Vector3f(390.0f,396.0f,402.0f), math::Vector3f(678.0f,684.0f,690.0f), math::Vector3f(966.0f,972.0f,978.0f), math::Vector3f(1254.0f,1260.0f,1266.0f)},
		  {math::Vector3f(174.0f,180.0f,186.0f), math::Vector3f(462.0f,468.0f,474.0f), math::Vector3f(750.0f,756.0f,762.0f), math::Vector3f(1038.0f,1044.0f,1050.0f), math::Vector3f(1326.0f,1332.0f,1338.0f)},
		  {math::Vector3f(246.0f,252.0f,258.0f), math::Vector3f(534.0f,540.0f,546.0f), math::Vector3f(822.0f,828.0f,834.0f), math::Vector3f(1110.0f,1116.0f,1122.0f), math::Vector3f(1398.0f,1404.0f,1410.0f)}},
		 {{math::Vector3f(48.0f,54.0f,60.0f), math::Vector3f(336.0f,342.0f,348.0f), math::Vector3f(624.0f,630.0f,636.0f), math::Vector3f(912.0f,918.0f,924.0f), math::Vector3f(1200.0f,1206.0f,1212.0f)},
		  {math::Vector3f(120.0f,126.0f,132.0f), math::Vector3f(408.0f,414.0f,420.0f), math::Vector3f(696.0f,702.0f,708.0f), math::Vector3f(984.0f,990.0f,996.0f), math::Vector3f(1272.0f,1278.0f,1284.0f)},
		  {math::Vector3f(192.0f,198.0f,204.0f), math::Vector3f(480.0f,486.0f,492.0f), math::Vector3f(768.0f,774.0f,780.0f), math::Vector3f(1056.0f,1062.0f,1068.0f), math::Vector3f(1344.0f,1350.0f,1356.0f)},
		  {math::Vector3f(264.0f,270.0f,276.0f), math::Vector3f(552.0f,558.0f,564.0f), math::Vector3f(840.0f,846.0f,852.0f), math::Vector3f(1128.0f,1134.0f,1140.0f), math::Vector3f(1416.0f,1422.0f,1428.0f)}},
		 {{math::Vector3f(27.0f,30.0f,33.0f), math::Vector3f(171.0f,174.0f,177.0f), math::Vector3f(315.0f,318.0f,321.0f), math::Vector3f(459.0f,462.0f,465.0f), math::Vector3f(603.0f,606.0f,609.0f)},
		  {math::Vector3f(63.0f,66.0f,69.0f), math::Vector3f(207.0f,210.0f,213.0f), math::Vector3f(351.0f,354.0f,357.0f), math::Vector3f(495.0f,498.0f,501.0f), math::Vector3f(639.0f,642.0f,645.0f)},
		  {math::Vector3f(99.0f,102.0f,105.0f), math::Vector3f(243.0f,246.0f,249.0f), math::Vector3f(387.0f,390.0f,393.0f), math::Vector3f(531.0f,534.0f,537.0f), math::Vector3f(675.0f,678.0f,681.0f)},
		  {math::Vector3f(135.0f,138.0f,141.0f), math::Vector3f(279.0f,282.0f,285.0f), math::Vector3f(423.0f,426.0f,429.0f), math::Vector3f(567.0f,570.0f,573.0f), math::Vector3f(711.0f,714.0f,717.0f)}}}); // @formatter:on
		return convolved_x_3d_vector_field;
}();

}  // namespace test_data

