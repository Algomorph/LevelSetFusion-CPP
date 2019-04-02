/*
 * test_data_tsdf.hpp
 *
 *  Created on: Jan 30, 2019
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

#include <unsupported/Eigen/CXX11/Tensor>
#include <Eigen/Eigen>

#include "../../src/math/tensor_operations.hpp"
#include "../src/math/typedefs.hpp"

namespace eig = Eigen;

namespace test_data {
static eig::Matrix<unsigned short, eig::Dynamic, eig::Dynamic> depth_image_region = [] {
	eig::Matrix<unsigned short, eig::Dynamic, eig::Dynamic> depth_image_region(3,18);
	depth_image_region <<
	3233, 3246, 3243, 3256, 3253, 3268, 3263, 3279, 3272, 3289, 3282, 3299, 3291, 3308, 3301, 3317, 3310, 3326,
	3233, 3246, 3243, 3256, 3253, 3268, 3263, 3279, 3272, 3289, 3282, 3299, 3291, 3308, 3301, 3317, 3310, 3326,
	3233, 3246, 3243, 3256, 3253, 3268, 3263, 3279, 3272, 3289, 3282, 3299, 3291, 3308, 3301, 3317, 3310, 3326;
	return depth_image_region;
}();

static eig::MatrixXf out_sdf_field =
		[] {
			eig::MatrixXf out_sdf_field(16,16);
			out_sdf_field <<
			0.8667111f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
			0.7550358f, 0.9348034f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
			0.65500134f, 0.7687389f, 0.88503355f, 0.93358153f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
			0.5375206f, 0.61014885f, 0.78497523f, 0.8335411f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
			0.43746826f, 0.5100786f, 0.674951f, 0.73348874f, 0.949794f, 0.98340505f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
			0.31999943f, 0.36687848f, 0.5749106f, 0.63345426f, 0.8497476f, 0.87506765f, 0.99062914f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
			0.20832418f, 0.26682612f, 0.47484037f, 0.53340787f, 0.70737f, 0.7687866f, 0.82769984f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
			0.10827779f, 0.16677974f, 0.37479398f, 0.42007563f, 0.6072938f, 0.6687402f, 0.7276177f, 0.89989895f, 0.9251892f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
			0.0082314f, 0.06673931f, 0.27474162f, 0.3200173f, 0.44060943f, 0.5549907f, 0.5835771f, 0.7998585f, 0.82513684f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
			-0.09181499f, -0.03331303f, 0.13484357f, 0.2062857f, 0.2776563f, 0.4499614f, 0.4835486f, 0.69980615f, 0.72510237f, 0.904864f, 0.9593963f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
			-0.19185542f, -0.1449704f, -0.03123283f, 0.10622143f, 0.17758606f, 0.349915f, 0.38350818f, 0.5997657f, 0.62005514f, 0.7375359f, 0.8593678f, 0.9101092f, 1.0f, 1.0f, 1.0f, 1.0f,
			-0.29189584f, -0.24503468f, -0.13129114f, -0.01502037f, 0.03356337f, 0.24986266f, 0.28344986f, 0.4997134f, 0.5200147f, 0.6374776f, 0.752449f, 0.766921f, 0.9831607f, 1.0f, 1.0f, 1.0f,
			-0.4301488f, -0.3625035f, -0.28989312f, -0.12502669f, -0.06648897f, 0.14979838f, 0.18342136f, 0.3573179f, 0.41248795f, 0.47016737f, 0.64991707f, 0.6668686f, 0.88312024f, 0.9000837f, 1.0f, 1.0f,
			-0.5874693f, -0.47997233f, -0.43309924f, -0.22507308f, -0.16653536f, 0.04975795f, 0.07506608f, 0.19062756f, 0.30499098f, 0.32528636f, 0.5498647f, 0.5668461f, 0.7830798f, 0.79754585f, 0.90939397f, 1.0f,
			-0.6875276f, -0.5800306f, -0.53313965f, -0.3251314f, -0.26658174f, -0.09260177f, -0.03118515f, 0.09058117f, 0.20493864f, 0.22523998f, 0.44983622f, 0.46679375f, 0.63987964f, 0.69065684f, 0.80935353f, 0.919944f,
			-0.84488386f, -0.69169396f, -0.63319796f, -0.4251897f, -0.3666341f, -0.19269584f, -0.13125537f, -0.07237195f, 0.09989738f, 0.12519954f, 0.3497779f, 0.3667414f, 0.5397975f, 0.59059256f, 0.6426274f, 0.8165538f;
			return out_sdf_field;
		}();

static math::MatrixXus depth_00064_sample = [] {
	math::MatrixXus depth_00064_sample(1,20);
	depth_00064_sample << 2121, 2126, 2124, 2123, 2128, 2133, 2138, 2130, 2135, 2140, 2145,
	2147, 2142, 2147, 2152, 2158, 2150, 2155, 2160, 2165;
	return depth_00064_sample;

}();

static eig::MatrixXf out_sdf_chunk =
		[] {
			eig::MatrixXf out_sdf_chunk(16,16);
			out_sdf_chunk <<
			-0.04320666f, 0.002749264f, 0.08112564f, 0.16363336f, 0.22640525f, 0.31301898f, 0.40417385f, 0.47360805f, 0.5570359f, 0.6488211f, 0.7187366f, 0.80698574f, 0.9093962f, 0.9763605f, 1.0f, 1.0f,
			-0.12583135f, -0.06246939f, 0.013691931f, 0.09301155f, 0.16130133f, 0.24867652f, 0.33713576f, 0.400155f, 0.48867163f, 0.58037335f, 0.6434604f, 0.73981655f, 0.8347153f, 0.8992798f, 1.0f, 1.0f,
			-0.20549594f, -0.12973769f, -0.05194321f, 0.015703587f, 0.09777545f, 0.18215998f, 0.25875565f, 0.33447963f, 0.42195317f, 0.50180626f, 0.57749295f, 0.6732955f, 0.7544204f, 0.8345335f, 0.93907034f, 1.0f,
			-0.27252433f, -0.19825025f, -0.11878087f, -0.054252144f, 0.030430404f, 0.11837929f, 0.18046571f, 0.26702133f, 0.3556527f, 0.42181906f, 0.5116626f, 0.60652936f, 0.67295575f, 0.76651496f, 0.86855584f, 0.9376883f,
			-0.33972633f, -0.26704144f, -0.19430517f, -0.11922642f, -0.035235282f, 0.044540312f, 0.113796435f, 0.19574164f, 0.28354076f, 0.34833696f, 0.44170392f, 0.5329772f, 0.59677434f, 0.69731396f, 0.79287136f, 0.8631765f,
			-0.41136292f, -0.3316812f, -0.26894954f, -0.18491371f, -0.102952115f, -0.032687183f, 0.04349723f, 0.1286961f, 0.20427181f, 0.27947795f, 0.37498918f, 0.45099926f, 0.529769f, 0.6311513f, 0.708919f, 0.7949911f,
			-0.47914532f, -0.40470284f, -0.33389997f, -0.25253442f, -0.17085297f, -0.10713934f, -0.025161354f, 0.061155852f, 0.124452256f, 0.20976661f, 0.30560043f, 0.37058663f, 0.46084446f, 0.5608588f, 0.6278716f, 0.7277667f,
			-0.5472712f, -0.4805125f, -0.40146482f, -0.32412034f, -0.24523436f, -0.1753956f, -0.09643509f, -0.01124367f, 0.05172267f, 0.14212205f, 0.23280455f, 0.2929635f, 0.38876233f, 0.48513037f, 0.55229884f, 0.6570123f,
			-0.61521155f, -0.5495049f, -0.4694931f, -0.3906205f, -0.3259219f, -0.24449004f, -0.16578509f, -0.09245425f, -0.01316294f, 0.070360295f, 0.15091299f, 0.2271816f, 0.3201112f, 0.40204373f, 0.4853807f, 0.5871072f,
			-0.69130135f, -0.61537844f, -0.5399882f, -0.46140474f, -0.3939904f, -0.31775084f, -0.23545174f, -0.17031057f, -0.085763626f, 0.00060200685f, 0.06825402f, 0.1561649f, 0.24968384f, 0.3188729f, 0.4129946f, 0.5150556f,
			-0.76188886f, -0.68412566f, -0.6098687f, -0.5394779f, -0.46542433f, -0.3886215f, -0.30816418f, -0.24103223f, -0.15787108f, -0.07129385f, -0.007195025f, 0.08385181f, 0.17855538f, 0.24048834f, 0.34071875f, 0.43531504f,
			-0.82933897f, -0.7570855f, -0.6821259f, -0.6134502f, -0.53375655f, -0.46051142f, -0.3891654f, -0.3123313f, -0.23019983f, -0.14981551f, -0.07721781f, 0.009345262f, 0.09509622f, 0.16990526f, 0.2689786f, 0.35209057f,
			-0.89840883f, -0.82836294f, -0.7534399f, -0.68324053f, -0.6099455f, -0.5312122f, -0.46723035f, -0.38338748f, -0.29916015f, -0.23177339f, -0.14903395f, -0.060713287f, 0.013361125f, 0.099740915f, 0.19600911f, 0.26567277f,
			-0.97112423f, -0.8994579f, -0.8309796f, -0.75469905f, -0.68325096f, -0.6048031f, -0.5356587f, -0.45639944f, -0.37169155f, -0.31004918f, -0.22217406f, -0.13091637f, -0.067415826f, 0.028172879f, 0.124010436f, 0.18436907f,
			-1.0f, -0.9690135f, -0.9011447f, -0.8277602f, -0.75371486f, -0.6856985f, -0.60859615f, -0.5280971f, -0.4523262f, -0.38129014f, -0.29598993f, -0.21053775f, -0.14167501f, -0.04709437f, 0.041146573f, 0.10676905f,
			-1.0f, -1.0f, -0.9719833f, -0.90253645f, -0.8250981f, -0.7620625f, -0.68250144f, -0.60331744f, -0.53554547f, -0.45532805f, -0.36834028f, -0.2917036f, -0.2118118f, -0.11907815f, -0.04318803f, 0.035587694f;
			return out_sdf_chunk;

		}();

static eig::Tensor<float, 3> TSDF_slice01 =
		[] {
			float data[] = {
					-0.20475684f, -0.13041718f, -0.05159303f, 0.01631528f, 0.09734481f, 0.18223374f, 0.25901046f, 0.3345288f, 0.4215054f, 0.5017995f, 0.57773066f, 0.67372465f, 0.7540442f, 0.8337743f, 0.9401298f, 1.0f,
					-0.2718329f, -0.19898264f, -0.11795311f, -0.05366355f, 0.03163889f, 0.11868029f, 0.18068849f, 0.2674229f, 0.35445613f, 0.42136607f, 0.51035875f, 0.6072543f, 0.6730221f, 0.76607686f, 0.8696056f, 0.9367048f,
					-0.33940297f, -0.26726198f, -0.19352509f, -0.11860802f, -0.0351347f, 0.04477277f, 0.11306702f, 0.19761248f, 0.28388125f, 0.34753007f, 0.44209582f, 0.5338877f, 0.59785986f, 0.69736683f, 0.7920019f, 0.86319065f,
					-0.41053814f, -0.33195686f, -0.26949567f, -0.18563195f, -0.10349526f, -0.03297701f, 0.04537329f, 0.12810154f, 0.20457803f, 0.28029305f, 0.3743246f, 0.4515655f, 0.529439f, 0.6300881f, 0.7094442f, 0.7964156f,
					-0.47973615f, -0.40533316f, -0.3329001f, -0.25372428f, -0.17178132f, -0.10733827f, -0.0254929f, 0.06117373f, 0.12537389f, 0.21193548f, 0.30558106f, 0.37155968f, 0.46049577f, 0.56024337f, 0.6278216f, 0.72762364f,
					-0.54754835f, -0.48085f, -0.40127334f, -0.32249835f, -0.24583338f, -0.17565785f, -0.09573101f, -0.01198053f, 0.05252584f, 0.14263166f, 0.23218466f, 0.29368994f, 0.3898285f, 0.484772f, 0.5527272f, 0.6566673f,
					-0.6144799f, -0.54812276f, -0.4697546f, -0.3909312f, -0.32430065f, -0.2446696f, -0.16611888f, -0.09179189f, -0.01491904f, 0.07018074f, 0.15120803f, 0.22628976f, 0.3201082f, 0.40280667f, 0.48454177f, 0.58679205f,
					-0.69139594f, -0.61513406f, -0.5401842f, -0.4608303f, -0.3944121f, -0.31756383f, -0.23505314f, -0.17100646f, -0.08574425f, 0.00108704f, 0.06889998f, 0.1552522f, 0.24996398f, 0.31985265f, 0.41284856f, 0.5157612f,
					-0.76291186f, -0.6857901f, -0.6103374f, -0.5386583f, -0.46513748f, -0.38871688f, -0.30887496f, -0.2414055f, -0.15761329f, -0.06929486f, -0.00930652f, 0.08366852f, 0.17710625f, 0.24074091f, 0.34142134f, 0.43653396f,
					-0.82926893f, -0.7568284f, -0.6821744f, -0.6148077f, -0.5358427f, -0.45973283f, -0.38855296f, -0.3113508f, -0.22937803f, -0.15131308f, -0.07774457f, 0.00998154f, 0.09599998f, 0.17126425f, 0.26991662f, 0.35248694f,
					-0.8988074f, -0.8265122f, -0.7519416f, -0.6832592f, -0.60939413f, -0.5306705f, -0.46718267f, -0.38460863f, -0.30043197f, -0.23241787f, -0.14970823f, -0.06056577f, 0.01350045f, 0.09886324f, 0.19591151f, 0.26573387f,
					-0.97098565f, -0.89905554f, -0.8297361f, -0.7546924f, -0.68361086f, -0.6053954f, -0.53770316f, -0.45659837f, -0.37168336f, -0.31055582f, -0.22116004f, -0.1305498f, -0.06811767f, 0.02796873f, 0.12376829f, 0.18438174f,
					-1.0f, -0.96868193f, -0.9008161f, -0.82786226f, -0.7554948f, -0.68548465f, -0.607904f, -0.5285464f, -0.45156848f, -0.38096604f, -0.2947412f, -0.20934565f, -0.14037861f, -0.04654675f, 0.04146173f, 0.10772794f,
					-1.0f, -1.0f, -0.97198033f, -0.9039304f, -0.8274629f, -0.7603347f, -0.6806403f, -0.60253733f, -0.5349218f, -0.45473945f, -0.37067902f, -0.29290763f, -0.21186395f, -0.12094676f, -0.04442408f, 0.03523007f,
					-1.0f, -1.0f, -1.0f, -0.9779266f, -0.9029112f, -0.832349f, -0.75637394f, -0.6757326f, -0.6126381f, -0.52764934f, -0.44117716f, -0.37548465f, -0.28660965f, -0.19398032f, -0.12813507f, -0.03899559f,
					-1.0f, -1.0f, -1.0f, -1.0f, -0.98226213f, -0.90591604f, -0.83188707f, -0.75543445f, -0.68388426f, -0.60274744f, -0.51609796f, -0.45198423f, -0.3620803f, -0.27120334f, -0.20937918f, -0.11253878f
			};

			auto mapped_t = Eigen::TensorMap<Eigen::Tensor<float, 3, Eigen::RowMajor>>(data, 16, 1, 16);
			Eigen::Tensor<float, 3> TSDF_slice01 = Eigen::TensorLayoutSwapOp<Eigen::Tensor<float, 3, Eigen::RowMajor>>(mapped_t);
			return TSDF_slice01;
		}();

static eig::Tensor<float, 3> TSDF_slice02 =
		[] {
			float data[] = {
					-0.15828474f, -0.11724376f, -0.10635137f, 0.0347948f, 0.13013457f, 0.13970095f, 0.2607098f, 0.37702486f, 0.38655585f, 0.4914882f, 0.6236901f, 0.63738084f, 0.7511944f, 0.8743434f, 0.89005965f, 1.0f,
					-0.2258139f, -0.2199551f, -0.15794252f, -0.00265987f, 0.03356163f, 0.06181556f, 0.22034676f, 0.28386486f, 0.29687205f, 0.4515414f, 0.533245f, 0.54962367f, 0.7096594f, 0.7776409f, 0.8098362f, 0.9918554f,
					-0.31726792f, -0.31567535f, -0.19318739f, -0.07427778f, -0.06489212f, 0.01729878f, 0.16743435f, 0.18366536f, 0.23672199f, 0.40936098f, 0.43442923f, 0.48488897f, 0.65332645f, 0.67893976f, 0.75372607f, 0.92021626f,
					-0.41854447f, -0.3823676f, -0.23076715f, -0.16607293f, -0.15566197f, -0.01788376f, 0.082306f, 0.08623048f, 0.19767742f, 0.32805297f, 0.33478448f, 0.44285813f, 0.5730631f, 0.5839522f, 0.7176227f, 0.830434f,
					-0.51883006f, -0.42022684f, -0.28082898f, -0.26674107f, -0.21839245f, -0.05833106f, -0.01604484f, 0.00105352f, 0.16258742f, 0.23426425f, 0.24564722f, 0.40389f, 0.47854367f, 0.49812043f, 0.6721107f, 0.73466957f,
					-0.6033763f, -0.4579158f, -0.3659076f, -0.36507088f, -0.2567028f, -0.12278296f, -0.11890986f, -0.05184137f, 0.11196412f, 0.13258268f, 0.17926085f, 0.3554237f, 0.37924826f, 0.4338902f, 0.6157429f, 0.63608646f,
					-0.645462f, -0.49682447f, -0.46659848f, -0.44712663f, -0.29905558f, -0.21631671f, -0.21445934f, -0.0931825f, 0.02886167f, 0.0339981f, 0.13835897f, 0.27350315f, 0.28232044f, 0.3953811f, 0.5281225f, 0.54116374f,
					-0.6869416f, -0.5688098f, -0.5686373f, -0.49458835f, -0.3414111f, -0.31428707f, -0.29624635f, -0.13149002f, -0.063048f, -0.05861344f, 0.09790006f, 0.17983738f, 0.18881369f, 0.35134864f, 0.43510342f, 0.45342973f,
					-0.72327024f, -0.66508883f, -0.661359f, -0.52946675f, -0.41768035f, -0.41714862f, -0.34437734f, -0.18377991f, -0.16648252f, -0.12047808f, 0.05118029f, 0.07906744f, 0.11693326f, 0.30343905f, 0.3356216f, 0.38579437f,
					-0.7763358f, -0.7674736f, -0.7283486f, -0.57718f, -0.51306856f, -0.514597f, -0.38262054f, -0.26476237f, -0.26809445f, -0.16572699f, -0.02654833f, -0.02175728f, 0.07301583f, 0.22863467f, 0.23383684f, 0.3388197f,
					-0.8644161f, -0.8672254f, -0.7708329f, -0.6276073f, -0.6174873f, -0.5900806f, -0.4260473f, -0.36359546f, -0.3600693f, -0.20942566f, -0.12119135f, -0.1146424f, 0.0320527f, 0.13398415f, 0.14114645f, 0.2985325f,
					-0.96556437f, -0.9566214f, -0.8115571f, -0.71163124f, -0.7212831f, -0.6320864f, -0.47746128f, -0.46737003f, -0.42747727f, -0.2539534f, -0.21941717f, -0.19709189f, -0.01466543f, 0.03628113f, 0.05920541f, 0.23854096f,
					-1.0f, -1.0f, -0.85546386f, -0.81321603f, -0.81221074f, -0.6737332f, -0.5629941f, -0.57266414f, -0.47133508f, -0.32217124f, -0.32374653f, -0.2472215f, -0.07647318f, -0.06676186f, 0.00130505f, 0.16856915f,
					-1.0f, -1.0f, -0.92087454f, -0.9182008f, -0.87941104f, -0.7177957f, -0.661619f, -0.6644632f, -0.51623654f, -0.41769585f, -0.42159325f, -0.28783286f, -0.16332234f, -0.167712f, -0.03794361f, 0.0806923f,
					-1.0f, -1.0f, -1.0f, -1.0f, -0.9208766f, -0.7725691f, -0.7679372f, -0.7299541f, -0.5587841f, -0.519628f, -0.5087995f, -0.3345998f, -0.2650984f, -0.25806686f, -0.09272385f, -0.019745f,
					-1.0f, -1.0f, -1.0f, -1.0f, -0.9690759f, -0.8593376f, -0.8730626f, -0.77391493f, -0.6238807f, -0.6228114f, -0.56681186f, -0.38827598f, -0.36741194f, -0.32917243f, -0.14534502f, -0.12005218f
			};

			auto mapped_t = Eigen::TensorMap<Eigen::Tensor<float, 3, Eigen::RowMajor>>(data, 16, 1, 16);
			Eigen::Tensor<float, 3> TSDF_slice01 = Eigen::TensorLayoutSwapOp<Eigen::Tensor<float, 3, Eigen::RowMajor>>(mapped_t);
			return TSDF_slice01;
		}();

} //namespace test_data
