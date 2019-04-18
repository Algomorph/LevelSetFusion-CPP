//  ================================================================
//  Created by Gregory Kramida on 10/23/18.
//  Copyright (c) 2018 Gregory Kramida
//  Licensed under the Apache License, Version 2.0 (the "License");
//  you may not use this file except in compliance with the License.
//  You may obtain a copy of the License at

//  http://www.apache.org/licenses/LICENSE-2.0

//  Unless required by applicable law or agreed to in writing, software
//  distributed under the License is distributed on an "AS IS" BASIS,
//  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
//  See the License for the specific language governing permissions and
//  limitations under the License.
//  ================================================================
#pragma once

#include <Eigen/Eigen>

#include "../../src/math/stacking.hpp"
#include "../../src/math/typedefs.hpp"

namespace eig = Eigen;
namespace test_data {



static math::MatrixXv2f vector_field = [] {
	math::MatrixXv2f field(4, 4);
	field << math::Vector2f(0.66137378f, 0.22941163f), math::Vector2f(-0.79364663f, -0.51078996f),
			math::Vector2f(0.31330802f, -0.62231087f), math::Vector2f(0.38155258f, 0.25911068f),

			math::Vector2f(-0.93761754f, 0.22711085f), math::Vector2f(-0.84484027f, 0.74134703f),
			math::Vector2f(-0.77734907f, 0.31051154f), math::Vector2f(0.05594392f, 0.62550403f),

			math::Vector2f(0.24144975f, -0.03810476f), math::Vector2f(-0.83927967f, 0.2171229f),
			math::Vector2f(0.3517115f, -0.34761186f), math::Vector2f(-0.3781738f, 0.4708583f),

			math::Vector2f(-0.60896495f, 0.32025099f), math::Vector2f(0.11699246f, -0.98680021f),
			math::Vector2f(-0.96371592f, -0.93434108f), math::Vector2f(0.42603218f, 0.76691092f);
	return field;
}();

static math::MatrixXv2f vector_field2 = []{
		math::MatrixXv2f vector_field(4,4);
		vector_field <<
		math::Vector2f(0.8562016f,0.876527f),
		math::Vector2f(0.8056713f,0.31369442f),
		math::Vector2f(0.28571403f,0.38419583f),
		math::Vector2f(0.86377007f,0.9078812f),

		math::Vector2f(0.12255816f,0.22223428f),
		math::Vector2f(0.4487159f,0.7280231f),
		math::Vector2f(0.61369246f,0.43351218f),
		math::Vector2f(0.3545089f,0.33867624f),

		math::Vector2f(0.5658683f,0.53506494f),
		math::Vector2f(0.69546276f,0.9331944f),
		math::Vector2f(0.05706289f,0.06915309f),
		math::Vector2f(0.5286004f,0.9154799f),

		math::Vector2f(0.98797816f,0.60008055f),
		math::Vector2f(0.07343615f,0.10326899f),
		math::Vector2f(0.28764063f,0.05625961f),
		math::Vector2f(0.32258928f,0.84611595f);
		return vector_field;
}();

static math::MatrixXv2f min_max_vector_field_2d = []{
		math::MatrixXv2f a(4,4);
		a <<
		math::Vector2f(0.78405046f,0.9798877f),
		math::Vector2f(0.28015256f,0.40328237f),
		math::Vector2f(0.5662102f,0.28645408f),
		math::Vector2f(0.6614452f,0.92681277f),

		math::Vector2f(0.3718108f,0.42288345f),
		math::Vector2f(0.67824876f,0.27738795f),
		math::Vector2f(0.71616673f,0.9417121f),
		math::Vector2f(0.96488345f,0.2267198f),

		math::Vector2f(0.16374928f,0.25913543f),
		math::Vector2f(0.5285367f,0.16944581f),
		math::Vector2f(0.23083617f,0.97394335f),
		math::Vector2f(0.3437231f,0.30852303f),

		math::Vector2f(0.5015861f,0.0235084f),
		math::Vector2f(0.5898168f,0.10401133f),
		math::Vector2f(0.94596475f,0.40647414f),
		math::Vector2f(0.790539f,0.80683255f);
		return a;
}();

static math::Tensor3v3f min_max_vector_field_3d = []{
		math::Tensor3v3f a(2,10,10);
		a.setValues(  // @formatter:off
		{{{math::Vector3f(0.7165495f,0.48943874f,0.9207701f), math::Vector3f(0.80157256f,0.72613907f,0.88975f), math::Vector3f(0.31079254f,0.40324473f,0.9938947f), math::Vector3f(0.80578816f,0.48779085f,0.25972238f), math::Vector3f(0.24042216f,0.16470388f,0.28835446f), math::Vector3f(0.41443664f,0.46210143f,0.03430727f), math::Vector3f(0.04968845f,0.03093819f,0.85559446f), math::Vector3f(0.4722724f,0.98116887f,0.5199828f), math::Vector3f(0.03048815f,0.56451744f,0.7113241f), math::Vector3f(0.06167091f,0.9954184f,0.73246366f)},
		  {math::Vector3f(0.80596656f,0.7375079f,0.49496296f), math::Vector3f(0.2230136f,0.636113f,0.02736912f), math::Vector3f(0.36140412f,0.12373508f,0.72105324f), math::Vector3f(0.36695564f,0.7056854f,0.26764566f), math::Vector3f(0.4517172f,0.6977666f,0.3181249f), math::Vector3f(0.4317303f,0.10946632f,0.38373315f), math::Vector3f(0.10085374f,0.83288336f,0.09121688f), math::Vector3f(0.7824683f,0.39280668f,0.41132256f), math::Vector3f(0.95743316f,0.22547875f,0.5265264f), math::Vector3f(0.71360743f,0.721003f,0.27660295f)},
		  {math::Vector3f(0.56913656f,0.1384271f,0.16458325f), math::Vector3f(0.42977756f,0.0441877f,0.42075092f), math::Vector3f(0.86102897f,0.6235421f,0.10170314f), math::Vector3f(0.2550346f,0.21024588f,0.22188064f), math::Vector3f(0.68453175f,0.71459335f,0.71503234f), math::Vector3f(0.25356814f,0.5912812f,0.3343177f), math::Vector3f(0.07231545f,0.9508454f,0.29499188f), math::Vector3f(0.27075624f,0.17369547f,0.8506954f), math::Vector3f(0.60843563f,0.47400165f,0.41718873f), math::Vector3f(0.88524026f,0.2190986f,0.6739966f)},
		  {math::Vector3f(0.21833654f,0.5049851f,0.3328534f), math::Vector3f(0.28239897f,0.43071187f,0.60568845f), math::Vector3f(0.01415459f,0.595458f,0.65082407f), math::Vector3f(0.4108039f,0.8241738f,0.6467382f), math::Vector3f(0.11497631f,0.7785938f,0.28198898f), math::Vector3f(0.13655585f,0.4830826f,0.965976f), math::Vector3f(0.3625159f,0.28079143f,0.7789122f), math::Vector3f(0.37134537f,0.6300833f,0.68851125f), math::Vector3f(0.08108666f,0.61167115f,0.3754735f), math::Vector3f(0.89472353f,0.93174136f,0.45355394f)},
		  {math::Vector3f(0.601546f,0.18818672f,0.5109048f), math::Vector3f(0.5718542f,0.16037366f,0.46362168f), math::Vector3f(0.40741074f,0.46748132f,0.8081004f), math::Vector3f(0.97745675f,0.16132233f,0.49627987f), math::Vector3f(0.961376f,0.12959576f,0.9439975f), math::Vector3f(0.9091847f,0.22394343f,0.11843015f), math::Vector3f(0.35505652f,0.2348139f,0.04319603f), math::Vector3f(0.7086228f,0.3792046f,0.18445279f), math::Vector3f(0.55224645f,0.39422f,0.91316056f), math::Vector3f(0.5320043f,0.6763147f,0.25592574f)},
		  {math::Vector3f(0.32010037f,0.8897935f,0.9785528f), math::Vector3f(0.88225335f,0.27801135f,0.1445218f), math::Vector3f(0.55393004f,0.5554608f,0.04068211f), math::Vector3f(0.8830587f,0.36007914f,0.2747109f), math::Vector3f(0.50977707f,0.5492996f,0.5972368f), math::Vector3f(0.46802458f,0.42881694f,0.39760315f), math::Vector3f(0.68647516f,0.07118869f,0.78249675f), math::Vector3f(0.7308035f,0.07806385f,0.953805f), math::Vector3f(0.39254466f,0.2959708f,0.00763412f), math::Vector3f(0.07032321f,0.05416281f,0.19167818f)},
		  {math::Vector3f(0.04738608f,0.32286173f,0.13959946f), math::Vector3f(0.6835638f,0.48305517f,0.82623714f), math::Vector3f(0.47755605f,0.40649208f,0.70649034f), math::Vector3f(0.9790725f,0.9251069f,0.18646184f), math::Vector3f(0.1487571f,0.99989426f,0.90959316f), math::Vector3f(0.39396682f,0.29178575f,0.47584656f), math::Vector3f(0.04802097f,0.8639744f,0.08738898f), math::Vector3f(0.310701f,0.8144276f,0.23702073f), math::Vector3f(0.9186077f,0.98608834f,0.8588128f), math::Vector3f(0.1861132f,0.49193507f,0.5612229f)},
		  {math::Vector3f(0.48620385f,0.35160193f,0.37852994f), math::Vector3f(0.2970069f,0.36889243f,0.16444731f), math::Vector3f(0.5422468f,0.93241346f,0.99423134f), math::Vector3f(0.8326817f,0.24499069f,0.914082f), math::Vector3f(0.5642468f,0.74430054f,0.22376251f), math::Vector3f(0.91525364f,0.17002325f,0.23794839f), math::Vector3f(0.22115082f,0.6743431f,0.83369064f), math::Vector3f(0.50809103f,0.30003193f,0.11224443f), math::Vector3f(0.13560216f,0.81885517f,0.11446422f), math::Vector3f(0.35240725f,0.88065434f,0.71952367f)},
		  {math::Vector3f(0.7715522f,0.6998401f,0.84034866f), math::Vector3f(0.9498834f,0.980101f,0.2851007f), math::Vector3f(0.00702147f,0.666687f,0.9340406f), math::Vector3f(0.1880537f,0.53499866f,0.50477123f), math::Vector3f(0.8581709f,0.7490976f,0.15079397f), math::Vector3f(0.9140811f,0.33590075f,0.28340685f), math::Vector3f(0.4174464f,0.48869196f,0.71161383f), math::Vector3f(0.2795076f,0.11121028f,0.7659716f), math::Vector3f(0.28354314f,0.38321045f,0.16627522f), math::Vector3f(0.296919f,0.37539157f,0.12180661f)},
		  {math::Vector3f(0.8306059f,0.10048822f,0.80276775f), math::Vector3f(0.60362726f,0.7738774f,0.56822795f), math::Vector3f(0.6450635f,0.77167594f,0.04447372f), math::Vector3f(0.0726187f,0.10139822f,0.03579007f), math::Vector3f(0.71114147f,0.21784894f,0.44995192f), math::Vector3f(0.7290512f,0.35070765f,0.35842982f), math::Vector3f(0.26505494f,0.04682811f,0.35498154f), math::Vector3f(0.5031307f,0.98547107f,0.94668645f), math::Vector3f(0.5212484f,0.7094825f,0.2226205f), math::Vector3f(0.79231954f,0.4001398f,0.776595f)}},
		 {{math::Vector3f(0.73665255f,0.17917411f,0.04248227f), math::Vector3f(0.10607181f,0.03741137f,0.52373147f), math::Vector3f(0.38223308f,0.3212718f,0.13221565f), math::Vector3f(0.42975214f,0.14827327f,0.21753071f), math::Vector3f(0.6708201f,0.97724175f,0.37819362f), math::Vector3f(0.9477836f,0.33380938f,0.76464134f), math::Vector3f(0.89301956f,0.76701325f,0.70243317f), math::Vector3f(0.43827674f,0.6351484f,0.8301497f), math::Vector3f(0.82056403f,0.63304996f,0.746216f), math::Vector3f(0.70050454f,0.76326203f,0.4471647f)},
		  {math::Vector3f(0.9912523f,0.545864f,0.40106866f), math::Vector3f(0.2023253f,0.8836931f,0.10978477f), math::Vector3f(0.04983322f,0.35848424f,0.12689388f), math::Vector3f(0.1471453f,0.37336504f,0.1737832f), math::Vector3f(0.31306207f,0.86201423f,0.6077333f), math::Vector3f(0.37406996f,0.75725645f,0.45936355f), math::Vector3f(0.3722261f,0.5865631f,0.63636434f), math::Vector3f(0.50199974f,0.2376044f,0.14877377f), math::Vector3f(0.15317386f,0.20477113f,0.8509241f), math::Vector3f(0.55403054f,0.5237999f,0.06301054f)},
		  {math::Vector3f(0.21110488f,0.82454896f,0.4165882f), math::Vector3f(0.74689585f,0.31065026f,0.52945894f), math::Vector3f(0.03741373f,0.33460376f,0.33037263f), math::Vector3f(0.7501f,0.96481854f,0.00449334f), math::Vector3f(0.7998611f,0.80457467f,0.58868647f), math::Vector3f(0.60129106f,0.5186596f,0.70704246f), math::Vector3f(0.93854433f,0.8133978f,0.34690863f), math::Vector3f(0.02506773f,0.56068724f,0.13221107f), math::Vector3f(0.2817118f,0.70835143f,0.7808086f), math::Vector3f(0.814768f,0.04139499f,0.3227954f)},
		  {math::Vector3f(0.674373f,0.31737888f,0.05131278f), math::Vector3f(0.8397229f,0.23341344f,0.4350822f), math::Vector3f(0.8660819f,0.7426962f,0.92227125f), math::Vector3f(0.77737516f,0.20000704f,0.62955123f), math::Vector3f(0.9563891f,0.4249911f,0.09006146f), math::Vector3f(0.23654528f,0.18202102f,0.00685253f), math::Vector3f(0.592255f,0.0273354f,0.5307227f), math::Vector3f(0.40426803f,0.43413797f,0.26554418f), math::Vector3f(0.3806874f,0.46337134f,0.19087f), math::Vector3f(0.613566f,0.9860318f,0.18839535f)},
		  {math::Vector3f(0.37258324f,0.9165035f,0.36954308f), math::Vector3f(0.43727762f,0.13340063f,0.15748289f), math::Vector3f(0.2923092f,0.04496201f,0.28134656f), math::Vector3f(0.34786385f,0.6091274f,0.13185965f), math::Vector3f(0.9867467f,0.21717338f,0.5778075f), math::Vector3f(0.01098542f,0.3537981f,0.118691f), math::Vector3f(0.23814857f,0.913951f,0.39035133f), math::Vector3f(0.58184063f,0.4321736f,0.25048533f), math::Vector3f(0.43943056f,0.0634557f,0.80364794f), math::Vector3f(0.79180354f,0.6316572f,0.13894425f)},
		  {math::Vector3f(0.71214926f,0.7002201f,0.03187998f), math::Vector3f(0.76128596f,0.6138041f,0.16124888f), math::Vector3f(0.37784988f,0.5094309f,0.01041031f), math::Vector3f(0.16157876f,0.9885043f,0.3521205f), math::Vector3f(0.04165163f,0.8176184f,0.86007285f), math::Vector3f(0.9961686f,0.5624157f,0.93111086f), math::Vector3f(0.11385946f,0.531999f,0.4401212f), math::Vector3f(0.10407481f,0.08359636f,0.7222147f), math::Vector3f(0.18707024f,0.34987763f,0.88587284f), math::Vector3f(0.92889655f,0.4063178f,0.32538363f)},
		  {math::Vector3f(0.9355647f,0.09742573f,0.75055224f), math::Vector3f(0.5915744f,0.79331684f,0.30303752f), math::Vector3f(0.8459528f,0.8472265f,0.19697371f), math::Vector3f(0.16640303f,0.55578905f,0.21867305f), math::Vector3f(0.41168267f,0.81031615f,0.06525451f), math::Vector3f(0.5057709f,0.47736827f,0.85477006f), math::Vector3f(0.4579038f,0.65513587f,0.25861123f), math::Vector3f(0.99175847f,0.31484577f,0.97216094f), math::Vector3f(0.3468683f,0.7306849f,0.5959604f), math::Vector3f(0.1419366f,0.86559856f,0.6955921f)},
		  {math::Vector3f(0.50752914f,0.5838128f,0.72045344f), math::Vector3f(0.27202478f,0.85838205f,0.03600685f), math::Vector3f(0.70728713f,0.24630766f,0.8250742f), math::Vector3f(0.91394806f,0.39316395f,0.89746f), math::Vector3f(0.3685869f,0.37157622f,0.7243056f), math::Vector3f(0.9744373f,0.15927649f,0.55028206f), math::Vector3f(0.8682159f,0.5505595f,0.32264516f), math::Vector3f(0.6075008f,0.17175043f,0.26533332f), math::Vector3f(0.88359654f,0.76342744f,0.26430908f), math::Vector3f(0.896029f,0.6412402f,0.01338448f)},
		  {math::Vector3f(0.05488047f,0.5275411f,0.85693145f), math::Vector3f(0.62269026f,0.62321186f,0.4341763f), math::Vector3f(0.991264f,0.5485282f,0.5561482f), math::Vector3f(0.48634046f,0.8303475f,0.42862904f), math::Vector3f(0.4278228f,0.71441853f,0.79258543f), math::Vector3f(0.90307087f,0.9975391f,0.36571348f), math::Vector3f(0.760766f,0.08254142f,0.9419347f), math::Vector3f(0.06465627f,0.87850565f,0.43280444f), math::Vector3f(0.80075425f,0.99291587f,0.88699013f), math::Vector3f(0.05121073f,0.46574306f,0.8170013f)},
		  {math::Vector3f(0.41661528f,0.43684065f,0.3336525f), math::Vector3f(0.03390671f,0.43147627f,0.03734098f), math::Vector3f(0.7021477f,0.6097165f,0.63963664f), math::Vector3f(0.8664309f,0.23374116f,0.9752651f), math::Vector3f(0.5264927f,0.5311655f,0.11091515f), math::Vector3f(0.801757f,0.05203671f,0.5540435f), math::Vector3f(0.6294477f,0.35832343f,0.10859773f), math::Vector3f(0.4732011f,0.4532958f,0.18179259f), math::Vector3f(0.8695046f,0.42692405f,0.41529346f), math::Vector3f(0.2236462f,0.23026372f,0.9685977f)}}}); // @formatter:on
		return a;
}();

static math::Tensor3f min_max_scalar_field_3d = []{
		math::Tensor3f a(4,2,2);
		a.setValues(  // @formatter:off
		{{{0.27834353f, 0.39645594f},
		  {0.2151391f, 0.7547714f}},
		 {{0.20948082f, 0.9190089f},
		  {0.2720658f, 0.90341747f}},
		 {{0.524159f, 0.30052534f},
		  {0.616307f, 0.11654199f}},
		 {{0.76239586f, 0.81117475f},
		  {0.48138294f, 0.60255134f}}}); // @formatter:on
		return a;
}();


}//namespace test_data
