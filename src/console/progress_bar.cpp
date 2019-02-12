/*
 * progress_bar.cpp
 *
 *  Created on: Feb 12, 2019
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

#include <iostream>

#include "progress_bar.hpp"

namespace console {

ProgressBar::ProgressBar() {
}

//TODO: change to "maximum character width", possibly, that calculates filler_count_cap based on maximum possible width
//of characters around it.
ProgressBar::ProgressBar(int filler_count_cap) : filler_count_cap(filler_count_cap) {
}

void ProgressBar::update(double progress_increment) {
	current_progress += progress_increment;
	filler_count = (int) ((current_progress / needed_progress) * (double) filler_count_cap);
}
void ProgressBar::print() {
	using namespace std;
	bar_end_character_index %= bar_end_characters.length();
	cout << "\r" //Bring cursor to start of line
			<< opener; //Print out first part of pBar
	for (int a = 0; a < filler_count; a++) { //Print out current progress
		cout << filler;
	}
	cout << bar_end_characters[bar_end_character_index];
	for (int b = 0; b < filler_count_cap - filler_count; b++) { //Print out spaces
		cout << " ";
	}
	cout << closer //Print out last part of progress bar
			<< " (" << (int) (100 * (current_progress / needed_progress)) << "%)" //This just prints out the percent
			<< flush;
	bar_end_character_index += 1;
}

} //namespace console

