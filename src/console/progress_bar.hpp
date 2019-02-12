/*
 * progress_bar.hpp
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

#pragma once
#include <string>

namespace console{

class ProgressBar {

public:
	ProgressBar();
	ProgressBar(int filler_count_cap);

    void update(double progress_increment);
    void print();
    std::string opener = "[", //beginning and end 'bracket' characters of the progress bar
        closer = "]",
        filler = "=",
        bar_end_characters = "/-\\|"; //will be cycled through
private:
    int filler_count = 0,
        filler_count_cap = 50,
        bar_end_character_index = 0;
    double current_progress = 0,
        needed_progress = 1.0;
};

}//namespace console
