#!/usr/bin/python3
#  ================================================================
#  Created by Gregory Kramida on 12/10/18.
#  Copyright (c) 2018 Gregory Kramida
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#  http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#  ================================================================

# a simple utility script that prints out basic paths for the python environment in which it is run
# stdlib
import sys, argparse, os, os.path
from sysconfig import get_paths

EXIT_CODE_SUCCESS = 0
EXIT_CODE_FAILURE = 1


def main():
    parser = argparse.ArgumentParser(
        "a simple utility script that prints out basic paths for the python environment in which it is run")
    parser.add_argument("--executable", action="store_true")
    parser.add_argument("--include_directory", action="store_true")
    parser.add_argument("--library_directory", action="store_true")
    args = parser.parse_args()
    info = get_paths()
    if args.executable:
        py_interpreter = os.path.join(os.__file__.split("lib/")[0], "bin", "python")
        print(py_interpreter)
    if args.include_directory:
        print(info['include'])
    if args.library_directory:
        print(os.path.dirname(info['stdlib']))
    return EXIT_CODE_SUCCESS


if __name__ == "__main__":
    sys.exit(main())
