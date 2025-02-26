#!/usr/bin/env python

# Copyright 2019 Stanford University
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

from __future__ import print_function

import legion
from legion import task, RW
import numpy

@task(privileges=[RW])
def hello(R):
    print(R.x)

# Define the main task. This task is called first.
@task
def main():
    R = legion.Region.create([0, 0], {'x': legion.float64})
    legion.fill(R, 'x', 3.14)

    hello(R)

if __name__ == '__legion_main__':
    main()
