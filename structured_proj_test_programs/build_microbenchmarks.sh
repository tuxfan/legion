#!/bin/bash

set -e

cd x_parallel_only
make -j 2
cd ..

cd zero_projections
make -j 2
cd ..

cd current_optimum_axis_aligned
make -j 2
cd ..

cd current_optimum_angle     
make -j 2
cd ..
