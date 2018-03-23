#!/bin/bash

set -e

cd 2p5d_matrix_multiply
make -j 2 $1
cd ..

cd affine_2p5d_matrix_multiply
make -j 2 $1
cd ..

cd index_2p5d_matrix_multiply
make -j 2 $1
cd ..
