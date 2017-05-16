#!/bin/bash
echo ===== RUN_ONE with these parameters:
cat $0
echo =====
touch ../legion/examples/visualization/prototype1.cc
touch ../legion/runtime/legion/RenderSpace.cc
(cd ../legion/examples/visualization; echo "$1 TREE_REDUCTION=1 NULL_COMPOSITE_TASKS=0 TIME_PER_FRAME=1 TIME_OVERALL=1 DEBUG=0 make" | bash)
echo === RUNNING WITH "$1" ===
mpirun -n 4 -npernode 1 -H n0000,n0001,n0002,n0003 --bind-to none ../legion/examples/visualization/prototype1
rm display.*
