#!/bin/bash
echo ===== RUN_ONE with these parameters:
cat $0
echo =====
touch $LG_RT_DIR/../examples/visualization/prototype1.cc
touch $LG_RT_DIR/legion/RenderSpace.cc
(cd $LG_RT_DIR/../examples/visualization; echo "$1 TREE_REDUCTION=1 NULL_COMPOSITE_TASKS=0 TIME_PER_FRAME=1 TIME_OVERALL=1 DEBUG=0 make" | bash)
echo === RUNNING WITH "$1" ===
rm -rf profileData
mkdir profileData
mpirun -n 4 -npernode 1 -H n0000,n0001,n0002,n0003 --bind-to none $LG_RT_DIR/../examples/visualization/prototype1 -hl:prof 4 -logfile profileData/node_%.log
rm display.*
