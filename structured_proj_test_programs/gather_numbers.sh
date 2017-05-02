#!/bin/bash

ITERATIONS=11
TRIALS=2

for xdim in 2 3 4 5
do
  for ydim in 2 3 4 5
  do
  echo "I am at xdim $xdim and ydim $ydim"
    for ((x=1; x<=$TRIALS; x++ ))
      do
        ./simple_test/runtime_stress_test -nx $xdim -bx $xdim -ny $ydim -by $ydim -a 180 -i $ITERATIONS -hl:prof 1 -logfile simple_test/prof_%.log > simple_test/out 2>&1
        ../tools/legion_prof.py simple_test/prof_0.log
        mv legion_prof simple_test/"prof_$xdim""_$ydim""_$x"
        ./current_optimum_axis_aligned/current_optimum_axis_aligned -nx $xdim -bx $xdim -ny $ydim -by $ydim -a 180 -i $ITERATIONS -hl:prof 1 -logfile current_optimum_axis_aligned/prof_%.log > current_optimum_axis_aligned/out 2>&1
        ../tools/legion_prof.py current_optimum_axis_aligned/prof_0.log
        mv legion_prof current_optimum_axis_aligned/"prof_$xdim""_$ydim""_$x"
      done
  done
done
