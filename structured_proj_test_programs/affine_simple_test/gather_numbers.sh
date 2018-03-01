#!/bin/bash

ITERATIONS=1
TRIALS=2

for ((x=1; x<=$TRIALS; x++ ))
do
  for sleep_ms in 0
  do
    for ssl in 4 8 2
    do
      (>&2 echo "mpirun -n 4 ./runtime_stress_test -i $ITERATIONS -n 4 -b 4 -sm -ssl $ssl -sms $sleep_ms -ll:cpu 4")
      echo "mpirun -n 4 ./runtime_stress_test -i $ITERATIONS -n 4 -b 4 -sm -ssl $ssl -sms $sleep_ms -ll:cpu 4"
      mpirun -n 4 ./runtime_stress_test -i $ITERATIONS -n 4 -b 4 -sm -ssl $ssl -sms $sleep_ms -ll:cpu 4
      echo ""
    done
  done
done

exit 1

# EVERYTHING BELOW HERE JUST KEPT FOR REFERENCE

for xdim in 0 1 2 3 4 5 6 7 8 9
#for xdim in 10
do
  ydim=`expr 10 - $xdim`
  echo "I am at xdim $xdim and ydim $ydim"
  for ((x=1; x<=$TRIALS; x++ ))
  do
    ./simple_test/runtime_stress_test -nx $xdim -bx $xdim -ny $ydim -by $ydim -a 180 -i $ITERATIONS -hl:prof 1 -logfile simple_test/"prof_$xdim""_$ydim""_$x.log" > simple_test/"out_$xdim""_$ydim""_$x"  2>&1
    ../tools/legion_prof.py simple_test/"prof_$xdim""_$ydim""_$x.log"
    mv legion_prof simple_test/"prof_$xdim""_$ydim""_$x"
    ./current_optimum_axis_aligned/current_optimum_axis_aligned -nx $xdim -bx $xdim -ny $ydim -by $ydim -a 180 -i $ITERATIONS -hl:prof 1 -logfile current_optimum_axis_aligned/"prof_$xdim""_$ydim""_$x.log" > current_optimum_axis_aligned/"out_$xdim""_$ydim""_$x" 2>&1
    ../tools/legion_prof.py current_optimum_axis_aligned/"prof_$xdim""_$ydim""_$x.log"
    mv legion_prof current_optimum_axis_aligned/"prof_$xdim""_$ydim""_$x"
    ./no_projections/no_projections -nx $xdim -bx $xdim -ny $ydim -by $ydim -a 180 -i $ITERATIONS -hl:prof 1 -logfile no_projections/"prof_$xdim""_$ydim""_$x.log" > no_projections/"out_$xdim""_$ydim""_$x" 2>&1
    ../tools/legion_prof.py no_projections/"prof_$xdim""_$ydim""_$x.log"
    mv legion_prof no_projections/"prof_$xdim""_$ydim""_$x"
  done
done
