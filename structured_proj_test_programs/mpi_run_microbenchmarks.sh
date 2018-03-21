#!/bin/bash

TRIALS=1
ITERATIONS=10

for ((trial=1; trial<=$TRIALS; trial++ ))
do
  for nodes in 1 2 4
  do
    hostfile="hostfile$nodes"
    dim=`expr 4 \* $nodes`
    for sleep_us in 0 1000 10000
    do
      affine_command="mpirun -hostfile $hostfile -n $nodes -npernode 1 --bind-to none ./x_parallel_only/runtime_stress_test -i $ITERATIONS -nx $dim -bx $dim -ny $dim -by $dim -sms $sleep_us -sm -ssl 4 -a 225 -ll:cpu 4 -ll:util 1"
      index_command="mpirun -hostfile $hostfile -n $nodes -npernode 1 --bind-to none ./current_optimum_angle/current_optimum_angle -i $ITERATIONS -nx $dim -bx $dim -ny $dim -by $dim -sms $sleep_us -ll:cpu 4 -ll:util 1"
      no_proj_command="mpirun -hostfile $hostfile -n $nodes -npernode 1 --bind-to none ./zero_projections/zero_projections -i $ITERATIONS -nx $dim -bx $dim -ny $dim -by $dim -sms $sleep_us -a 225 -ll:cpu 4 -ll:util 1"

      (>&2 echo $affine_command)
      echo $affine_command
      $affine_command

      (>&2 echo $index_command)
      echo $index_command
      $index_command

      (>&2 echo $no_proj_command)
      echo $no_proj_command
      $no_proj_command
    done

    for sleep_us in 0 1000 10000 100000
    do
      affine_command="mpirun -hostfile $hostfile -n $nodes -npernode 1 --bind-to none ./x_parallel_only/runtime_stress_test -i $ITERATIONS -nx 1 -bx 1 -ny $dim -by $dim -sms $sleep_us -a 180 -ll:cpu 4 -ll:util 1"
      index_command="mpirun -hostfile $hostfile -n $nodes -npernode 1 --bind-to none ./current_optimum_axis_aligned/current_optimum_axis_aligned -i $ITERATIONS -nx 1 -bx 1 -ny $dim -by $dim -sms $sleep_us -ll:cpu 4 -ll:util 1"
      no_proj_command="mpirun -hostfile $hostfile -n $nodes -npernode 1 --bind-to none ./zero_projections/zero_projections -i $ITERATIONS -nx 1 -bx 1 -ny $dim -by $dim -sms $sleep_us -a 180 -ll:cpu 4 -ll:util 1"

      (>&2 echo $affine_command)
      echo $affine_command
      $affine_command

      (>&2 echo $index_command)
      echo $index_command
      $index_command

      (>&2 echo $no_proj_command)
      echo $no_proj_command
      $no_proj_command
    done
  done
done
