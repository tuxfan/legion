#!/bin/bash

TRIALS=1
ITERATIONS=10
NODES=4

for ((trial=1; trial<=$TRIALS; trial++ ))
do
  hostfile="hostfile$NODES"
  base_dim=`expr 4 \* $NODES`
  base_sleep=32000
  base_ssl=4
  for scale in 1 2 4 8
  do
    dim=$(($scale * $base_dim))
    sleep_us=$(($base_sleep / $scale / $scale))
    ssl=$(($scale * $base_ssl))

    affine_command="mpirun -hostfile $hostfile -n $NODES -npernode 1 --bind-to none ./x_parallel_only/runtime_stress_test -i $ITERATIONS -nx $dim -bx $dim -ny $dim -by $dim -sms $sleep_us -sm -ssl $ssl -a 225 -ll:cpu 4 -ll:util 1"
    affine_command_base_ssl="mpirun -hostfile $hostfile -n $NODES -npernode 1 --bind-to none ./x_parallel_only/runtime_stress_test -i $ITERATIONS -nx $dim -bx $dim -ny $dim -by $dim -sms $sleep_us -sm -ssl $base_ssl -a 225 -ll:cpu 4 -ll:util 1"
    index_command="mpirun -hostfile $hostfile -n $NODES -npernode 1 --bind-to none ./current_optimum_angle/current_optimum_angle -i $ITERATIONS -nx $dim -bx $dim -ny $dim -by $dim -sms $sleep_us -ll:cpu 4 -ll:util 1"
    no_proj_command="mpirun -hostfile $hostfile -n $NODES -npernode 1 --bind-to none ./zero_projections/zero_projections -i $ITERATIONS -nx $dim -bx $dim -ny $dim -by $dim -sms $sleep_us -a 225 -ll:cpu 4 -ll:util 1"

    (>&2 echo $affine_command)
    echo $affine_command
    $affine_command

    (>&2 echo $affine_command_base_ssl)
    echo $affine_command_base_ssl
    $affine_command_base_ssl

    (>&2 echo $index_command)
    echo $index_command
    $index_command

    (>&2 echo $no_proj_command)
    echo $no_proj_command
    $no_proj_command
  done
done
