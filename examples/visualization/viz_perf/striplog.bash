#!/bin/bash
grep ":" $1 | sed -e "s/^.*: //" | sed -e "s/ /,/g" > $1.csv
