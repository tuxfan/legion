#!/bin/bash
grep ":" | sed -e "s/^.*: //" | sed -e "s/ /,/g" 
