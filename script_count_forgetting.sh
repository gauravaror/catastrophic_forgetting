#!/bin/bash

grep Forgetting $1  | awk '{print $4}' | awk -F, ' { sum+=$1; count+=1} END {print "Sum "sum" Count  " count"  Forgetting "sum/count}'
