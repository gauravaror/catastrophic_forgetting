#!/bin/bash

logfile=$1
grep Forgetting $1  | awk '{print $4}' | awk -F, -v logfile=$logfile ' { sum+=$1; count+=1} END {print "Forgetting "logfile" Sum "sum" Count  " count"  Forgetting "sum/count}'
