#!/bin/bash
path=$1
exper=$2
exec_python="/home/gaurava/miniconda3/envs/catastrophic/bin/python"
find $path -name '*_exper__*' | while read p; do $exec_python aggregator.py --path $p --operations mean  --store_path $path/evaluate_csv_agg --output csv --store_df --allowed_keys standard_evaluate --allowed_keys evaluate --allowed_keys restore_checkpoint --allowed_keys forgetting_metric; done > ${exper}_agg.log
#find $path -name '*_exper__*' | while read p; do python aggregator.py --path $p --operations mean  --store_path evaluate_csv_agg --output csv --store_df --allowed_keys standard_evaluate --allowed_keys evaluate --allowed_keys restore_checkpoint; done > ${exper}_agg.log
#python process_results.py --path $path/evaluate_csv_agg/aggregates/  --exper $exper  --get_df > ${exper}_df.log
#python process_results.py --path $path/evaluate_csv_agg/aggregates/  --exper $exper > ${exper}_new.log
#python process_results.py --path $path/evaluate_csv_agg/aggregates/  --exper $exper --two_task --restore_checkpoint > ${exper}_two_task_new_restor.log

#while read p; do python ~/catastrophic_forgetting/tensorboard-aggregator/aggregator.py --path $p --operations mean  --store_path ./agg; done < all_runs
