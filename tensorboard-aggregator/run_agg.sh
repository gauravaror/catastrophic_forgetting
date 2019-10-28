
#while read p; do python ~/catastrophic_forgetting/tensorboard-aggregator/aggregator.py --path $p --operations mean  --store_path ./agg; done < all_runs
while read p; do python ~/catastrophic_forgetting/tensorboard-aggregator/aggregator.py --path $p --operations mean  --store_path ./evaluate_csv_agg --output csv --store_df --allowed_keys standard_evaluate --allowed_keys evaluate --allowed_keys restore_checkpoint; done < all_runs
