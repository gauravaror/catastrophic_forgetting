# MIT License
# Copyright (c) 2019 Sebastian Penhouet
# GitHub project: https://github.com/Spenhouet/tensorboard-aggregator
# ==============================================================================
"""Aggregates multiple tensorbaord runs"""

import ast
import argparse
import os
import re
from pathlib import Path

import pandas as pd
import numpy as np
import tensorflow as tf
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from tensorflow.core.util.event_pb2 import Event

FOLDER_NAME = 'aggregates'


def extract(dpath, subpath, args):
    scalar_accumulators = [EventAccumulator(str(dpath / dname / subpath)).Reload(
    ).scalars for dname in os.listdir(dpath) if dname != FOLDER_NAME]

    # Filter non event files
    scalar_accumulators = [scalar_accumulator for scalar_accumulator in scalar_accumulators if scalar_accumulator.Keys()]

    # Get and validate all scalar keys
    all_keys = [tuple(scalar_accumulator.Keys()) for scalar_accumulator in scalar_accumulators]
    print("set of all keys ", all_keys)
    all_keys_set = [set(scalar_accumulator.Keys()) for scalar_accumulator in scalar_accumulators]
    #assert len(set(all_keys_set)) == 1, "All runs need to have the same scalar keys. There are mismatches in {}".format(all_keys_set)
    keys = all_keys[0]
    if args.allowed_keys:
        allowed_keys =  args.allowed_keys
    else:
        allowed_keys = ['evaluate',
                        'standard_evaluate',
                        'forgetting_metric', 
                        'weight_stat']
    found_keys=[]
    for key in all_keys_set[0]:
        # Check if current key occurs starts with any of allowed keys.
        log_key = (len(list(filter(lambda x: key.startswith(x), allowed_keys))) > 0)
        if log_key:
            present_in_all = True
            for av_set in all_keys_set:
                if not key in av_set:
                    present_in_all = False
            if present_in_all:
                found_keys.append(key)
    keys=found_keys

    keys_list = all_keys[0]

    all_scalar_events_per_key = [[scalar_accumulator.Items(key) for scalar_accumulator in scalar_accumulators] for key in keys]

    # Get and validate all steps per key
    all_steps_per_key = [[tuple(scalar_event.step for scalar_event in scalar_events) for scalar_events in all_scalar_events]
                         for all_scalar_events in all_scalar_events_per_key]

    for i, all_steps in enumerate(all_steps_per_key):
        print(i, all_steps)
        assert len(set(all_steps)) == 1, "For scalar {} the step numbering or count doesn't match. Step count for all runs: {}".format(
            keys[i], [len(steps) for steps in all_steps])
        #del keys_list[i]

    all_scalar_events_per_key = [[scalar_accumulator.Items(key) for scalar_accumulator in scalar_accumulators] for key in keys]
    print(all_scalar_events_per_key)

    # Get and validate all steps per key
    all_steps_per_key = [[tuple(scalar_event.step for scalar_event in scalar_events) for scalar_events in all_scalar_events]
                         for all_scalar_events in all_scalar_events_per_key]
    steps_per_key = [all_steps[0] for all_steps in all_steps_per_key]

    # Get and average wall times per step per key
    wall_times_per_key = [np.mean([tuple(scalar_event.wall_time for scalar_event in scalar_events) for scalar_events in all_scalar_events], axis=0)
                          for all_scalar_events in all_scalar_events_per_key]

    # Get values per step per key
    values_per_key = [[[scalar_event.value for scalar_event in scalar_events] for scalar_events in all_scalar_events]
                      for all_scalar_events in all_scalar_events_per_key]

    all_per_key = dict(zip(keys, zip(steps_per_key, wall_times_per_key, values_per_key)))

    return all_per_key


def aggregate_to_summary(dpath, aggregation_ops, extracts_per_subpath, args):
    for op in aggregation_ops:
        for subpath, all_per_key in extracts_per_subpath.items():
            path = Path(args.store_path) / FOLDER_NAME / op.__name__ / dpath.name / subpath
            aggregations_per_key = {key: (steps, wall_times, op(values, axis=0)) for key, (steps, wall_times, values) in all_per_key.items()}
            write_summary(path, aggregations_per_key)


def write_summary(dpath, aggregations_per_key):
    writer = tf.summary.FileWriter(dpath)

    for key, (steps, wall_times, aggregations) in aggregations_per_key.items():
        for step, wall_time, aggregation in zip(steps, wall_times, aggregations):
            summary = tf.Summary(value=[tf.Summary.Value(tag=key, simple_value=aggregation)])
            scalar_event = Event(wall_time=wall_time, step=step, summary=summary)
            writer.add_event(scalar_event)

        writer.flush()
    writer.close()


def aggregate_to_csv(dpath, aggregation_ops, extracts_per_subpath, args):
    print(" Got in aggregations function")
    fullname = dpath.name
    sub_parts = fullname.split('__')
    exper = sub_parts[0]
    main_sub_parts = sub_parts[1].split('_')
    layer = main_sub_parts[0]
    hdim = main_sub_parts[2]
    code = sub_parts[2]
    for subpath, all_per_key in extracts_per_subpath.items():
        for key, (steps, wall_times, values) in all_per_key.items():
            aggregations = [op(values, axis=0) for op in aggregation_ops]
            agg_final = {}
            print("This is tracking , : ", key, steps, list(aggregations[0]))
            for step, agg in zip(steps, list(aggregations[0])):
                agg_final['step_' + str(step)] = agg

            agg_final['exper'] = exper
            agg_final['layer'] = layer
            agg_final['hdim'] = hdim
            agg_final['code'] = code
            print("Steps things, ", agg_final)

            write_csv(dpath, subpath, key, dpath.name, aggregations, steps, aggregation_ops, args, [agg_final])


def get_valid_filename(s):
    s = str(s).strip().replace(' ', '_')
    return re.sub(r'(?u)[^-\w.]', '', s)


def write_csv(dpath, subpath, key, fname, aggregations, steps, aggregation_ops, args, agg_df):
    path = Path(args.store_path) / FOLDER_NAME / dpath.name 
    
    if not path.exists():
        os.makedirs(path)


    file_name = get_valid_filename(key) + '-' + get_valid_filename(subpath) + '-' + fname + '.csv'
    print("Writing csv for file : ", file_name)
    aggregation_ops_names = [aggregation_op.__name__ for aggregation_op in aggregation_ops]
    df = pd.DataFrame(np.transpose(aggregations), index=steps, columns=aggregation_ops_names)
    df.to_csv(path / file_name, sep=';')

    if args.store_df:
        df_path = Path(args.store_path) / FOLDER_NAME / (key + '.df')
        df_path_dir = Path(args.store_path) / FOLDER_NAME / key
        if not df_path_dir.exists():
            os.makedirs(df_path_dir)

        print("Store DF", df_path, os.path.isfile(df_path))
        if os.path.isfile(df_path):
            key_df = pd.read_pickle(df_path)
        else:
            key_df = pd.DataFrame()
        df_stor = pd.DataFrame(agg_df)
        print("Store_df ", df_stor)
        if not df_stor.empty:
            if key_df.empty:
                df_stor.to_pickle(df_path)
            else:
                print("old_df ", key_df)
                key_df = key_df.append(df_stor, ignore_index=True)
                print("New_df ", key_df)
                key_df.to_pickle(df_path)


def aggregate(dpath, args):
    name = dpath.name

    if args.operations:
        ## TODO: Convert from operations in string to actual np operations
        aggregation_ops = args.operations
        aggregation_ops = [np.mean]
    else:
        aggregation_ops = [np.mean, np.min, np.max, np.median, np.std, np.var]
        aggregation_ops = args.operations

    ops = {
        'summary': aggregate_to_summary,
        'csv': aggregate_to_csv
    }

    print("Started aggregation {}".format(name))

    extracts_per_subpath = {subpath: extract(dpath, subpath, args) for subpath in args.subpaths}
    print("Subpaths extracted ", extracts_per_subpath)

    ops.get(args.output)(dpath, aggregation_ops, extracts_per_subpath, args)

    print("Ended aggregation {}".format(name))


if __name__ == '__main__':
    def param_list(param):
        print("Doing param", param)
        p_list = ast.literal_eval(param)
        if type(p_list) is not list:
            raise argparse.ArgumentTypeError("Parameter {} is not a list".format(param))
        return p_list

    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, help="main path for tensorboard files", default=os.getcwd())
    parser.add_argument("--subpaths", type=str, action='append', help="subpath sturctures", default=["log/train"])
    parser.add_argument("--allowed_keys", type=str, action='append', help="Keys to aggregate on")
    parser.add_argument("--operations", type=str, action='append', help="Default operations to perform.")
    parser.add_argument("--store_path", type=str, help="Default store path for arguments.")
    parser.add_argument("--store_df", action='store_true', help="Makes us to store the values in dataframe for analysis.")
    parser.add_argument("--output", type=str, help="aggregation can be saves as tensorboard file (summary) or as table (csv)", default='summary')

    args = parser.parse_args()
    print("Running with args", args)

    path = Path(args.path)

    if not path.exists():
        raise argparse.ArgumentTypeError("Parameter {} is not a valid path".format(path))

    subpaths = [path / dname / subpath for subpath in args.subpaths for dname in os.listdir(path) if dname != FOLDER_NAME]

    for subpath in subpaths:
        if not os.path.exists(subpath):
            raise argparse.ArgumentTypeError("Parameter {} is not a valid path".format(subpath))

    if args.output not in ['summary', 'csv']:
        raise argparse.ArgumentTypeError("Parameter {} is not summary or csv".format(args.output))

    aggregate(path, args)
