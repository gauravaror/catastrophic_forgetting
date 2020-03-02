import sys
from allennlp.training.util import evaluate
from allennlp.data.iterators import BucketIterator
import torch
import pandas as pd
def evaluate_all_tasks(task, evaluate_tasks, dev_data, vocabulary, model, args, save_weight):
    devicea = -1
    if torch.cuda.is_available():
        devicea = 0
    majority = {
                 'subjectivity': 0.5,
                 'sst': 0.2534059946,
                 'trec': 0.188,
                 'cola': 0,
                 'ag': 0.25,
                 'sst_2c': 0.51
                }

    sota = {
                 'subjectivity': 0.955,
                 'sst': 0.547,
                 'trec': 0.9807,
                 'cola': 0.341,
                 'ag' : 0.955 ,
                 'sst_2c': 0.968
            }

    overall_metric = {}
    standard_metric = {}
    for j in evaluate_tasks:
        model.set_task(j)
        print("\nEvaluating ", j)
        sys.stdout.flush()
        print("Now evaluating ", j, len(dev_data[j]))
        iterator1 = BucketIterator(batch_size=args.bs, sorting_keys=[("tokens", "num_tokens")])
        iterator1.index_with(vocabulary[j])
        metric = evaluate(model=model,
                          instances=dev_data[j],
                          data_iterator=iterator1,
                          cuda_device=devicea,
                          batch_weight_key=None)

        # Take first 500 instances for evaluating activations.
        if not args.no_save_weight:
            iterator1 = BucketIterator(batch_size=500, sorting_keys=[("tokens", "num_tokens")])
            iterator1.index_with(vocabulary[j])
            evaluate(model=model,
                     instances=dev_data[j][:500],
                     data_iterator=iterator1,
                     cuda_device=devicea,
                     batch_weight_key=None)
            save_weight.add_activations(model, task,j)

        if j == 'cola':
            metric['metric'] = metric['average']
        else:
            metric['metric'] = metric['accuracy']
        smetric = (float(metric['metric']) - majority[j]) / (sota[j] - majority[j])
        overall_metric[j] = metric
        standard_metric[j] = smetric
    return overall_metric, standard_metric

        
def print_evaluate_stats(train, evaluate_tasks, args, overall_metrics, task_code, experiment):
    print("Accuracy and Loss")
    header="Accuracy"
    for i in evaluate_tasks:
      header = header + "\t\t" + i
    insert_in_pandas_list=[]
    print(header)
    for d in train:
      print_data=d
      insert_pandas_dict={'code': task_code, 'layer': args.layers, 'h_dim': args.h_dim, 'task': d, 'try': args.tryno, 'experiment': experiment, 'metric': 'accuracy'}
      i=0
      for k in evaluate_tasks:
        print_data = print_data + "\t\t" + '{:.2}'.format(overall_metrics[d][k]["metric"])
        insert_pandas_dict[k] = overall_metrics[d][k]["metric"]
      insert_in_pandas_list.append(insert_pandas_dict)
      print(print_data)
    print("\n\n")
    initial_path="dfs/Results" + experiment + "_" + args.run_name
    df=pd.DataFrame(insert_in_pandas_list)
    df.to_pickle(path=str(initial_path+task_code+"_"+str(args.layers)+"_"+str(args.h_dim)+"_"+str(args.tryno)+".df"))

