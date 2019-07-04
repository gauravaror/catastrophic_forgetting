from collections import Counter

def get_catastrophic_metric(tasks, metrics):
     forgetting_metrics = Counter()
     count_task = Counter()
     forgetting={'total': 0}
     for i,task in enumerate(tasks):
         for j in range(i):
             print("Calculating backward for ",tasks[i] ,
                   " taking calculation at tasks", tasks[j])
             step = i-j
             current_forgetting = (metrics[tasks[j]][tasks[j]] - metrics[tasks[i]][tasks[j]])
             if step > 0 and (current_forgetting > 0):
               forgetting_metrics[str(step) + "_step"] += current_forgetting
               count_task[str(step) + "_step"] += 1
             forgetting_metrics[tasks[j]] += current_forgetting
             count_task[tasks[j]] += 1
     for metric in forgetting_metrics:
         if metric in tasks:
           print("Calculating forgetting for", forgetting_metrics[task], count_task[task])
           forgetting[metric] = forgetting_metrics[metric] / count_task[metric]
           forgetting['total'] += forgetting[metric]
         forgetting[metric] = forgetting_metrics[metric] / count_task[metric]
     
     # Calculate total forgetting of all the
     forgetting['total'] = (forgetting['total']/(len(tasks) - 1))

     return forgetting
