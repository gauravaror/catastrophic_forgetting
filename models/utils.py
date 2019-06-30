from collections import Counter

def get_catastrophic_metric(tasks, metrics):
     forgetting_metrics = Counter()
     count_task = Counter()
     forgetting={}
     for i,task in enumerate(tasks):
         for j in range(i):
             print("Calculating backward for ",tasks[i] ,
                   " taking calculation at tasks", tasks[j])
             forgetting_metrics[tasks[j]] += (metrics[tasks[j]][tasks[j]] - metrics[tasks[j]][tasks[i]])
             count_task[tasks[j]] += 1
     for task in tasks[:-1]:
         print("Calculating forgetting for", forgetting_metrics[task], count_task[task])
         forgetting[task] = forgetting_metrics[task] / count_task[task]
     return forgetting
