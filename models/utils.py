from collections import Counter

def get_catastrophic_metric(tasks, metrics):
     forgetting_metrics = Counter()
     count_task = Counter()
     forgetting={'total': 0, '1_step': 0}
     last_task = tasks[len(tasks)-1]

     for i,task in enumerate(tasks):
        # Calculate forgetting between first trained and last trained task
        current_forgetting = (metrics[tasks[i]][tasks[i]] - metrics[tasks[i]][last_task])
        # Normalize it by it's value.
        current_forgetting = current_forgetting/metrics[tasks[i]][tasks[i]]
        #print(f'Got forgetting for task {task} :  {current_forgetting}')
        # This finds number of tasks trained after current task.
        # This is to find expected loss per trained class.
        if current_forgetting == 0:
            continue
        number_training_steps = (len(tasks) - i - 1)
        forgetting_metrics["1_step"] += (current_forgetting / number_training_steps)
        forgetting_metrics["total"] += current_forgetting
        forgetting_metrics[tasks[i]] = current_forgetting
     
     # Calculate total forgetting of all the
     length_tasks = len(tasks) - 1
     if length_tasks > 1:
         # This is subtracted by 1 as last task never sees any forgetting.
         forgetting_metrics['total'] = forgetting_metrics['total']
         forgetting_metrics['1_step'] = (forgetting_metrics['1_step']/(len(tasks) - 1))

     return forgetting_metrics
