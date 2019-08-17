from collections import Counter
from sklearn.manifold import TSNE
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import tensorflow as tf
import io
import PIL.Image
from torchvision.transforms import ToTensor

def gen_plot(plt):
    """Create a pyplot plot and save to buffer."""
    buf = io.BytesIO()
    plt.savefig(buf, format='jpeg')
    buf.seek(0)
    image = PIL.Image.open(buf)
    image = ToTensor()(image).unsqueeze(0)
    return image

def run_tsne_embeddings(data_view_tsne, labels_orig, train, evaluate, getlayer, gram):
  print("Running TSNE plotting")
  tnse_embedding = TSNE(n_components=2, perplexity=30.0).fit_transform(data_view_tsne)
  print("Running TSNE got tsne embeddings back", tnse_embedding)
  for i in range(0, len(tnse_embedding)):
    print(i,tnse_embedding[i],labels_orig[i])
    if labels_orig[i] == 1:
      plt.plot(tnse_embedding[i][0], tnse_embedding[i][1],"ro")
    elif labels_orig[i] == 2:
      plt.plot(tnse_embedding[i][0], tnse_embedding[i][1],"go")
    elif labels_orig[i] == 3:
      plt.plot(tnse_embedding[i][0], tnse_embedding[i][1],"yo")
    else:
      plt.plot(tnse_embedding[i][0], tnse_embedding[i][1],"bo")
  plt.savefig("embedding_plot_"+ "_layer_" + str(getlayer)+ "_train_"+ train + "_evaluate_" + evaluate + '_gram_' + str(gram) + ".png")
  return gen_plot(plt)

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
             if step > 0:
               forgetting_metrics[str(step) + "_step"] += current_forgetting
               count_task[str(step) + "_step"] += 1
             forgetting_metrics[tasks[j]] += current_forgetting
             count_task[tasks[j]] += 1
     for metric in forgetting_metrics:
         if metric in tasks:
           print("Calculating forgetting for", forgetting_metrics[task], count_task[task])
           forgetting[metric] = forgetting_metrics[metric] / count_task[metric]
           forgetting['total'] += forgetting[metric]
         else:
           forgetting[metric] = forgetting_metrics[metric] / count_task[metric]
     
     # Calculate total forgetting of all the
     length_tasks = len(tasks) - 1
     if length_tasks > 1:
         forgetting['total'] = (forgetting['total']/(len(tasks) - 1))

     return forgetting
