import torch
import matplotlib.pyplot as plt
x = torch.randn(10)
soft = torch.nn.Softmax()
temps = [1,10,100,0.5,0.1,0.01]                                                                                                                      

f = plt.figure()
# Set common labels
f.text(0.5, 0.04, 'Logits', ha='center', va='center')
f.text(0.06, 0.5, 'Softmax Probability', ha='center', va='center', rotation='vertical')
for idx,temp in enumerate(temps,1): 
    print(idx, temp) 
    fig = f.add_subplot("23"+str(idx)) 
    fig.plot(soft(x/temps[idx-1])) 
    fig.set_title("Temperature: " + str(temps[idx-1])) 

plt.show()
