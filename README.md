# Distributed Machine Learning LAB

This repo hosts all relevant files about the distributed ML Lab in Neuchatel.

## Building LLMs

For this lab we will use a library to help us build Large Language Models and train them easily. You can download [simplellm](https://github.com/NikolayBlagoev/simplellm/tree/main) via `git clone` and then install with `pip install .`

Let us set up a simple model and dataset which we will use throughout this lab (the file can be found in [primer/intro.py](primer/intro.py))
```python
from simplellm.llama import CausalLLama, LLama # get our models
from simplellm.tokenizers import SPTokenizer # get our tokenizer
from simplellm.dataloaders import TinyStories # get our dataset
from torch.optim import SGD
import torch.nn.functional as F
dmodel = 288
num_heads = 6
n_layers = 6
seq_l = 256
batch_size = 1
device = "cuda"

# make the tokenizer
tokenizer = SPTokenizer()
# make the model
net = LLama(CausalLLama,tokenizer.vocab_size,dmodel=dmodel,num_heads=num_heads,
                device=device, n_layers=n_layers, ctx_size=seq_l)
ds = TinyStories(tokenizer,batch_size=batch_size, seq_l=seq_l)
# we can iterate the dataset with:
iter_ds = iter(ds)
optim = SGD(net.parameters(),lr=4e-3,momentum=0, dampening=0,weight_decay=0,nesterov=False)
for _ in range(10_000):
    optim.zero_grad()
    x,y = next(iter_ds)
    x = x.to(device)
    x = net(x)
    B, T, C = x.shape
    x = x.view(B*T,C)
    y = y.view(B*T).to(device)
    #compute loss:
    loss = F.cross_entropy(x,y)
    # log the loss:
    print(loss.item())
    loss.backward()
    optim.step()

```

Throughout this lab we will be expanding this implementation

## Primer on Batched Stochastic Gradient Descent

As we all remember, when performing SGD we do the following steps:

1. Run a forward pass through our model for a given sample $x$
2. Compute the loss $L(x,y)$ given some target $y$ 
3. Compute the gradient per weight $ \frac{\delta y}{\delta W}  $
4. Update the weights with $ W = W - \lambda \frac{\delta y}{\delta W}$ where $\lambda$ is the learning rate

When we use a batch of size larger than 1 the same process is performed, except that the gradient upate is the average across all the gradients for the different samples. This gives less noisy updates over time. The larger the batch is usually better (though there are things such as TOO LARGE of a batch, but in the context of LLMs this is hardly ever the case).

## From Federated to Data Parallel

Previously we learnt about federated learning as a means of preserving the privacy of each participant. Data Parallelism (DP) is pretty much the same as Federated Learning, except the concern is about throughput and not privacy, thus making our lives a lot easier when implementing DP solutions.

As we mentioned the larger your batch, usually the better. But with large models, the GPU memory is usually your bottleneck. So you quite often can't even do a batch of more than 2 samples. This however would make our gradient updates too noisy and lead to poor convergence. 

A very obvious solution is - if one GPU is not enough, let's use multiple. This is usually how problems in distributed systems are solved. Similar to federated learning, we will have multiple nodes (or workers or devices if you prefer) perform training locally on some subset of shared data and then at the end of an iteration, average their gradients, before performing an update step. It is easy to see how this is equivalent to increasing your batch size in batched SGD. In the context of DP, each device is said to train a mini-batch, with the global size of the batch being the summation of all the mini-batches. So, if you have 4 devices, each training with a mini-batch of size 4 samples, your global batch size is 16 and is equivalent to doing batched SGD with batch size of 16.

In DP all workers perform their local iteration in parallel, thus you can have a speed up with the number of devices. However, you pay with an increase in communication, as each device will need to communicate with every other their updates during the aggregation phase. There is extensive literature on optimising this phase, as it can be incredibly costly.

Let us see how to implement this in torch. First we need to set up our group of workers that will be communicating with each other. PyTorch provides an abstraction for this for you with their [torch.distributed](https://pytorch.org/docs/stable/distributed.html) package. They support three backends - gloo, nccl, and mpi. Use gloo if you want to do cpu to cpu communication (and mpi if gloo is failing). Use nccl if you are doing gpu to gpu communication. Fair warning, nccl does not allow a gpu to itself communication so if you are hosting multiple nodes on the same gpu device, nccl will not work. Therefore for our demos we will be using the gloo backend. In practice, however, you should use nccl as it has significantly faster throughput.

When initialising the distributed communication you need to provide 5 things to torch - the desired backend, the world size (how many devices will participate), the rank of the current device (unique per device in the range of $[0,world size]$), the master address and port. The last two are used when devices need to discover each other. One device (the master) will need to bind to the given address and listen for incoming messages. To do it in code is quite simple:

```python
import torch.distributed as dist
import os
from sys import argv
rank = int(argv[1])
os.environ["MASTER_ADDR"] = "localhost"
os.environ["MASTER_PORT"] = "29500"
dist.init_process_group("gloo", rank=rank, world_size=3)
```

The process will block until 3 different processes reach the init_process_group statement. 

The next thing we need to modify is at the weight update step. We need to synchronise across all devices. Thus we will modify that part of our primer to:

```python
dist.barrier() # wait for everyone
tmp = []
for param in net.parameters():
    if param.grad == None:
        tmp.append(torch.zeros_like(param).view(-1))                      
        continue
    tmp.append(param.grad.view(-1))
    param.grad = None
prev_grad = torch.cat(tmp).to("cpu")
dist.all_reduce(prev_grad, op = dist.ReduceOp.SUM)
```

The last line runs an [all reduce](https://en.wikipedia.org/wiki/Collective_operation#Reduce), which will make the `prev_grad` tensor the same across all devices. What we would like is to reduce to the average across all devices, however gloo supports only sum. So we would need to add an extra operation to average the prev_grad tensor.

The working full file is available in [intro_DP_GA.py](DP/gradient_aggr/intro_DP_GA.py):

```python
from simplellm.llama import CausalLLama, LLama # get our models
from simplellm.tokenizers import SPTokenizer # get our tokenizer
from simplellm.dataloaders import TinyStories # get our dataset
from torch.optim import SGD
import torch.nn.functional as F
import torch
import torch.distributed as dist
import os
from sys import argv
rank = int(argv[1])
os.environ["MASTER_ADDR"] = "localhost"
world_size = 3
os.environ["MASTER_PORT"] = "29500"
dist.init_process_group("gloo", rank=rank, world_size=world_size)
torch.manual_seed(0)
dmodel = 288
num_heads = 6
n_layers = 6
seq_l = 256
batch_size = 1
device = "cuda"

# make the tokenizer
tokenizer = SPTokenizer()
# make the model
net = LLama(CausalLLama,tokenizer.vocab_size,dmodel=dmodel,num_heads=num_heads,
                device=device, n_layers=n_layers, ctx_size=seq_l)
ds = TinyStories(tokenizer,batch_size=batch_size, seq_l=seq_l)
# we can iterate the dataset with:
iter_ds = iter(ds)
for _ in range(rank): # offset dataset
    next(iter_ds)
optim = SGD(net.parameters(),lr=4e-3,momentum=0, dampening=0,weight_decay=0,nesterov=False)

sizes = []
len_sizes = []
for param in net.parameters():
    sizes.append(param.shape)
    len_sizes.append(len(param.view(-1)))

for _ in range(10_000):
    optim.zero_grad()
    x,y = next(iter_ds)
    x = x.to(device)
    x = net(x)
    B, T, C = x.shape
    x = x.view(B*T,C)
    y = y.view(B*T).to(device)
    #compute loss:
    loss = F.cross_entropy(x,y)
    # log the loss:
    print(loss.item())
    loss.backward()
    
    dist.barrier() # wait for everyone
    tmp = []
    for param in net.parameters():
        if param.grad == None:
            tmp.append(torch.zeros_like(param).view(-1))                      
            continue
        tmp.append(param.grad.view(-1))
        param.grad = None
    prev_grad = torch.cat(tmp).to("cpu")
    dist.all_reduce(prev_grad, op = dist.ReduceOp.SUM)
    tmp = torch.split(prev_grad, len_sizes)
    for i, param in enumerate(net.parameters()):
        param.grad = tmp[i].view(sizes[i]).to(device)/world_size # average
    optim.step()

```


As we learnt in Federated Learning, it is also possible to synchronise on model weights, rather than on gradients. The modification to the above code to accomodate is simple and can be found in [intro_DP_WA.py](DP/weight_aggr/intro_DP_WA.py)

Torch provides a nice abstraction to our own implementation for [Data Parallel Training](https://pytorch.org/docs/stable/generated/torch.nn.parallel.DistributedDataParallel.html#torch.nn.parallel.DistributedDataParallel). From personal experience, the timing between the two is almost identical. It is also a bit more limiting as it only synchronises gradients and not weights and has a few quirks to it. 

### What to remember

Data Parallelism is useful when you want to increase your batch size by training independently on multiple devices. It can also provide a speed up in computation by parallelising across devices, at the cost of greater overhead in communication. In practical applications, Data Parallel communication is the bottleneck of your training.

### A small note on optimizers

So far we have used the vanilla SGD optimizer (no momentum, no dampening, no nothing). This is purely so that we can create a comparison between Weight Aggregation and Gradient Aggregation. Both have their strengths, but a major one of Gradient Aggregation is that since the gradients will be identical across all devices, we can use better optimizers like Adam, without the need to synchronise the states of the optimizer. This is trivial to see, since the optimizer state is updated deterministically and depends only on the gradient of each parameter.


## Pipeline Parallelism

We learnt about data parallelism. But what if we cannot fit our model on a single device, because it is just **SO** big. We could try offloading, but that is for people who don't have access to multiple devices. We are in distributed systems. We have as many devices as we ~~want~~ can afford. So let us think of a smart way we can make use of our devices to run a model that doesn't fit on one of them.


