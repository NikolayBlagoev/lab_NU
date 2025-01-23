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



