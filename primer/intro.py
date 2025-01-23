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



