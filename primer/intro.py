from simplellm.llama import CausalLLama, LLama # get our models
from simplellm.tokenizers import SPTokenizer # get our tokenizer
from simplellm.dataloaders import TinyStories # get our dataset
from simplellm.losses import causalLLMLoss # our loss
from torch.optim import Adam
import torch.nn.functional as F
dmodel = 288
num_heads = 6
n_layers = 6
seq_l = 256
batch_size = 3
device = "cuda"

# make the tokenizer
tokenizer = SPTokenizer()
# make the model
net = LLama(CausalLLama,tokenizer.vocab_size,dmodel=dmodel,num_heads=num_heads,
                device=device, n_layers=n_layers, ctx_size=seq_l,padding_idx=tokenizer.pad_id)
ds = TinyStories(tokenizer,batch_size=batch_size, seq_l=seq_l)
# we can iterate the dataset with:
iter_ds = iter(ds)
optim = Adam(net.parameters(),lr=8e-4)
for itr in range(5_000):
    optim.zero_grad()
    x = next(iter_ds)
    x = x.to(device)
    target = x.clone().detach()
    x = net(x)
    loss = causalLLMLoss(x,target,tokenizer.vocab_size)
    # log the loss:
    print(itr,loss.item())
    loss.backward()
    optim.step()



