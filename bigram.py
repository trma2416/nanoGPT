#to activate venv run $source GPT_env/bin/activate$ into terminal
#then to run code $python src/bigram.py$ 
import torch
import torch.nn as nn
from torch.nn import functional as F
 
batch_size = 64
block_size = 16
max_iters = 5000
eval_interval = 200
learning_rate = 1e-5
device = 'cuda' if torch.cuda.is_available() else 'cpu'
if device == 'cuda':
    print("GPU selected")
else:
    print("Running on CPU")
eval_iters = 200
n_embed = 32
dropout = 0.2 #drops out 20% of neurons at each training pass
n_layer = 6
n_head = 6


 
torch.manual_seed(1337)
##read in our source file
with open('src/input.txt', 'r', encoding='utf-8') as f:
   text = f.read()
 
## get unique chars
chars = sorted(list(set(text)))
vocab_size = len(chars)

# mapping form characters to integers
stoi = { ch: i for i,ch in enumerate(chars)} ##lookup table from character to integer (ch to i)
itos = { i:ch for i, ch in enumerate(chars)} ## enumerate gives an integer count starting from the first elem in char to last with start at 0
# encode goes from string (st) to list
encode = lambda st: [stoi[c] for c in st] ## create a list of integers using lookup table

#decode goes from list (lst) to string
decode = lambda lst: ''.join([itos[i] for i in lst])
## encode input data set
data = torch.tensor(encode(text), dtype=torch.long)

# lets set aside some data for validation 
n = int(0.9*len(data))
#first 90% will be training data, rest will be validation
train_data = data[:n]
val_data = data[n:]

#-------Helper Functions ---------------
# generate batches of data to work with
def get_batch(split):
    #generate small batches of data split defines if were getting training data or validation data
    data = train_data if split == "train" else val_data
    #get random chuncks of data (batches)
    ix = torch.randint(len(data) -  block_size, (batch_size,))
    #context
    x = torch.stack([data[i:i+block_size] for i in ix])
    #targets
    y = torch.stack([data[i + 1: i + block_size + 1] for i in ix])
    x,y = x.to(device), y.to(device)
    return x,y

## get average loss for eval and train data 
@torch.no_grad() ## this allows pytorch to not perform back propagation when we dont want it
def estimate_loss():
    out = {}
    model.eval()
    for split in ["train", "val"]:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

def train(model, optimizer, max_iters, eval_int):
    for iter in range(max_iters):
        # trains val sets every once and a while (when iter divides eval_interval)
        if iter % eval_int == 0:
            losses = estimate_loss()
            print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
        
        xb, yb = get_batch('train')
    
        #eval loss
        logits, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        
#---------------Class defintions-----------------

class Head(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embed, head_size, bias=False)
        self.query = nn.Linear(n_embed, head_size, bias=False)
        self.value = nn.Linear(n_embed, head_size, bias=False)
        self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size))) #lower triangular matrix
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)
        q = self.query(x)
        v = self.query(x)
        
        wei = q @ k.transpose(-2, -1) * C**-0.5 ## normalization to multiply by the inverse sqrt(C) scaled attention
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        v = self.value(x)
        out = wei @ v
        return out
            
        
class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)]) ## a list of heads to perform attention in parallel
        self.proj = nn.Linear(n_embed, n_embed)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1) ## concatenate all outputs over channel dimension
        out = self.proj(out) #linear transformation of out in prev line
        return out
        
    
class FeedForward(nn.Module):
    def __init__(self, n_embed):
        super().__init__()
        ## this is neural network where we feed in n_embed (embedded layer) to a linear layer, then the output
        ## of that gets fed into our activation function with is a ReLu
        self.net = nn.Sequential(
            nn.Linear(n_embed, 4*n_embed),
            nn.ReLU(),
            nn.Linear(4*n_embed, n_embed),
            nn.Dropout(dropout),)
        
    def forward(self, x):
        return self.net(x)
    
class Block(nn.Module):
    def __init__(self, n_embed, n_head):
        super().__init__()
        head_size = n_embed // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedForward(n_embed)
        self.ln1 = nn.LayerNorm(n_embed)
        self.ln2 = nn.LayerNorm(n_embed)
        
        
    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x
    
    
class BigramLanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        ## each token directly reads off the logits for the next token from a lookup table
        
        # each number in our input tensor will select a row at index of its value
        self.token_embedding_table = nn.Embedding(vocab_size, n_embed) ## tensor of 8x8
        self.position_embedding_table = nn.Embedding(block_size, n_embed)
        self.blocks = nn.Sequential(*[Block(n_embed, n_head=n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embed),
        
        self.lm_head = nn.Linear(n_embed, vocab_size) ## linear layer
        #self.attn_heads = nn. may need to revisit this
        
    def forward(self, idx, targets = None):
        B, T = idx.shape
        #predicting what comes next based on identity of single token 
        #idx and targes are both (B, T) tensor of integers
        tkn_embed = self.token_embedding_table(idx) #Tensor of dim (B, T, C) (4, 8, vocab_size = 65)
        pos_embed = self.position_embedding_table(torch.arange(T, device=device)) #(T, C)
        x = tkn_embed + pos_embed # (B, T, C)
        #x = self.attn_heads(x)
        logits = self.lm_head(x) # (B, T, vocab_size)
        
        if targets is None:
            loss = None
        else:
            # cross entropy wants (BxT, C)
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)
        
        return logits, loss
    
    def generate(self, idx, max_new_tkns):
        for _ in range(max_new_tkns):
            idx_cond = idx[:,-block_size:]
            logits, loss = self(idx_cond)
            logits = logits[:, -1, :] # becomes (B, C)
            
            #get probability 
            probs = F.softmax(logits, dim = -1) 
            #new tensor produced 
            idx_next = torch.multinomial(probs, num_samples=1) #(B, 1)
            #concatenate tensors
            idx = torch.cat((idx, idx_next), dim=1) #(B, T + 1)
        return idx
    
    
    
#tests
model = BigramLanguageModel()
m = model.to(device)
optimizer = torch.optim.Adam(m.parameters(), lr =learning_rate)
train(model=model, optimizer=optimizer, max_iters=max_iters, eval_int=eval_interval)
    
#generate from model
context = torch.zeros((1,1), dtype = torch.long, device=device)
print("----------------------------------------")
print(decode(m.generate(context, 500)[0].tolist()))
print("----------------------------------------")


