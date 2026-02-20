#!/usr/bin/env python
# coding: utf-8

# # Exercise 3: Neural networks in PyTorch
# 
# In this exercise you’ll implement small neural-network building blocks from scratch and use them to train a simple classifier.
# 
# You’ll cover:
# - **Basic layers**: Linear, Embedding, Dropout
# - **Normalization**: LayerNorm and RMSNorm
# - **MLPs + residual**: composing layers into deeper networks
# - **Classification**: generating a learnable dataset, implementing cross-entropy from logits, and writing a minimal training loop
# 
# As before: fill in all `TODO`s without changing function names or signatures.
# Use small sanity checks and compare to PyTorch reference implementations when useful.

# In[1]:


from __future__ import annotations

import torch
from torch import nn


# ## Basic layers
# 
# In this section you’ll implement a few core layers that appear everywhere:
# 
# ### `Linear`
# A fully-connected layer that follows nn.Linear conventions:  
# `y = x @ Wᵀ + b`
# 
# Important details:
# - Parameters should be registered as `nn.Parameter`
# - Store weight as (out_features, in_features) like nn.Linear.
# - The forward pass should support leading batch dimensions: `x` can be shape `(..., in_features)`
# 
# ### `Embedding`
# An embedding table maps integer ids to vectors:
# - input: token ids `idx` of shape `(...,)`
# - output: vectors of shape `(..., embedding_dim)`
# 
# This is essentially a learnable lookup table.
# 
# ### `Dropout`
# Dropout randomly zeroes activations during training to reduce overfitting.
# Implementation details:
# - Only active in `model.train()` mode
# - In training: drop with probability `p` and scale the kept values by `1/(1-p)` so the expected value stays the same
# - In eval: return the input unchanged
# 
# ## Instructions
# - Do not use PyTorch reference seq for the parts you implement (e.g. don’t call nn.Linear inside your Linear).
# - You may use standard tensor ops that you learned before (matmul, sum, mean, rsqrt, indexing, etc.).
# - Use a parameter initialization method of your choice. We recommend something like Xavier-uniform.
# 

# In[21]:


class Linear(nn.Module):
    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        weight = torch.zeros(size=(out_features, in_features))
        torch.nn.init.xavier_uniform_(weight)
        self.weight = nn.Parameter(weight)
        self.bias = None
        if bias:
            self.bias = nn.Parameter(torch.zeros([out_features]))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (..., in_features)
        return: (..., out_features)
        """
        result = x@self.weight.T
        if self.bias is not None:
            result += self.bias
        return result

Linear(5,4).forward(torch.ones(1,5))


# In[23]:


class Embedding(nn.Module):
    def __init__(self, num_embeddings: int, embedding_dim: int):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(num_embeddings, embedding_dim))
        torch.nn.init.xavier_uniform_(self.weight)

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        """
        idx: (...,) int64
        return: (..., embedding_dim)
        """
        return self.weight[idx]

Embedding(10,3).forward(torch.tensor([[[0,1,1],[0,1,2]],[[0,1,1],[0,1,2]]]))


# In[43]:


class Dropout(nn.Module):
    def __init__(self, p: float):
        super().__init__()
        self.p = p

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        In train mode: drop with prob p and scale by 1/(1-p).
        In eval mode: return x unchanged.
        """
        if not self.training:
            return x
        prob = torch.zeros_like(x)
        prob.uniform_(0,1)
        return x.where(prob > self.p, 0) * (1/(1-self.p))

dropout = Dropout(0.3)
dropout.train()
dropout.forward(torch.ones(1,5,5))



# ## Normalization
# 
# Normalization layers help stabilize training by controlling activation statistics.
# 
# ### LayerNorm
# LayerNorm normalizes each example across its **feature dimension** (the last dimension):
# 
# - compute mean and variance over the last dimension
# - normalize: `(x - mean) / sqrt(var + eps)`
# - apply learnable per-feature scale and shift (`weight`, `bias`)
# 
# **In this exercise, assume `elementwise_affine=True` (always include `weight` and `bias`).**  
# `weight` and `bias` each have shape `(D,)`.
# 
# LayerNorm is widely used in transformers because it does not depend on batch statistics.
# 
# ### RMSNorm
# RMSNorm is similar to LayerNorm but normalizes using only the root-mean-square:
# - `x / sqrt(mean(x^2) + eps)` over the last dimension
# - usually includes a learnable scale (`weight`)
# - no mean subtraction
# 
# RMSNorm is popular in modern LLMs because it's faster.
# 

# In[56]:


class LayerNorm(nn.Module):
    def __init__(
        self, normalized_shape: int, eps: float = 1e-5, elementwise_affine: bool = True):
        super().__init__()
        self.eps = eps
        self.normalized_shape = normalized_shape
        self.weight = nn.Parameter(torch.ones([normalized_shape]))
        self.bias = nn.Parameter(torch.zeros([normalized_shape]))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Normalize over the last dimension.
        x: (..., D)
        """
        layer_mean = torch.mean(x, dim=-1, keepdim=True)
        layer_var = torch.var(x, dim=-1, correction=0, keepdim=True)
        result = (x - layer_mean) / (layer_var + self.eps)**0.5
        return result*self.weight + self.bias

LayerNorm(10).forward(torch.randn(10,10))


# In[52]:


class RMSNorm(nn.Module):
    def __init__(self, normalized_shape: int, eps: float = 1e-8):
        super().__init__()
        self.eps = eps
        self.normalized_shape = normalized_shape
        self.weight = nn.Parameter(torch.ones([normalized_shape]))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        RMSNorm: x / sqrt(mean(x^2) + eps) * weight
        over the last dimension.
        """
        return (x / (torch.mean(x**2, dim=-1, keepdim=True)**0.5 + self.eps)) * self.weight
RMSNorm(10).forward(torch.randn(1,10))


# ## MLPs and residual networks
# 
# Now you’ll build larger networks by composing layers.
# 
# ### MLP
# An MLP is a stack of `depth` Linear layers with non-linear activations (use GELU) in between.
# In this exercise you’ll support:
# - configurable depth
# - a hidden dimension
# - optional LayerNorm between layers (a common stabilization trick)
# 
# A key skill is building networks using `nn.ModuleList` / `nn.Sequential` while keeping shapes consistent.
# 
# ### Transformer-style FeedForward (FFN)
# A transformer block contains a position-wise feedforward network:
# - `D -> 4D -> D` (by default)
# - activation is typically **GELU**
# 
# This is essentially an MLP applied independently at each token position.
# 
# ### Residual wrapper
# Residual connections are the simplest form of “skip connection”:
# - output is `x + fn(x)`
# 
# They improve gradient flow and allow training deeper networks more reliably.

# In[61]:


class MLP(nn.Module):
    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        out_dim: int,
        depth: int,
        use_layernorm: bool = False,
    ):
        super().__init__()
        self.seq = nn.Sequential(Linear(in_dim, hidden_dim))
        self.seq.append(nn.GELU())
        for _ in range(depth):
            if use_layernorm:
                self.seq.append(LayerNorm(hidden_dim))
            self.seq.append(Linear(hidden_dim, hidden_dim))
            self.seq.append(nn.GELU())
        if use_layernorm:
            self.seq.append(LayerNorm(hidden_dim))
        self.seq.append(Linear(hidden_dim, out_dim))        

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.seq(x)

MLP(10,10,10,1)


# In[ ]:


class FeedForward(nn.Module):
    """
    Transformer-style FFN: D -> 4D -> D (default)
    """

    def __init__(self, d_model: int, d_ff: int | None = None):
        super().__init__()
        d_ff = d_ff or 4 * d_model
        # TODO: create two Linear layers and choose an activation (GELU)
        self.linear1 = Linear(d_model, d_ff)
        self.gelu = nn.GELU()
        self.linear2 = Linear(d_ff, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear2(self.gelu(self.linear1(x)))


# In[ ]:


class Residual(nn.Module):
    def __init__(self, fn: nn.Module):
        super().__init__()
        self.fn = fn

    def forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        return x + self.fn(x)


# ## Classification problem
# 
# In this section you’ll put everything together in a minimal MNIST classification experiment.
# 
# You will:
# 1) download and load the MNIST dataset
# 2) implement cross-entropy from logits (stable, using log-softmax)
# 3) build a simple MLP-based classifier (flatten MNIST images first)
# 4) write a minimal training loop
# 5) report train loss curve and final accuracy
# 
# The goal here is not to reach state-of-the-art accuracy, but to understand the full pipeline:
# data → model → logits → loss → gradients → parameter update.
# 
# ### Model notes
# - We want you to combine the MLP we implemented above with the classification head we define below into one model 
# 
# ### MNIST notes
# - MNIST images are `28×28` grayscale.
# - After `ToTensor()`, each image has shape `(1, 28, 28)` and values in `[0, 1]`.
# - For an MLP classifier, we flatten to a vector of length `784`.
# 
# ## Deliverables
# - Include a plot of your train loss curve in the video submission as well as a final accuracy. 
# - **NOTE** Here we don't grade on model performance but we expect you to achieve at least 70% accuracy to confirm a correct model implementation.

# In[ ]:


from torch.utils.data import DataLoader
from torchvision import datasets, transforms


# In[ ]:


transform = transforms.ToTensor()  # -> float32 in [0,1], shape (1, 28, 28)

train_ds = datasets.MNIST(root="data", train=True, download=True, transform=transform)
test_ds  = datasets.MNIST(root="data", train=False, download=True, transform=transform)

batch_size = 64
train_dl = DataLoader(train_ds, batch_size, shuffle=True)
test_dl = DataLoader(test_ds, batch_size, shuffle=True)


# In[83]:


def cross_entropy_from_logits(
    logits: torch.Tensor,
    targets: torch.Tensor,
) -> torch.Tensor:
    """
    Compute mean cross-entropy loss from logits.

    logits: (B, C)
    targets: (B,) int64

    Requirements:
    - Use log-softmax for stability (do not use torch.nn.CrossEntropyLoss, we check this in the autograder).
    """
    softmax = torch.exp(logits)/torch.sum(torch.exp(logits),dim=-1, keepdim=True)
    mask = torch.nn.functional.one_hot(targets, num_classes=logits.size(-1)) == 1
    return -torch.log(softmax).where(mask, 0).sum() / targets.size(0)

cross_entropy_from_logits(torch.ones(5,5), torch.ones([5],dtype=torch.int64))


# In[ ]:


class ClassificationHead(nn.Module):
    def __init__(self, d_in: int, num_classes: int):
        super().__init__()
        self.linear = Linear(d_in, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (..., d_in)
        return: (..., num_classes) logits
        """
        return self.linear(x)


# In[ ]:


def accuracy(loader):
    # TODO: You can use this function to evaluate your model accuracy.
    raise NotImplementedError


# In[ ]:

from matplotlib import pyplot as plt

device = "mps"

def train_classifier(
    model: nn.Module,
    train_data_loader: DataLoader,
    test_data_loader: DataLoader,
    lr: float,
    epochs: int,
    seed: int = 0,
) -> list[float]:
    """
    Minimal training loop for MNIST classification.

    Steps:
    - define optimizer
    - for each epoch:
        - sample minibatches
        - forward -> cross-entropy -> backward -> optimizer step
      - compute test accuracy at the end of each epoch
    - return list of training losses (one per update step)

    Requirements:
    - call model.train() during training and model.eval() during evaluation
    - do not use torch.nn.CrossEntropyLoss (use your cross_entropy_from_logits)
    """
    torch.manual_seed(seed)
    model.to(device)
    optim = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    
    training_losses = []
    test_losses = []

    for epoch in range(epochs):
        print(f"Staring epoch {epoch}")
        for i, batch in enumerate(train_data_loader):
            model.train()
            optim.zero_grad()
            logits = model(batch[0].flatten(start_dim=1).to(device))
            predicted_class = logits.argmax(dim=-1)
            loss = cross_entropy_from_logits(logits, batch[1].to(device))
            loss.backward()
            '''
            with torch.no_grad():
                accuracy = torch.where(predicted_class==batch[1], 1, 0).sum(dim=-1)/predicted_class.size(-1)
                print(f"Training accuracy at batch {i}: {accuracy.item()*100:.2f}%")
            '''
            optim.step()
            training_losses.append(loss.item())

            correct = total = 0
            max_samples = 50
            if i%50 == 0:
                with torch.no_grad():
                    for j, batch in enumerate(test_data_loader):
                        if j >= max_samples:
                            break
                        model.eval()
                        logits = model(batch[0].flatten(start_dim=1).to(device))
                        predicted_class = logits.argmax(dim=-1)
                        loss = cross_entropy_from_logits(logits, batch[1].to(device))
                        test_losses.append(loss.item())
                        correct += torch.where(predicted_class==batch[1].to(device), 1, 0).sum(dim=-1).item()
                        total += predicted_class.size(-1)
                    print(f"Test accuracy after batch {i}: {correct*100/total:.2f}%")
    plt.plot(training_losses)
    plt.plot(test_losses)
    plt.show()

model = MLP(in_dim=28*28, hidden_dim=28*14, out_dim=10, depth=1, use_layernorm=True)
train_classifier(model, train_dl, test_dl, 1e-4, 1)

