#!/usr/bin/env python
# coding: utf-8

# # Exercise 4: Transformers on Images + GLU-MLP Ablations (ViT × GLU Variants)
# 
# ## In this exercise you will combine two influential ideas:
# 
# Vision Transformers (ViT) from “An Image is Worth 16×16 Words: Transformers for Image Recognition at Scale” (Dosovitskiy et al., 2020) https://arxiv.org/pdf/2010.11929:
# ViT shows that you can treat an image like a sequence of tokens by splitting it into non-overlapping patches (e.g. 16×16 in the paper), embedding each patch into a vector, adding positional information, and then applying standard Transformer blocks for classification.
# 
# Gated MLPs (GLU variants) from “GLU Variants Improve Transformer” (Shazeer, 2020) https://arxiv.org/pdf/2002.05202:
# Shazeer proposes replacing the standard Transformer feed-forward layer (FFN/MLP) with gated linear unit (GLU) variants such as GEGLU and SwiGLU, which often improves training dynamics and final performance under comparable compute/parameter budgets.
# 
# ## What you will do
# 
# You will implement a tiny ViT-style classifier for MNIST, then run a controlled ablation where you replace the MLP inside each Transformer block:
# 
# Baseline FFN (GELU):
# Linear(d_model → d_ff) → GELU → Linear(d_ff → d_model)
# 
# GLU-family MLPs (choose at least two and justify):
# 
# GEGLU, SwiGLU, other activation functions
# 
# Your goal is to evaluate whether these GLU variants change:
# 
# - convergence speed (loss vs steps),
# 
# - final test accuracy,
# 
# - and/or stability across runs.
# 
# ## Key ViT concepts you will implement
# 
# - To convert MNIST images into Transformer tokens, you will:
#   Patchify each 28×28 image into non-overlapping P×P patches.
#   If P=4, then you get a 7×7 patch grid → 49 tokens per image.
# 
# - Embed patches with a linear layer: patch vectors → d_model.
# 
# - Add positional embeddings so the model knows where each patch came from.
# 
# - Apply n_layers Transformer encoder blocks.
# 
# - Pool token features (e.g., mean pooling) and project to 10 classes.
# 
# ## Key GLU concept you will implement
# 
# GLU-style MLPs replace a standard FFN with a gating mechanism:
# compute two projections a and b, apply a nonlinearity to a (variant-dependent), multiply elementwise: act(a) * b, project back to d_model.
# To keep the comparison fair, use the 2/3 width rule from Shazeer.
# 
# What we provide vs what you implement
# 
# ### We provide:
# 
# - MNIST loading + dataloaders
# 
# - a minimal training loop structure (AdamW)
# 
# - a suggested small model configuration that runs on CPU
# 
# ### You implement:
# 
# - patch tokenization (patchify)
# 
# - patch embedding + positional embedding strategy
# 
# - a pre-LN Transformer encoder block using nn.MultiheadAttention
# 
# - at least two GLU MLP variants + one FFN baseline
# 
# - metric logging sufficient to support your conclusion
# 
# ## Deliverables
# 
# Run at least 3 variants (baseline + the activation functions you choose for GLU) and report:
# 
# - final and best test accuracy
# 
# - number of trainable parameters
# 
# - a plot or printed summary of loss/accuracy over epochs
# 
# - a short discussion of your results

# In[3]:


from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


# In[ ]:


def patchify(x: torch.Tensor, patch_size: int) -> torch.Tensor:
    """Convert images to patch tokens."""
    # input: [B][1][H][W]
    # output: [B][HW/P^2][P^2]

    x = x.squeeze(dim=1)#get rid of channel dim
    patches = x.reshape([x.size(0), int(x.size(-2)/patch_size), patch_size, int(x.size(-1)/patch_size), patch_size])
    patches = patches.permute(0, 1, 3, 2, 4)
    return patches.reshape([patches.size(0), int((x.size(-2)*x.size(-1))/patch_size**2), patch_size**2]) 

# In[ ]:


# TODO: Add positional encoding as done in the ViT paper and patch projection
class PatchEmbed(nn.Module):
    def __init__(self, patch_dim: int, d_model: int):
        super().__init__()
        self.linear = nn.Linear(patch_dim, d_model)

    def forward(self, x_patches: torch.Tensor) -> torch.Tensor:
        return self.linear(x_patches)


class PositionalEmbedding(nn.Module):
    def __init__(self, num_tokens: int, d_model: int):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(num_tokens, d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        '''
        Only supports fixed length sequences
        '''
        #input: [B][T][D]
        #output: [B][T][D]
        return x+self.weight

# In[ ]:


# TODO: Define the variants you want to compare against each other from the GLU paper. Justify your choice.
class FeedForward(nn.Module):
    """
    Standard Transformer FFN:
      x -> Linear(d_model->d_ff) -> GELU -> Dropout -> Linear(d_ff->d_model) -> Dropout
    """
    def __init__(self, d_model: int, d_ff: int, dropout: float):
        super().__init__()
        self.seq = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.seq(x)

class GLUFeedForward(nn.Module):
    """GLU-family FFN"""
    def __init__(self, d_model: int, d_ff_gated: int, dropout: float, variant: str):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff_gated, bias=False)
        self.linear2 = nn.Linear(d_model, d_ff_gated, bias=False)#omit bias like Shazeer(2020)
        self.projection = nn.Linear(d_ff_gated, d_model, bias=False)
        self.dropout =  nn.Dropout(dropout)

        match variant:
            case "relu": 
                # acheived the best performance on a few of the metrics in Shazeer(2020) 
                self.activation = nn.ReLU()
            case "gelu": 
                # maintains the same activation function as the feedforward baseline, 
                # isolating for any performance the GLU approach provides
                self.activation = nn.GELU()
            case _:
                raise Exception(f"please provide a valid activation type, got: {variant}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.projection(self.dropout(self.activation(self.linear1(x))*self.linear2(x))))
        # this is so ugly but idc

# In[ ]:


class TransformerEncoderBlock(nn.Module):
    """
    Pre-LN encoder block:
      x = x + Dropout(SelfAttn(LN(x)))
      x = x + Dropout(MLP(LN(x)))
    """
    def __init__(self, d_model: int, n_heads: int, mlp: nn.Module, dropout: float):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.attention_heads = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.ln2 = nn.LayerNorm(d_model)
        self.mlp = mlp
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        normed_x = self.ln1(x)
        x = x + self.dropout(self.attention_heads(self.q_proj(normed_x),
                                                  self.k_proj(normed_x),
                                                  self.v_proj(normed_x))[0])
        x = x + self.dropout(self.mlp(self.ln2(x)))
        return x

# In[ ]:


class TinyViT(nn.Module):
    """
    Tiny ViT-style classifier for MNIST.
    - patchify -> patch embed -> pos embed -> blocks -> mean pool -> head
    """
    def __init__(
        self,
        patch_size: int,
        d_model: int,
        n_heads: int,
        n_layers: int,
        d_ff: int,
        dropout: float,
        mlp_kind: str,
    ):
        super().__init__()
        assert 28 % patch_size == 0
        grid = 28 // patch_size
        self.num_tokens = grid * grid
        self.patch_size = patch_size
        patch_dim = patch_size * patch_size

        self.embedding = PatchEmbed(patch_dim, d_model)
        self.pos_embedding = PositionalEmbedding(self.num_tokens,d_model)
    
        def create_mlp(mlp_kind: str):
            match mlp_kind:
                case "ff":
                    return FeedForward(d_model, d_ff, dropout)
                case "geglu":
                    return GLUFeedForward(d_model, int(d_ff*2/3), dropout, 'gelu')
                case "reglu":
                    return GLUFeedForward(d_model, int(d_ff*2/3), dropout, 'relu')
                case _:
                    raise Exception(f"please provide a valid mlp type, got: {mlp_kind}")

        self.blocks = nn.ModuleList([
            TransformerEncoderBlock(
                d_model=d_model,
                n_heads=n_heads,
                mlp=create_mlp(mlp_kind), 
                dropout=dropout,
            )
            for _ in range(n_layers)
        ])
    
        self.projection = nn.Linear(d_model, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        tokens = patchify(x, self.patch_size)
        embeddings = self.embedding(tokens)
        enriched_embeddings = self.pos_embedding(embeddings)
        for block in self.blocks:
            enriched_embeddings = block(enriched_embeddings)
        pooled_embeddings = torch.mean(enriched_embeddings, dim=1) #[B][T][D] -> [B][D]
        logits = self.projection(pooled_embeddings)
        return logits


# In[ ]:


@dataclass(frozen=True)
class TrainConfig:
    seed: int = 0
    batch_size: int = 128
    epochs: int = 3
    lr: float = 3e-4
    weight_decay: float = 0.01
    device: str = "cpu"  # set "cuda" if available


# In[ ]:


def train_one_run(
    mlp_kind: str,
    model: nn.Module,
    train_loader: DataLoader,
    test_loader: DataLoader,
    cfg: TrainConfig,
) -> dict:
    model.to(cfg.device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    train_losses: list[float] = []
    test_accs: list[float] = []

    for epoch in range(cfg.epochs):

        # Train loop
        model.train()
        for i, (xb, yb) in enumerate(train_loader):
            xb = xb.to(cfg.device)
            yb = yb.to(cfg.device)

            logits = model(xb)
            loss = F.cross_entropy(logits, yb)
            #print(f"batch {i} loss: {loss}")

            opt.zero_grad()
            loss.backward()
            opt.step()

            train_losses.append(loss.item())

        # Evaluation loop NOTE: Should be no need to change this
        model.eval()
        correct = 0.0
        total = 0.0
        with torch.no_grad():
            for xb, yb in test_loader:
                xb = xb.to(cfg.device)
                yb = yb.to(cfg.device)
                logits = model(xb)
                correct += (logits.argmax(dim=-1) == yb).float().sum().item()
                total += yb.numel()

        test_accs.append(correct / total)
        print(f"[{mlp_kind}] epoch {epoch+1}/{cfg.epochs} | test acc: {test_accs[-1]:.4f}")

    return {
        # TODO: Return your metrics that you think will support your claim for this experiment
    }


# In[ ]:


cfg = TrainConfig(seed=0, batch_size=128, epochs=1, lr=3e-4, weight_decay=0.01, device="cpu")

tfm = transforms.Compose([transforms.ToTensor()])

train_ds = datasets.MNIST(root="./data", train=True, download=True, transform=tfm)
test_ds = datasets.MNIST(root="./data", train=False, download=True, transform=tfm)

train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, num_workers=0)
test_loader = DataLoader(test_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=0)

# Tiny model example. TODO: You're welcome to experiment with these parameters
patch_size = 4
d_model = 64
n_heads = 4
n_layers = 2
d_ff = 256
dropout = 0.1

runs = ['ff', 'reglu', 'geglu']
results = []

train_it = train_loader._get_iterator()
batch = next(train_it)

for kind in runs:
    model = TinyViT(
        patch_size=patch_size,
        d_model=d_model,
        n_heads=n_heads,
        n_layers=n_layers,
        d_ff=d_ff,
        dropout=dropout,
        mlp_kind=kind,
    )
    # TODO: print anything you might want here
    print(f"\nRun: {kind} | param count: {sum(p.numel() for p in model.parameters() if p.requires_grad)}" )
    out = train_one_run(kind, model, train_loader, test_loader, cfg)
    results.append(out)




