"""Model definitions for SO-100 imitation policies."""

from __future__ import annotations

import abc
from typing import Literal, TypeAlias

import torch
from torch import nn


class BasePolicy(nn.Module, metaclass=abc.ABCMeta):
    """Base class for action chunking policies."""

    def __init__(self, state_dim: int, action_dim: int, chunk_size: int) -> None:
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.chunk_size = chunk_size

    @abc.abstractmethod
    def compute_loss(
        self, state: torch.Tensor, action_chunk: torch.Tensor
    ) -> torch.Tensor:
        """Compute training loss for a batch."""

    @abc.abstractmethod
    def sample_actions(
        self,
        state: torch.Tensor,
    ) -> torch.Tensor:
        """Generate a chunk of actions with shape (batch, chunk_size, action_dim)."""


# TODO: Students implement ObstaclePolicy here.
class ObstaclePolicy(BasePolicy):
    """Predicts action chunks with an MSE loss.

    A simple MLP that maps a state vector to a flat action chunk
    (chunk_size * action_dim) and reshapes to (B, chunk_size, action_dim).
    """

    def __init__(
        self, 
        h_dim = 200, 
        num_layers = 24,# ~1M params

        *args, 
        **kwargs
    ) -> None:
        super().__init__(*args, **kwargs)
        self.gripper_action_dim = 21
        self.gripper_bounds=torch.linspace(-0.2, 1.8, self.gripper_action_dim)
        self.action_dim = 7 #[0, +x, +y, +z, -x, -y, -z]
        self.num_layers = num_layers 
        self.activation = nn.GELU()
        self.input_layer = nn.Linear(self.state_dim,h_dim)
        self.layer_norms = nn.ModuleList([nn.LayerNorm([h_dim]) for _ in range(self.num_layers)])
        self.hidden_layers = nn.ModuleList([nn.Linear(h_dim, h_dim) for _ in range(self.num_layers)])
        self.ee_output_layer = nn.Linear(h_dim, self.action_dim*self.chunk_size)
        self.gripper_output_layer = nn.Linear(h_dim, self.gripper_action_dim*self.chunk_size)
        # multiplying by 2 to get a positive and negative direction for each axis

    def forward(
        self, x
    ) -> torch.Tensor:
        """
        x: must have batch dim
        Return predicted action chunk of shape (B, chunk_size, action_dim).
        """
        x = self.activation(self.input_layer(x))
        for i in range(self.num_layers):
            x = self.activation(self.hidden_layers[i](self.layer_norms[i](x))) + x
        x = self.output_layer(x)
        return torch.reshape(x, [x.size(0), self.chunk_size, self.action_dim*2])
        #TODO CHANGE TO OUTPUT MULTIPLE ACTION SPACES (1 for gripper and one for ee) 
        # outputs logits

    def compute_loss(
        self,
    ) -> torch.Tensor:
        raise NotImplementedError

    def sample_actions(
        self,
        state: torch.Tensor,
    ) -> torch.Tensor:
        # TODO need to softmax and sample logits of output actions
        # then, convert to original action space (xyz) based
        # on denormalization params (move 0.01 per time step)
        # and concat the gripper to create R4 action vector
        return self.forward(state)

    def discretize_action(self, action):
        '''
        Action should be [B, action_chunk, action_dim]
        Returns seperate discretized actions for ee and gripper [B, action_chunk, 2]
        ([+x,+y, +z, -x, -y, -z], [-0.1, 0.0, 0.1, 0.2 ..., 1.1])
        '''
        ee_movement_thresh = 0.005
        decimals = 3

        positive_mask = torch.zeros_like(action[:, :, :3], dtype=bool)
        negative_mask = torch.zeros_like(action[:, :, :3], dtype=bool)
        positive_mask = action[:, :, :3].round(decimals=decimals)>=ee_movement_thresh
        negative_mask = action[:, :, :3].round(decimals=decimals)<=-ee_movement_thresh

        movement_mask = torch.cat((positive_mask, negative_mask), dim=2).int()
        no_movement_mask = (movement_mask == 0).all(dim=-1, keepdim=True).int()
        mask = torch.cat([no_movement_mask, movement_mask], dim=-1)
        ee_idx = (mask.argmax(dim=-1))
        # if multiple movements happen at once, prioritizes lowest idx (kind of arbitrary but should work)

        gripper_idx = torch.bucketize(action[:,:, 3], boundaries=self.gripper_bounds)

        return {"ee": ee_idx, "gripper": gripper_idx}
    
    #def one_hotify_action(self, action_idx):
    #   return {"ee": nn.functional.one_hot(action_idx[:, :, 0], self.action_dim), 
    #          "gripper": nn.functional.one_hot(action_idx[:, :, 1], self.gripper_dim)}

# TODO: Students implement MultiTaskPolicy here.
class MultiTaskPolicy(BasePolicy):
    """Goal-conditioned policy for the multicube scene."""

    def __init__(
        self,
    ) -> None:
        super().__init__()

    def compute_loss(
        self,
    ) -> torch.Tensor:
        raise NotImplementedError

    def sample_actions(
        self,
    ) -> torch.Tensor:
        raise NotImplementedError

    def forward(
        self,
    ) -> torch.Tensor:
        """Return predicted action chunk of shape (B, chunk_size, action_dim)."""
        raise NotImplementedError


PolicyType: TypeAlias = Literal["obstacle", "multitask"]


def build_policy(
    policy_type: PolicyType,
    state_dim: int,
    action_dim: int,
    chunk_size: int
) -> BasePolicy:
    if policy_type == "obstacle":
        return ObstaclePolicy(
            action_dim=action_dim,
            state_dim=state_dim,
            chunk_size=chunk_size
        )
    if policy_type == "multitask":
        return MultiTaskPolicy(
            action_dim=action_dim,
            state_dim=state_dim,
            # TODO: Build with your chosen specifications
        )
    raise ValueError(f"Unknown policy type: {policy_type}")
