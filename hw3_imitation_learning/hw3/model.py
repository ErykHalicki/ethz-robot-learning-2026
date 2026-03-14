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
        d_model = 200, 
        depth = 24,# ~1M params
        *args, 
        **kwargs
    ) -> None:
        super().__init__(*args, **kwargs)
        # model size parameters
        self.gripper_action_dim = 30
        self.ee_action_dim = 7 #[0, +x, +y, +z, -x, -y, -z]
        self.depth = depth 
        self.d_model = d_model

        self.activation = nn.GELU()
        self.input_layer = nn.Linear(self.state_dim,d_model)
        self.layer_norms = nn.ModuleList([nn.LayerNorm([d_model]) for _ in range(self.depth)])
        self.hidden_layers = nn.ModuleList([nn.Linear(d_model, d_model) for _ in range(self.depth)])
        self.ee_output_layer = nn.Linear(d_model, self.ee_action_dim*self.chunk_size)
        self.gripper_output_layer = nn.Linear(d_model, self.gripper_action_dim*self.chunk_size)

        self.ee_loss_weight = 0.3
        zero_movement_weight = 0.03
        ee_ce_weights = torch.zeros([7])
        ee_ce_weights[:] = (1.-zero_movement_weight)/6.
        ee_ce_weights[0] = zero_movement_weight
        self.ee_loss_function = torch.nn.CrossEntropyLoss(weight=ee_ce_weights)
        self.gripper_loss_function = torch.nn.CrossEntropyLoss()
        self.softmax = torch.nn.Softmax(dim=-1)

        gripper_bounds = torch.linspace(-0.2, 1.75, self.gripper_action_dim)
        ee_action_map = torch.tensor([[0.,0.,0.],  # 0 movement
                                          [1.,0.,0.],   # +x
                                          [0.,1.,0.],   # +y
                                          [0.,0.,1.],   # +z
                                          [-1.,0.,0.],  # -x
                                          [0.,-1.,0.],  # -y
                                          [0.,0.,-1.]]) # -z
        self.ee_translation_per_step = 0.01
        
        self.register_buffer('gripper_centers', (gripper_bounds[:-1] + gripper_bounds[1:]) / 2)
        self.register_buffer('gripper_bounds', gripper_bounds)
        self.register_buffer('ee_action_map', ee_action_map)

    def forward(
        self, x
    ) -> torch.Tensor:
        """
        x: must have batch dim
        returns a pair of action logits (ee, gripper)
        ee: [B, chunk_dim, ee_action_dim]
        gripper: [B, chunk_dim, gripper_action_dim]
        """
        x = self.activation(self.input_layer(x))
        for i in range(self.depth):
            x = self.activation(self.hidden_layers[i](self.layer_norms[i](x))) + x
        gripper_out = self.gripper_output_layer(x)
        ee_out = self.ee_output_layer(x)
        return {"ee": torch.reshape(ee_out, [x.size(0), self.chunk_size, self.ee_action_dim]),
                "gripper": torch.reshape(gripper_out, [x.size(0), self.chunk_size, self.gripper_action_dim])}

    def compute_loss(
        self, state: torch.Tensor, action_chunk: torch.Tensor
    ) -> torch.Tensor:
        predicted_action_chunks = self.forward(state)
        target_action_chunks = self.discretize_action(action_chunk)
        
        ee_loss = self.ee_loss_function(predicted_action_chunks["ee"].flatten(end_dim=-2), 
                                     target_action_chunks["ee"].flatten())
        gripper_loss = self.gripper_loss_function(predicted_action_chunks["gripper"].flatten(end_dim=-2), 
                                          target_action_chunks["gripper"].flatten())
        return (ee_loss * self.ee_loss_weight) + (gripper_loss* (1.0-self.ee_loss_weight))

    def sample_actions(
        self,
        state: torch.Tensor,
    ) -> torch.Tensor:
        with torch.no_grad():
            action_logits = self.forward(state)
            #action_logits["ee"][:, :, 0] /= 2
            #print(action_logits["ee"])
            ee_probabilities = self.softmax(action_logits["ee"]).flatten(end_dim=-2)
            gripper_probabilities = self.softmax(action_logits["gripper"]).flatten(end_dim=-2)
            gripper_idx = torch.multinomial(gripper_probabilities, num_samples=1).reshape([state.size(0), self.chunk_size, 1])
            ee_idx = torch.multinomial(ee_probabilities, num_samples=1).reshape([state.size(0), self.chunk_size])
            ee_actions = self.ee_action_map[ee_idx]*self.ee_translation_per_step
            gripper_actions = self.gripper_centers[gripper_idx.clamp(0, len(self.gripper_centers) - 1)]
            action_chunks = torch.cat([ee_actions, gripper_actions], dim=-1)
        return action_chunks

    def discretize_action(self, action):
        '''
        Action should be [B, action_chunk, action_dim]
        Returns seperate discretized actions for ee and gripper {[B, action_chunk], [B,action_chunk]}
        ([0, +x,+y, +z, -x, -y, -z], [-0.2, 0.0, 0.1, 0.2 ..., 1.8])
        '''
        ee_movement_thresh = 0.005

        positive_mask = torch.zeros_like(action[:, :, :3], dtype=bool)
        negative_mask = torch.zeros_like(action[:, :, :3], dtype=bool)
        positive_mask = action[:, :, :3]>=ee_movement_thresh
        negative_mask = action[:, :, :3]<=-ee_movement_thresh

        movement_mask = torch.cat((positive_mask, negative_mask), dim=2).int()
        no_movement_mask = (movement_mask == 0).all(dim=-1, keepdim=True).int()
        mask = torch.cat([no_movement_mask, movement_mask], dim=-1)
        ee_idx = (mask.argmax(dim=-1))
        # if multiple movements happen at once, prioritizes lowest idx (kind of arbitrary but should work)
        gripper_action = action[:,:, 3].clone()
        gripper_idx = torch.bucketize(gripper_action, boundaries=self.gripper_bounds)

        return {"ee": ee_idx, "gripper": gripper_idx}
    

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
    chunk_size: int,
    d_model,
    depth,
) -> BasePolicy:
    if policy_type == "obstacle":
        return ObstaclePolicy(
            action_dim=action_dim,
            state_dim=state_dim,
            chunk_size=chunk_size,
            d_model=d_model,
            depth=depth
        )
    if policy_type == "multitask":
        return MultiTaskPolicy(
            action_dim=action_dim,
            state_dim=state_dim,
            # TODO: Build with your chosen specifications
        )
    raise ValueError(f"Unknown policy type: {policy_type}")
