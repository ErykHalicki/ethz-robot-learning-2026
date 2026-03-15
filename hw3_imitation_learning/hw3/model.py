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
        d_model = 350, 
        depth = 4,
        *args, 
        **kwargs
    ) -> None:
        super().__init__(*args, **kwargs)
        # model size parameters
        self.chunk_size = 10
        self.gripper_action_dim = 10
        self.ee_action_dim = 7 #[0, +x, +y, +z, -x, -y, -z]
        self.depth = depth 
        self.d_model = d_model

        self.activation = nn.GELU()
        self.input_layer = nn.Linear(self.state_dim,d_model)
        self.input_norm = nn.LayerNorm([self.state_dim])
        self.layer_norms = nn.ModuleList([nn.LayerNorm([d_model]) for _ in range(self.depth)])
        self.hidden_layers = nn.ModuleList([nn.Linear(d_model, d_model) for _ in range(self.depth)])
        self.ee_output_layer = nn.Linear(d_model, self.ee_action_dim*self.chunk_size)
        self.gripper_output_layer = nn.Linear(d_model, self.gripper_action_dim*self.chunk_size)
        self.dropout = torch.nn.Dropout(p=0.125)

        zero_movement_weight = 0.075
        self.log_var_ee = nn.Parameter(torch.zeros(1))
        self.log_var_gripper = nn.Parameter(torch.zeros(1))
        ee_ce_weights = torch.zeros([7])
        ee_ce_weights[:] = (1.-zero_movement_weight)/6.
        ee_ce_weights[0] = zero_movement_weight
        self.register_buffer('ee_ce_weights', ee_ce_weights)
        self.gripper_loss_function = torch.nn.CrossEntropyLoss()
        self.softmax = torch.nn.Softmax(dim=-1)

        gripper_bounds = torch.linspace(-0.2, 1.8, self.gripper_action_dim - 1)
        ee_action_map = torch.tensor([[0.,0.,0.],  # 0 movement
                                          [1.,0.,0.],   # +x
                                          [0.,1.,0.],   # +y
                                          [0.,0.,1.],   # +z
                                          [-1.,0.,0.],  # -x
                                          [0.,-1.,0.],  # -y
                                          [0.,0.,-1.]]) # -z
        self.ee_translation_per_step = 0.0075
        
        bin_midpoints = (gripper_bounds[:-1] + gripper_bounds[1:]) / 2
        gripper_centers = torch.cat([gripper_bounds[:1], bin_midpoints, gripper_bounds[-1:]])
        self.register_buffer('gripper_centers', gripper_centers)
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
        x = self.input_norm(x)
        x = self.dropout(self.activation(self.input_layer(x)))
        for i in range(self.depth):
            x = self.dropout(self.activation(self.hidden_layers[i](self.layer_norms[i](x))))
        gripper_out = self.gripper_output_layer(x)
        ee_out = self.ee_output_layer(x)
        return {"ee": torch.reshape(ee_out, [x.size(0), self.chunk_size, self.ee_action_dim]),
                "gripper": torch.reshape(gripper_out, [x.size(0), self.chunk_size, self.gripper_action_dim])}

    def compute_loss(
        self, state: torch.Tensor, action_chunk: torch.Tensor
    ) -> torch.Tensor:
        predicted_action_chunks = self.forward(state)
        target_action_chunks = self.discretize_action(action_chunk)
        
        ee_loss = torch.nn.functional.cross_entropy(
            predicted_action_chunks["ee"].flatten(end_dim=-2),
            target_action_chunks["ee"].flatten(),
            weight=self.ee_ce_weights,
        )
        gripper_loss = self.gripper_loss_function(predicted_action_chunks["gripper"].flatten(end_dim=-2), 
                                          target_action_chunks["gripper"].flatten())
        return ee_loss * 0.25 + gripper_loss * 0.75
        #return (ee_loss * torch.exp(-self.log_var_ee) + self.log_var_ee +
        #gripper_loss * torch.exp(-self.log_var_gripper) + self.log_var_gripper)
               #learned gripper - ee loss ratio

    def sample_actions(
        self,
        state: torch.Tensor,
    ) -> torch.Tensor:
        self.eval()
        ee_temp = 1.0
        gripper_temp = 1.0
        with torch.no_grad():
            action_logits = self.forward(state)
            #action_logits["ee"][:, :, 0] /= 5
            #print(action_logits["ee"])
            ee_probabilities = self.softmax(action_logits["ee"]/ee_temp).flatten(end_dim=-2)
            gripper_probabilities = self.softmax(action_logits["gripper"]/gripper_temp).flatten(end_dim=-2)
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
        ee_movement_thresh = 0.004

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
    

class MSEPolicy(BasePolicy):
    """Predicts continuous action chunks with an MSE loss.

    Same MLP backbone as ObstaclePolicy but outputs raw (B, chunk_size, action_dim)
    without any action discretization.
    """

    def __init__(
        self,
        d_model: int = 200,
        depth: int = 8,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.depth = depth
        self.d_model = d_model

        self.activation = nn.GELU()
        self.input_norm = nn.LayerNorm([self.state_dim])
        self.input_layer = nn.Linear(self.state_dim, d_model)
        self.layer_norms = nn.ModuleList([nn.LayerNorm([d_model]) for _ in range(depth)])
        self.hidden_layers = nn.ModuleList([nn.Linear(d_model, d_model) for _ in range(depth)])
        self.output_layer = nn.Linear(d_model, self.action_dim * self.chunk_size)
        self.dropout = nn.Dropout(p=0.1)
        self.loss_fn = nn.MSELoss()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_norm(x)
        x = self.dropout(self.activation(self.input_layer(x)))
        for norm, layer in zip(self.layer_norms, self.hidden_layers):
            x = self.dropout(self.activation(layer(norm(x))))
        return self.output_layer(x).reshape(x.size(0), self.chunk_size, self.action_dim)

    def compute_loss(self, state: torch.Tensor, action_chunk: torch.Tensor) -> torch.Tensor:
        return self.loss_fn(self.forward(state), action_chunk)

    def sample_actions(self, state: torch.Tensor) -> torch.Tensor:
        self.eval()
        with torch.no_grad():
            return self.forward(state)


# TODO: Students implement MultiTaskPolicy here.
class MultiTaskPolicy(ObstaclePolicy):
    """Goal-conditioned policy for the multicube scene."""
    def __init__(
        self,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.dropout = nn.Dropout(p=0.15)
        


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
            chunk_size=chunk_size,
            d_model=d_model,
            depth=depth
        )
    raise ValueError(f"Unknown policy type: {policy_type}")
