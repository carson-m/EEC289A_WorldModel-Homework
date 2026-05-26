"""Student one-step plus rollout loss with continuous cosine curriculum learning."""

from __future__ import annotations

import torch
import torch.nn.functional as F
import math

from .rollout import open_loop_rollout

# Stateful tracker to monitor total updates across epochs without resetting
_GLOBAL_UPDATE_STEP = 0

def one_step_delta_loss(model, states: torch.Tensor, actions: torch.Tensor, normalizer) -> torch.Tensor:
    """
    GRU-friendly one-step delta loss.
    Preserves temporal hidden-state evolution instead of flattening all steps.
    """
    B, T_plus_1, obs_dim = states.shape
    T = actions.shape[1]

    hidden = model.initial_hidden(B, states.device)
    losses = []

    for t in range(T):
        obs = states[:, t]
        act = actions[:, t]
        target_delta = states[:, t + 1] - states[:, t]

        obs_norm = normalizer.normalize_obs(obs)
        act_norm = normalizer.normalize_act(act)
        target_norm = normalizer.normalize_delta(target_delta)

        pred_norm, hidden = model(obs_norm, act_norm, hidden)
        losses.append(F.mse_loss(pred_norm, target_norm))

    return torch.stack(losses).mean()


def rollout_loss(model, states: torch.Tensor, actions: torch.Tensor, normalizer, warmup_steps: int, horizon: int) -> torch.Tensor:
    # 1. Ensure sequence length can accommodate the current dynamic horizon
    needed_states = int(warmup_steps) + int(horizon) + 1
    if states.shape[1] < needed_states:
        horizon = max(1, states.shape[1] - int(warmup_steps) - 1)

    # 2. FIX: Added normalizer argument, and removed the dual unpacking.
    pred_states = open_loop_rollout(
        model, states, actions, normalizer, warmup_steps, horizon
    )
    
    # 3. Slice ground-truth states to align perfectly with the rollout predictions.
    # Predictions start immediately after warmup steps and span across the horizon.
    start_idx = int(warmup_steps) + 1
    end_idx = start_idx + int(horizon)
    target_states = states[:, start_idx:end_idx]
    
    # 4. Normalize both tensors
    pred_norm = normalizer.normalize_obs(pred_states)
    target_norm = normalizer.normalize_obs(target_states)
    
    # 5. Calculate Mean Squared Error PER STEP (shape: [Horizon])
    per_step_loss = F.smooth_l1_loss(pred_norm, target_norm, reduction='none').mean(dim=(0, 2))
    
    H = per_step_loss.shape[0]
    weights = torch.linspace(1.0, 0.2, H, device=per_step_loss.device)
    
    weighted_loss = per_step_loss * weights
    
    return weighted_loss.mean()


def compute_loss(model, batch: dict[str, torch.Tensor], normalizer, cfg: dict):
    global _GLOBAL_UPDATE_STEP
    loss_cfg = cfg["loss"]
    train_cfg = cfg["training"]

    states = batch["states"]
    actions = batch["actions"]

    # Calculate 1-step delta baseline error
    one = one_step_delta_loss(model, states, actions, normalizer)
    warmup = int(cfg["eval"].get("warmup_steps", 10))

    # -----------------------------------------------------------------
    # FIX 3: SMOOTH PROGRESSIVE CURRICULUM FORMULATION
    # -----------------------------------------------------------------
    # Define a curriculum delay (e.g., hold flat for the first 2500 updates)
    curriculum_delay_steps = 2500 
    total_updates = 12000 # Your total training steps
    
    # Calculate how far along we are, ignoring the delay period
    if _GLOBAL_UPDATE_STEP < curriculum_delay_steps:
        cos_factor = 0.0
    elif _GLOBAL_UPDATE_STEP > total_updates:
        cos_factor = 1.0
    else:
        # Progress from 0.0 to 1.0 over the remaining steps
        active_steps = total_updates - curriculum_delay_steps
        current_active_step = _GLOBAL_UPDATE_STEP - curriculum_delay_steps
        progress = min(1.0, current_active_step / active_steps)
        
        # Standard cosine schedule (0 to 1)
        cos_factor = 0.5 * (1.0 - math.cos(math.pi * progress))

    # Curve A: Horizon scaling configuration
    base_horizon = 8
    target_horizon = int(loss_cfg.get("rollout_train_horizon", 15))
    current_horizon = int(base_horizon + cos_factor * (target_horizon - base_horizon))
    current_horizon = max(1, min(current_horizon, target_horizon))

    # Curve B: Dynamic Loss Weighting curves
    base_rollout_w = 0.2
    target_rollout_w = float(loss_cfg.get("rollout_weight", 0.4))
    current_rollout_weight = base_rollout_w + cos_factor * (target_rollout_w - base_rollout_w)

    # -----------------------------------------------------------------
    # Execution using scheduled targets
    # -----------------------------------------------------------------
    roll = rollout_loss(
        model,
        states,
        actions,
        normalizer,
        warmup_steps=warmup,
        horizon=current_horizon,
    )

    one_w = float(loss_cfg.get("one_step_weight", 1.8))
    long_w = cos_factor * float(loss_cfg.get("long_rollout_weight", 0.0))
    base_long_h = 1
    target_long_h = int(loss_cfg.get("long_rollout_horizon", current_horizon))
    
    # FIX: Force long_h to be an integer and protect it with a max floor of 1
    long_h = int(base_long_h + cos_factor * (target_long_h - base_long_h))
    long_h = max(1, long_h)

    if long_w > 0.0:
        long_roll = rollout_loss(
            model,
            states,
            actions,
            normalizer,
            warmup_steps=warmup,
            horizon=long_h,
        )
    else:
        long_roll = torch.zeros((), device=states.device)

    # Combine regular terms + scale with dynamic weights
    total = (one_w * one) + (current_rollout_weight * roll) + (long_w * long_roll)

    # Increment update step safety counter
    _GLOBAL_UPDATE_STEP += 1

    return total, {
        "loss/total": float(total.detach().cpu()),
        "loss/one_step": float(one.detach().cpu()),
        "loss/rollout": float(roll.detach().cpu()),
        "loss/long_rollout": float(long_roll.detach().cpu()),
        "curriculum/horizon": float(current_horizon),
        "curriculum/rollout_weight": float(current_rollout_weight)
    }