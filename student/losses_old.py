"""Student one-step plus rollout loss."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from .rollout import open_loop_rollout


def _time_weights(length: int, device: torch.device, dtype: torch.dtype, start_weight: float, end_weight: float) -> torch.Tensor:
    return torch.linspace(float(start_weight), float(end_weight), int(length), device=device, dtype=dtype)


def _weighted_smooth_l1(
    pred: torch.Tensor,
    target: torch.Tensor,
    start_weight: float = 1.0,
    end_weight: float = 1.0,
    beta: float = 1.0,
) -> torch.Tensor:
    loss = F.smooth_l1_loss(pred, target, beta=float(beta), reduction="none").mean(dim=-1)
    if loss.ndim < 2:
        return loss.mean()
    weights = _time_weights(loss.shape[1], loss.device, loss.dtype, start_weight, end_weight)
    return (loss * weights.view(1, -1)).sum() / (loss.shape[0] * weights.sum())


def _normalized_step_mse(pred: torch.Tensor, target: torch.Tensor, normalizer) -> torch.Tensor:
    obs_std = torch.as_tensor(normalizer.obs_std, dtype=pred.dtype, device=pred.device).clamp_min(1e-6)
    return torch.mean(((pred - target) / obs_std) ** 2, dim=-1)


def _tail_vpt_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    normalizer,
    threshold: float,
    soft_threshold: float,
    tail_fraction: float,
    start_weight: float,
    end_weight: float,
) -> torch.Tensor:
    step_nmse = _normalized_step_mse(pred, target, normalizer)
    weights = _time_weights(step_nmse.shape[1], step_nmse.device, step_nmse.dtype, start_weight, end_weight)
    soft_margin = F.relu(step_nmse - float(soft_threshold)).pow(2)
    hard_margin = F.relu(step_nmse - float(threshold)).pow(2)
    per_window = ((soft_margin + hard_margin) * weights.view(1, -1)).sum(dim=1) / weights.sum()
    k = max(1, min(per_window.shape[0], int(math.ceil(per_window.shape[0] * float(tail_fraction)))))
    return torch.topk(per_window, k=k, largest=True).values.mean()


def one_step_delta_loss(model, states: torch.Tensor, actions: torch.Tensor, normalizer) -> torch.Tensor:
    if bool(getattr(model, "use_gru", False)):
        hidden = model.initial_hidden(states.shape[0], states.device)
        losses = []
        for t in range(actions.shape[1]):
            obs = states[:, t]
            act = actions[:, t]
            target_delta = states[:, t + 1] - states[:, t]
            obs_norm = normalizer.normalize_obs(obs)
            act_norm = normalizer.normalize_act(act)
            target_norm = normalizer.normalize_delta(target_delta)
            pred_norm, hidden = model(obs_norm, act_norm, hidden)
            losses.append(F.mse_loss(pred_norm, target_norm, reduction="none").mean(dim=-1))
        return torch.stack(losses, dim=1).mean()

    obs = states[:, :-1].reshape(-1, states.shape[-1])
    act = actions.reshape(-1, actions.shape[-1])
    target_delta = (states[:, 1:] - states[:, :-1]).reshape(-1, states.shape[-1])
    obs_norm = normalizer.normalize_obs(obs)
    act_norm = normalizer.normalize_act(act)
    target_norm = normalizer.normalize_delta(target_delta)
    pred_norm, _ = model(obs_norm, act_norm, None)
    return F.mse_loss(pred_norm, target_norm)


def _single_rollout_loss(
    model,
    states: torch.Tensor,
    actions: torch.Tensor,
    normalizer,
    warmup_steps: int,
    horizon: int,
    start_weight: float,
    end_weight: float,
    vpt_tail_weight: float,
    vpt_threshold: float,
    vpt_soft_threshold: float,
    vpt_tail_fraction: float,
    vpt_start_weight: float,
    vpt_end_weight: float,
) -> torch.Tensor:
    # Train local open-loop stability at random positions, not only at the
    # beginning of each stored window.
    needed_states = int(warmup_steps) + int(horizon) + 1
    if states.shape[1] < needed_states:
        raise ValueError(
            "training.train_sequence_length is too short for rollout loss: "
            f"need at least {needed_states - 1} actions for warmup={warmup_steps}, horizon={horizon}."
        )
    max_start = states.shape[1] - needed_states
    if max_start > 0:
        start = int(torch.randint(0, max_start + 1, (), device=states.device).item())
    else:
        start = 0
    sub_states = states[:, start : start + needed_states]
    sub_actions = actions[:, start : start + int(warmup_steps) + int(horizon)]
    preds = open_loop_rollout(model, sub_states, sub_actions, normalizer, warmup_steps=warmup_steps, horizon=horizon)
    targets = sub_states[:, warmup_steps + 1 : warmup_steps + 1 + horizon]
    pred_norm = normalizer.normalize_obs(preds)
    target_norm = normalizer.normalize_obs(targets)
    base = _weighted_smooth_l1(pred_norm, target_norm, start_weight=start_weight, end_weight=end_weight, beta=0.5)
    if float(vpt_tail_weight) <= 0.0:
        return base
    tail = _tail_vpt_loss(
        preds,
        targets,
        normalizer,
        threshold=vpt_threshold,
        soft_threshold=vpt_soft_threshold,
        tail_fraction=vpt_tail_fraction,
        start_weight=vpt_start_weight,
        end_weight=vpt_end_weight,
    )
    return base + float(vpt_tail_weight) * tail


def rollout_loss(
    model,
    states: torch.Tensor,
    actions: torch.Tensor,
    normalizer,
    warmup_steps: int,
    horizon: int,
    samples: int = 1,
    fixed_horizons: list[int] | None = None,
    start_weight: float = 1.0,
    end_weight: float = 1.0,
    vpt_tail_weight: float = 0.0,
    vpt_threshold: float = 0.25,
    vpt_soft_threshold: float = 0.15,
    vpt_tail_fraction: float = 0.25,
    vpt_start_weight: float = 1.0,
    vpt_end_weight: float = 1.0,
) -> torch.Tensor:
    horizon = max(1, int(horizon))
    samples = max(1, int(samples))
    horizons = [int(h) for h in (fixed_horizons or []) if int(h) > 0]
    total = states.new_tensor(0.0)
    count = 0
    for fixed_horizon in horizons:
        total = total + _single_rollout_loss(
            model,
            states,
            actions,
            normalizer,
            warmup_steps,
            fixed_horizon,
            start_weight,
            end_weight,
            vpt_tail_weight,
            vpt_threshold,
            vpt_soft_threshold,
            vpt_tail_fraction,
            vpt_start_weight,
            vpt_end_weight,
        )
        count += 1
    for _ in range(samples):
        low = max(1, horizon // 3)
        if horizon > low:
            sampled_horizon = int(torch.randint(low, horizon + 1, (), device=states.device).item())
        else:
            sampled_horizon = horizon
        total = total + _single_rollout_loss(
            model,
            states,
            actions,
            normalizer,
            warmup_steps,
            sampled_horizon,
            start_weight,
            end_weight,
            vpt_tail_weight,
            vpt_threshold,
            vpt_soft_threshold,
            vpt_tail_fraction,
            vpt_start_weight,
            vpt_end_weight,
        )
        count += 1
    return total / float(count)


def compute_loss(model, batch: dict[str, torch.Tensor], normalizer, cfg: dict):
    loss_cfg = cfg["loss"]
    states = batch["states"]
    actions = batch["actions"]
    one = one_step_delta_loss(model, states, actions, normalizer)
    horizon = int(loss_cfg.get("rollout_train_horizon", 5))
    warmup = int(cfg["eval"].get("warmup_steps", 5))
    roll = rollout_loss(
        model,
        states,
        actions,
        normalizer,
        warmup_steps=warmup,
        horizon=horizon,
        samples=int(loss_cfg.get("rollout_samples", 1)),
        fixed_horizons=[int(h) for h in loss_cfg.get("rollout_fixed_horizons", [])],
        start_weight=float(loss_cfg.get("rollout_start_weight", 1.0)),
        end_weight=float(loss_cfg.get("rollout_end_weight", 1.0)),
        vpt_tail_weight=float(loss_cfg.get("vpt_tail_weight", 0.0)),
        vpt_threshold=float(loss_cfg.get("vpt_threshold", 0.25)),
        vpt_soft_threshold=float(loss_cfg.get("vpt_soft_threshold", 0.15)),
        vpt_tail_fraction=float(loss_cfg.get("vpt_tail_fraction", 0.25)),
        vpt_start_weight=float(loss_cfg.get("vpt_start_weight", 1.0)),
        vpt_end_weight=float(loss_cfg.get("vpt_end_weight", 1.0)),
    )
    total = float(loss_cfg.get("one_step_weight", 1.0)) * one + float(loss_cfg.get("rollout_weight", 0.3)) * roll
    return total, {
        "loss/total": float(total.detach().cpu()),
        "loss/one_step": float(one.detach().cpu()),
        "loss/rollout": float(roll.detach().cpu()),
        "loss/rollout_horizon": float(horizon),
    }
