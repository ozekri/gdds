"""Campbell training."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from . import base as trainer_base
from ..forward_process.campbell import CampbellEventSampler
from ..utils.rank_masks import compute_rank_from_tau


def _clamp_tau(tau: torch.Tensor) -> torch.Tensor:
    return tau.clamp(min=0.0, max=1.0)


def _replace_non_finite(tensor: torch.Tensor, fill_value: float) -> torch.Tensor:
    if torch.isfinite(tensor).all():
        return tensor
    return torch.where(torch.isfinite(tensor), tensor, torch.full_like(tensor, fill_value))


class CampbellTrainer(trainer_base.AbsorbingState):
    """Trainer for the Campbell backbone."""

    def __init__(self, config, tokenizer):
        super().__init__(config, tokenizer)

        self.forward_process_type = getattr(config.algo, "forward_process_type", "absorbing")
        self.K_trunc = getattr(config.algo, "K_trunc", 1)
        self.K_max = getattr(config.algo, "K_max", 10)
        self.jump_loss_weights = getattr(config.algo, "jump_loss_weights", None)
        self.time_grid_size = getattr(config.algo, "time_grid_size", 4096)
        self.time_eps = getattr(config.algo, "time_eps", 1e-5)
        self.debug_masks = getattr(config.algo, "debug_masks", False)

        self.event_sampler = CampbellEventSampler(
            schedule=self.noise,
            tokenizer=self.tokenizer,
            mode=self.forward_process_type,
            mask_token_id=self.mask_id,
            time_grid_size=self.time_grid_size,
            time_eps=self.time_eps,
        )

        self.pad_token_id = self._validate_special_tokens()
        self._attach_special_tokens_to_backbone()

        if self.backbone.vocab_size != len(self.tokenizer):
            raise ValueError(
                f"Model vocab_size ({self.backbone.vocab_size}) != tokenizer vocab_size ({len(self.tokenizer)}). "
                "Build the model with vocab_size=len(tokenizer)."
            )

        super()._validate_configuration()

    def _validate_special_tokens(self) -> int:
        pad_token_id = self.tokenizer.pad_token_id
        mask_token_id = self.mask_id

        if pad_token_id is None:
            raise ValueError(
                f"Tokenizer {self.tokenizer} must have pad_token_id set."
            )
        if mask_token_id is None:
            raise ValueError(
                f"Tokenizer {self.tokenizer} must have mask_token_id set."
            )
        if not 0 <= pad_token_id < self.vocab_size:
            raise ValueError(f"pad_token_id ({pad_token_id}) must be in [0, {self.vocab_size})")
        if not 0 <= mask_token_id < self.vocab_size:
            raise ValueError(f"mask_token_id ({mask_token_id}) must be in [0, {self.vocab_size})")

        return pad_token_id

    def _attach_special_tokens_to_backbone(self) -> None:
        for attr, value in (("mask_token_id", self.mask_id), ("pad_token_id", self.pad_token_id)):
            if hasattr(self.backbone, attr):
                setattr(self.backbone, attr, value)

    def _process_model_output(self, model_output, xt, sigma):
        del xt, sigma
        return model_output

    def _loss(
        self,
        x0,
        valid_tokens,
        current_accumulation_step=None,
        train_mode=False,
        compute_extra_val_metrics=False,
    ):
        del current_accumulation_step, train_mode, compute_extra_val_metrics

        if self.K_trunc == 1:
            return self._loss_single_jump(x0, valid_tokens)
        return self._loss_multi_jump(x0, valid_tokens)

    def _loss_single_jump(self, x0, valid_tokens):
        batch_size, seq_len = x0.shape
        z_obs, tau, rank = self.event_sampler.sample(x0)
        pad_mask = ~valid_tokens.bool() if valid_tokens is not None else None

        logits = self.backbone(
            x0_ids=x0,
            xobs_ids=z_obs,
            tau=_clamp_tau(tau),
            rank=rank,
            pad_mask=pad_mask,
        )
        logits[:, :, self.mask_id] = self.neg_infinity

        loss_mask = (tau <= 1.0) & torch.isfinite(tau)
        if valid_tokens is not None:
            loss_mask &= valid_tokens.bool()

        num_events = loss_mask.float().sum()
        if num_events == 0:
            loss_mask = torch.isfinite(tau)
            if valid_tokens is not None:
                loss_mask &= valid_tokens.bool()
            num_events = loss_mask.float().sum()
        if num_events == 0:
            raise ValueError("No valid events in batch")

        loss_per_pos = F.cross_entropy(
            logits.view(-1, self.backbone.vocab_size),
            x0.view(-1),
            reduction="none",
        ).view(batch_size, seq_len)

        masked_loss = loss_per_pos * loss_mask.float()
        return trainer_base.Loss(
            loss=masked_loss.sum() / num_events,
            nlls=masked_loss.sum(),
            num_tokens=num_events,
        )

    def _get_jump_loss_weights(self) -> list[float]:
        if self.jump_loss_weights is None:
            return [1.0] * self.K_trunc

        weights = list(self.jump_loss_weights[: self.K_trunc])
        if len(weights) < self.K_trunc:
            weights.extend([1.0] * (self.K_trunc - len(weights)))
        return weights

    def _loss_multi_jump(self, x0, valid_tokens):
        batch_size, seq_len = x0.shape
        if self.forward_process_type != "uniform":
            return self._loss_single_jump(x0, valid_tokens)

        z_t, N_t, jump_times, pre_jump_targets, post_jump_obs, event_mask = (
            self.event_sampler.build_training_batch_multijump(
                x0,
                t=1.0,
                max_jumps=self.K_max,
            )
        )
        del z_t, N_t

        if valid_tokens is not None:
            event_mask &= valid_tokens.bool().unsqueeze(-1)

        pad_mask = ~valid_tokens.bool() if valid_tokens is not None else None
        total_nll = x0.new_tensor(0.0, dtype=torch.float32)
        total_events = x0.new_tensor(0.0, dtype=torch.float32)

        for jump_idx, weight in enumerate(self._get_jump_loss_weights()):
            if jump_idx >= self.K_trunc:
                break

            jump_mask = event_mask[:, :, jump_idx]
            if not jump_mask.any():
                break

            tau_k = jump_times[:, :, jump_idx]
            tau_k = torch.where(
                jump_mask & torch.isfinite(tau_k),
                tau_k,
                torch.full_like(tau_k, 1.0 + self.time_eps),
            )
            tau_k = _replace_non_finite(tau_k, 1.0 + self.time_eps)

            logits_k = self.backbone(
                x0_ids=x0,
                xobs_ids=post_jump_obs[:, :, jump_idx],
                tau=_clamp_tau(tau_k),
                rank=compute_rank_from_tau(tau_k),
                pad_mask=pad_mask,
            )
            logits_k = _replace_non_finite(logits_k, 0.0)
            logits_k[:, :, self.mask_id] = self.neg_infinity

            ce_k = F.cross_entropy(
                logits_k.view(-1, self.backbone.vocab_size),
                pre_jump_targets[:, :, jump_idx].view(-1),
                reduction="none",
            ).view(batch_size, seq_len)
            ce_k = _replace_non_finite(ce_k, 0.0)

            jump_events = jump_mask.float().sum()
            total_nll = total_nll + weight * (ce_k * jump_mask.float()).sum()
            total_events = total_events + jump_events

        if total_events == 0:
            return self._loss_single_jump(x0, valid_tokens)

        return trainer_base.Loss(
            loss=total_nll / total_events,
            nlls=total_nll,
            num_tokens=total_events,
        )

    def training_step(self, batch, batch_idx):
        current_accumulation_step = batch_idx % self.trainer.accumulate_grad_batches
        losses = self._loss(
            batch["input_ids"],
            batch["attention_mask"],
            current_accumulation_step,
            train_mode=True,
        )
        self.metrics.update_train(losses.nlls, losses.num_tokens)
        self.log(
            name="trainer/loss",
            value=losses.loss,
            on_step=True,
            on_epoch=False,
            sync_dist=True,
            prog_bar=True,
        )
        return losses.loss

    def validation_step(self, batch, batch_idx):
        del batch_idx
        losses = self._loss(
            batch["input_ids"],
            batch["attention_mask"],
            train_mode=False,
        )
        self.metrics.update_valid(losses.nlls, losses.num_tokens)
        return losses.loss

    def on_validation_epoch_end(self):
        valid_metrics = {
            key: metric.compute()
            for key, metric in self.metrics.valid_nlls.items()
            if getattr(metric, "weight", 0) > 0
        }
        if valid_metrics:
            self.log_dict(valid_metrics, on_step=False, on_epoch=True, sync_dist=True)

        if hasattr(self.metrics, "valid_aux") and self.metrics.valid_aux.weight > 0:
            self.log(
                name="val/aux",
                value=self.metrics.valid_aux.compute(),
                on_step=False,
                on_epoch=True,
                sync_dist=True,
            )

        should_sample = (
            (self.config.eval.compute_perplexity_on_sanity or not self.trainer.sanity_checking)
            and self.config.eval.generate_samples
        )
        if should_sample:
            try:
                samples = None
                text_samples = None
                for _ in range(self.config.sampling.num_sample_batches):
                    samples = self.generate_samples(num_samples=self.config.loader.eval_batch_size)
                    self.metrics.record_entropy(samples)
                    text_samples = self.tokenizer.batch_decode(
                        samples,
                        clean_up_tokenization_spaces=False,
                    )

                if text_samples is not None:
                    if self.trainer.global_rank == 0 and hasattr(self.trainer.logger, "log_table"):
                        text_samples = text_samples[: self.config.sampling.num_sample_log]
                        self.trainer.logger.log_table(
                            key=f"samples@global_step{self.global_step}",
                            columns=["Generated Samples"],
                            data=[[sample] for sample in text_samples],
                        )
                    self.log(
                        "val/sample_entropy",
                        self.metrics.sample_entropy.compute(),
                        on_epoch=True,
                        on_step=False,
                        sync_dist=True,
                    )
            except Exception as exc:
                print(f"Sampling failed at step {self.global_step}: {exc}")

        self._train_mode()

    @torch.no_grad()
    def build_inference_inputs(
        self,
        x: torch.Tensor,
        attn_mask: torch.Tensor | None,
        step_time: torch.Tensor,
    ) -> dict:
        _, seq_len = x.shape

        if step_time.ndim == 2:
            step_time = step_time[:, 0]

        positions = torch.arange(seq_len, device=x.device).unsqueeze(0).expand(x.shape[0], seq_len)
        return {
            "token_ids": x,
            "positions": positions,
            "step_time": step_time.float(),
            "attn_mask": attn_mask,
        }


__all__ = ["CampbellTrainer"]
