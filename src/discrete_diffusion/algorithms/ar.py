"""AR algorithm implementation extracted from :mod:`algorithms.algo`."""

import torch

from . import base as trainer_base


class AR(trainer_base.TrainerBase):
  def __init__(self, config, tokenizer):
    self.mask_id, vocab_size = trainer_base.ensure_mask_token(tokenizer)
    super().__init__(config, tokenizer, vocab_size=vocab_size)
    self.save_hyperparameters()
    self._validate_configuration()

  def _validate_configuration(self):
    super()._validate_configuration()
    assert not self.config.algo.time_conditioning
    assert self.config.prior.type == 'none'

  def _process_model_input(self, x0, valid_tokens):
    input_tokens = x0[:, :-1]
    output_tokens = x0[:, 1:]
    valid_tokens = valid_tokens[:, 1:]
    return input_tokens, output_tokens, valid_tokens

  def nll(self, input_tokens, output_tokens, current_accumulation_step=None, train_mode=False):
    output = self.backbone(input_tokens, None)
    output[:, :, self.mask_id] = self.neg_infinity
    output = output.log_softmax(-1)
    return -output.gather(-1, output_tokens[:, :, None])[:, :, 0]

  def _process_sigma(self, sigma):
    return None
