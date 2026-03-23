"""Hugging Face GPT-2 wrapper compatible with diffusion trainers."""

from __future__ import annotations

from typing import Optional

import torch


class HFGPT2Wrapper(torch.nn.Module):
  """Minimal GPT-2 wrapper exposing a unified `forward` signature.

  Parameters mirror the upstream helper located in the super-project while
  adding light glue so that diffusion trainers can pass extra keyword
  arguments (e.g. ``sigma``).
  """

  def __init__(
      self,
      pretrained_model_name_or_path: str = 'gpt2',
      bidirectional: bool = False,
      attn_type: Optional[str] = None,
      vocab_size: Optional[int] = None,
      max_seq_len: Optional[int] = None):
    super().__init__()

    from transformers import AutoConfig, AutoModelForCausalLM

    del max_seq_len  # Hydra parity; unused by HF GPT-2

    config = AutoConfig.from_pretrained(pretrained_model_name_or_path)
    self.model = AutoModelForCausalLM.from_pretrained(
        pretrained_model_name_or_path,
        config=config,
    )

    if attn_type is None:
      self.attn_type = 'bidirectional' if bidirectional else 'causal'
    else:
      self.attn_type = attn_type

    # No custom attention masks / annealing support

    if (self.attn_type in ('bidirectional', 'custom')
        and getattr(config, 'model_type', None) == 'gpt2'):
      # Remove GPT-2 causal mask by marking attention bias as fully visible.
      for block in self.model.transformer.h:
        if hasattr(block.attn, 'bias') and block.attn.bias is not None:
          block.attn.bias.fill_(True)

    if isinstance(vocab_size, int) and vocab_size > 0:
      embeddings = self.model.get_input_embeddings()
      if embeddings.weight.size(0) != vocab_size:
        self.model.resize_token_embeddings(vocab_size, pad_to_multiple_of=2)

    # Removed attention mask provider support

  def forward(
      self,
      input_ids: torch.Tensor,
      *unused_args,
      src_key_padding_mask: Optional[torch.Tensor] = None,
      t: Optional[torch.Tensor] = None,
      return_insertion_count: bool = False,
      **unused_kwargs) -> torch.Tensor:
    del src_key_padding_mask, t, return_insertion_count

    kwargs = {
        'input_ids': input_ids,
        'use_cache': False,
        'return_dict': False,
    }

    # No custom 4D attention mask injection

    outputs = self.model(**kwargs)
    logits = outputs[0]
    return logits

