"""Module entrypoint and Hydra-based CLI for discrete diffusion.

Usage:
  python -m discrete_diffusion [Hydra overrides]
"""

import os
from pathlib import Path

import hydra
import lightning as L
import omegaconf
import torch
import fsspec
import rich.syntax
import rich.tree

from .train import train as train_function
from . import utils


CONFIG_PATH = (Path(__file__).resolve().parents[2] / 'configs').as_posix()


def _register_resolver(name, resolver):
  if omegaconf.OmegaConf.has_resolver(name):
    return
  omegaconf.OmegaConf.register_new_resolver(name, resolver)


def _mul_resolver(*args):
  import functools, operator
  return functools.reduce(operator.mul, args) if args else ValueError('`mul` resolver requires at least one argument.')


# Register OmegaConf resolvers for Hydra configs
_register_resolver('cwd', os.getcwd)
_register_resolver('device_count', torch.cuda.device_count)
_register_resolver('div_up', lambda x, y: (x + y - 1) // y)
_register_resolver('mul', _mul_resolver)
_register_resolver('sub', lambda x, y: x - y)


@L.pytorch.utilities.rank_zero_only
def _print_config(config: omegaconf.DictConfig, resolve: bool = True, save_cfg: bool = True) -> None:
  style = 'dim'
  tree = rich.tree.Tree('CONFIG', style=style, guide_style=style)

  fields = config.keys()
  for field in fields:
    branch = tree.add(field, style=style, guide_style=style)

    config_section = config.get(field)
    branch_content = str(config_section)
    if isinstance(config_section, omegaconf.DictConfig):
      branch_content = omegaconf.OmegaConf.to_yaml(
        config_section, resolve=resolve)

    branch.add(rich.syntax.Syntax(branch_content, 'yaml'))
  rich.print(tree)
  if save_cfg:
    with fsspec.open(f'{config.checkpointing.save_dir}/config_tree.txt', 'w') as fp:
      rich.print(tree, file=fp)


@hydra.main(version_base=None, config_path=CONFIG_PATH, config_name='config')
def main(config):
  L.seed_everything(config.seed)
  _print_config(config)

  logger = utils.get_logger(__name__)
  logger.info('Starting training...')
  train_function(config)
  logger.info('Training completed successfully.')


if __name__ == '__main__':
  main()
