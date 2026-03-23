"""Algorithm implementations for discrete diffusion.

All OSS entrypoints resolve algorithms via Hydra `_target_` strings declared in
`configs/algo/*.yaml`. Example:

```
algo:
  _target_: discrete_diffusion.algorithms.mdlm.MDLM
  name: mdlm
  ...
```

See docs/01_algorithms.md for detailed documentation on algorithm structure.
"""

from .ar import AR  # noqa: F401
from .mdlm import MDLM  # noqa: F401
from .udlm import UDLM  # noqa: F401
from .campbell import CampbellTrainer  # noqa: F401
from .gdds import GDDSDiffusion  # noqa: F401

__all__ = [
  'GDDSDiffusion',
  'AR',
  'MDLM',
  'UDLM',
  'CampbellTrainer',
]
