# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
"""Run the upstream full-output/full-gradient oracle against an installed A5 wheel."""
from pathlib import Path
import json
import runpy
import sys

import hyper_multicore


def main() -> None:
    """Keep the installed package selected while loading the repository's reference."""
    root = Path(__file__).resolve().parents[2]
    package = Path(hyper_multicore.__file__).resolve()
    if root in package.parents:
        raise RuntimeError('Run outside the source tree with an installed hyper_multicore package')
    sys.path.insert(0, str(root))
    worker = runpy.run_path(str(root/'tests/torch/_test_mega_moe.py'))
    import torch.distributed as dist
    try:
        worker['test_mega_moe_level0_balanced']()
        print(json.dumps({'rank': dist.get_rank(), 'passed': True,
                          'cases': ['balanced', 'zero_token_experts', 'skew', 'single_destination'],
                          'layouts': ['EIO', 'EOI'], 'installed_package': str(package)}), flush=True)
    finally:
        dist.destroy_process_group()


if __name__ == '__main__':
    main()
