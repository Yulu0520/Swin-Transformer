"""Backward-compatible entrypoint for building a Stage 3 daytime bank."""

import copy

from build_memory_bank import build_bank as _build_bank, parse_args as _parse_args


def build_bank(args):
    if getattr(args, 'memory_stage', 3) != 3:
        raise ValueError('build_stage3_bank.py only builds Stage 3; use build_memory_bank.py')
    stage_args = copy.copy(args)
    stage_args.memory_stage = 3
    return _build_bank(stage_args)


def parse_args():
    return _parse_args(memory_stage=3)


if __name__ == '__main__':
    build_bank(parse_args())
