# =============================================================================
# @file   run_maddpg.py
# @author Juanwu Lu
# @date   Oct-23-22
# =============================================================================
"""Main handle for running MADDPG environment."""
from __future__ import annotations

import argparse
import os
from typing import Any, List, Mapping

import torch as th
from pettingzoo import ParallelEnv, mpe
from src.environment.stock_market import (
    LogarithmAndIntActionWrapper, StockMarketEnv)
from src.trainer.maddpg_trainer import MADDPGTrainer
from src.utils import load_config


def make_env(**kwargs) -> ParallelEnv:
    _id = kwargs['id'].lower()
    kwargs = {k: v for k, v in kwargs.items() if k != 'id'}
    if _id == 'simple_adversary_v2':
        return mpe.simple_adversary_v2.parallel_env(**kwargs)
    elif _id == 'simple_spread_v2':
        return mpe.simple_spred_v2.parallel_env(**kwargs)
    elif _id == 'simple_tag_v2':
        return mpe.simple_tag_v2.parallel_env(**kwargs)
    elif _id == 'stock_market':
        # TODO (Juanwu): implement stock market environment
        return LogarithmAndIntActionWrapper(StockMarketEnv(**kwargs))
    else:
        raise ValueError('Unsupported environment name %s' % _id)


def main(args: Mapping[str, Any]) -> None:
    env = make_env(**args['Env'])
    trainer = MADDPGTrainer(env, **args['Trainer'])
    trainer.train()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, required=True,
                        help='Configuration file path.')
    parser.add_argument('--gpu', action='store_true', default=False,
                        help='Enable training on GPU.')
    parser.add_argument('-i', '--gpu-id', nargs='+', type=int, default=0,
                        help='GPU devices to train on.')
    parser.add_argument('-b', '--batch-size', type=int, help='Batch Size.')
    parser.add_argument('-n', '--num-episodes', type=int, help='Total epochs.')
    parser.add_argument('--discount', type=float, help='Discount Factor')
    parser.add_argument('--work-dir', type=str, default=None,
                        help='Working directory.')

    args = vars(parser.parse_args())
    config = load_config(args['config'])
    for key, val in args.items():
        if val is not None:
            config['Trainer'][key] = val

    # Allocate device
    if args['gpu']:
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        config["Trainer"]["device"] = th.device('cuda')
        if isinstance(args['gpu_id'], List):
            config["_parallel"] = True
            os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(
                [str(i) for i in args['gpu_id']]
            )
        else:
            os.environ['CUDA_VISIBLE_DEVICES'] = str(args['gpu_id'])[0]

    main(config)
