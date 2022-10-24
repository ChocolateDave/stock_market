# =============================================================================
# @file   run_ddpg.py
# @author Juanwu Lu
# @date   Oct-23-22
# =============================================================================
from __future__ import annotations

import argparse
import os
from typing import Any, Mapping

import gym
import torch as th
from src.trainer.ddpg_trainer import DDPGTrainer
from src.utils import load_config


def main(args: Mapping[str, Any]) -> None:
    env = gym.make(**args['Env'])
    trainer = DDPGTrainer(env, **args['Trainer'])
    trainer.train(args['execute_at_train'])


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
    parser.add_argument('--log-dir', type=str, default='../logs',
                        help='Logging direction.')
    parser.add_argument('--execute-at-train', action='store_true',
                        default=False, help='Evaluate along with training.')

    args = vars(parser.parse_args())
    config = load_config(args['config'])
    for key, val in args.items():
        if val is not None:
            config[key] = val

    # Allocate device
    if args['gpu']:
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        config["trainer"]["device"] = th.device('cuda')
        if len(args['gpu_id']) > 1:
            config["_parallel"] = True
            os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(args.gpu)
        else:
            os.environ['CUDA_VISIBLE_DEVICES'] = str(args['gpu_id'])[0]

    main(config)
