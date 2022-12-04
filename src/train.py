# =============================================================================
# @file   train.py
# @author Juanwu Lu
# @date   Nov-28-22
# =============================================================================
from __future__ import annotations

import argparse
import os
from typing import List

import torch as th

from src.env import LogarithmAndIntActionWrapper, StockMarketEnv
from src.trainer import MADDPGTrainer


def main() -> None:
    parser = argparse.ArgumentParser()

    # Environment Arguments
    parser.add_argument('--num-agents', type=int, required=True,
                        help='Number of agents in the stock market.')
    parser.add_argument('--max-cycles', type=int, default=390,
                        help='Maximum number of minutes for market episode.')
    parser.add_argument('--num-company', type=int, default=5,
                        help='Number of companies in the stock market.')
    parser.add_argument('--start-prices', type=float, default=100.0,
                        help='Start price of the stocks.')
    parser.add_argument('--budget-discount', type=float, default=0.9,
                        help='Budge discount factor over time.')
    parser.add_argument('--worth-of-stocks', type=float, default=0.1,
                        help='Factor related to the worth of stock.')

    # Trainer Arguments
    parser.add_argument('--action-range', type=float, nargs='+',
                        default=[-0.99999, 0.99999], help='Action range.')
    parser.add_argument('-bs', '--batch-size', type=int, default=64,
                        help='Batch size for training.')
    parser.add_argument('--buffer-size', type=int, default=100000,
                        help='An integer memory replay buffer size.')
    parser.add_argument('--gpu', action='store_true', default=False,
                        help='Enabling training on GPU devices.')
    parser.add_argument('--gpu-id', type=int, nargs='+', default=0,
                        help='GPU device ids to train on.')
    parser.add_argument('--max-episode-steps', type=int, default=None,
                        help='Maximum number of steps for each episode.')
    parser.add_argument('--num-episodes', type=int, default=1000,
                        help='Total number of episodes to train.')
    parser.add_argument('--num-warm-up-steps', type=int, default=1000,
                        help='Total number of warm up steps before training.')
    parser.add_argument('-lr', '--learning-rate', type=float, default=1e-4,
                        help='Unified learning rate for critic and policy.')
    parser.add_argument('-clr', '--critic-lr', type=float, default=None,
                        help='Learning rate for Critic module.')
    parser.add_argument('-plr', '--policy-lr', type=float, default=None,
                        help='Learning rate for Policy module.')
    parser.add_argument('--discount', type=float, default=0.99,
                        help='Discount factor for bellman target estimation.')
    parser.add_argument('--soft-update-tau', type=float, default=0.01,
                        help='Update factor for soft target network update.')
    parser.add_argument('--exp-name', type=str, default='default',
                        help='Customize experiment name for logging.')
    parser.add_argument('--eval-frequency', type=int, default=-1,
                        help='Evaluation frequency when training, -1 for not.')
    parser.add_argument('--save-frequency', type=int, default=100,
                        help='Checkpoint saving episodic frequency.')
    parser.add_argument('--seed', type=int, default=42,
                        help='Seed for training.')
    args = vars(parser.parse_args())

    assert len(args['action_range']) == 2, ValueError(
        'Expect action range to be a sequence of 2 floats, '
        f'but got a sequence of size {len(args["action_range"]):d}.'
    )

    if args['gpu']:
        os.environ['CUDA_DEIVICE_ORDER'] = 'PCI_BUS_ID'
        device = th.device('cuda')
        if isinstance(args['gpu_id'], List):
            args['parallel'] = True
            os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(
                [str(i) for i in args['gpu_id']]
            )
        else:
            os.environ['CUDA_VISIBLE_DEVICES'] = str(args['gpu_id'])[0]

    env = LogarithmAndIntActionWrapper(StockMarketEnv(
        num_agents=args['num_agents'],
        max_cycles=args['max_cycles'],
        num_company=args['num_company'],
        seed=args['seed']
    ))
    trainer = MADDPGTrainer(
        env=env,
        batch_size=args['batch_size'],
        buffer_size=args['buffer_size'],
        device=device,
        action_range=args['action_range'],
        max_episode_steps=args['max_episode_steps'],
        num_episodes=args['num_episodes'],
        num_warm_up_steps=args['num_warm_up_steps'],
        exp_name=args['exp_name'],
        work_dir=args.get('work_dir'),
        eval_frequency=args['eval_frequency'],
        save_frequency=args['save_frequency'],
        learning_rate=args['learning_rate'],
        critic_lr=args['critic_lr'],
        policy_lr=args['policy_lr'],
        discount=args['discount'],
        soft_update_tau=args['soft_update_tau'],
        seed=args['seed']
    )
    trainer.train()


if __name__ == '__main__':
    main()
