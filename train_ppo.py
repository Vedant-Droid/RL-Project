#!/usr/bin/env python3
"""
Train a PPO policy for the goal-reaching TurtleBot task.

Policy: PPO with an MLP actor-critic (2 x 128 hidden units each).
Chosen because the action space is continuous (linear/angular velocity),
PPO is stable without much tuning, and works well with noisy real-time sims.

Usage:
    python3 train_ppo.py --world default --timesteps 200000
    python3 train_ppo.py --resume ppo_tb3_goal.zip --timesteps 100000
"""
import argparse

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.monitor import Monitor

from rl_env import TurtleBotGoalEnv


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--timesteps', type=int, default=200_000)
    p.add_argument('--world', default='default', help='Gazebo world name (for reset)')
    p.add_argument('--stamped', action=argparse.BooleanOptionalAction, default=True,
                   help='publish TwistStamped (Jazzy TurtleBot3). Use --no-stamped for Twist')
    p.add_argument('--no-sim-reset', action='store_true',
                   help='do not reset Gazebo between episodes')
    p.add_argument('--resume', default=None, help='path to a saved model to continue')
    p.add_argument('--out', default='ppo_tb3_goal')
    args = p.parse_args()

    env = Monitor(TurtleBotGoalEnv(
        world_name=args.world,
        use_stamped=args.stamped,
        do_sim_reset=not args.no_sim_reset,
    ))

    if args.resume:
        model = PPO.load(args.resume, env=env)
    else:
        model = PPO(
            'MlpPolicy',
            env,
            policy_kwargs=dict(net_arch=dict(pi=[128, 128], vf=[128, 128])),
            learning_rate=3e-4,
            n_steps=1024,
            batch_size=64,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.005,
            verbose=1,
        )

    checkpoint = CheckpointCallback(save_freq=10_000, save_path='./checkpoints',
                                    name_prefix='ppo_tb3')
    try:
        model.learn(total_timesteps=args.timesteps, callback=checkpoint,
                    reset_num_timesteps=not args.resume)
    except KeyboardInterrupt:
        print('Interrupted, saving model...')
    finally:
        model.save(args.out)
        print(f'Saved model to {args.out}.zip')
        env.close()


if __name__ == '__main__':
    main()