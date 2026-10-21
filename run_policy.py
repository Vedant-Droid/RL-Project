#!/usr/bin/env python3
"""
Run a trained PPO policy on the robot / simulation and report success rate.

Usage:
    python3 run_policy.py --model ppo_tb3_goal.zip --episodes 10
"""
import argparse

from stable_baselines3 import PPO

from rl_env import TurtleBotGoalEnv


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model', default='ppo_tb3_goal.zip')
    p.add_argument('--episodes', type=int, default=10)
    p.add_argument('--world', default='default')
    p.add_argument('--stamped', action=argparse.BooleanOptionalAction, default=True)
    args = p.parse_args()

    env = TurtleBotGoalEnv(world_name=args.world, use_stamped=args.stamped)
    model = PPO.load(args.model)

    successes = 0
    try:
        for ep in range(1, args.episodes + 1):
            obs, info = env.reset()
            print(f'Episode {ep}: goal at ({info["goal"][0]:.2f}, {info["goal"][1]:.2f})')
            done = False
            total = 0.0
            while not done:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = env.step(action)
                total += reward
                done = terminated or truncated
            ok = info.get('is_success', False)
            successes += int(ok)
            outcome = 'SUCCESS' if ok else ('COLLISION' if info.get('collision') else 'TIMEOUT')
            print(f'  {outcome}  reward={total:.1f}  final distance={info["distance"]:.2f} m')
        print(f'Success rate: {successes}/{args.episodes}')
    except KeyboardInterrupt:
        pass
    finally:
        env.close()


if __name__ == '__main__':
    main()