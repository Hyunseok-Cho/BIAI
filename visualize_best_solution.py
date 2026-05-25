import argparse
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from mountain_car_ga import (
    ACTION_MEANINGS,
    ENV_NAME,
    MAX_STEPS,
    calculate_fitness_components,
    create_environment,
    get_action,
)


DEFAULT_POLICY_FILE = "best_individual.npy"
DEFAULT_SCREENSHOT_FILE = "best_solution_screenshot.png"
DEFAULT_SEED = 2042


def load_policy(policy_path):
    path = Path(policy_path)

    if not path.exists():
        raise FileNotFoundError(
            f"Policy file '{policy_path}' was not found. "
            "Run mountain_car_ga.py first to create best_individual.npy."
        )

    return np.load(path)


def run_policy(policy, seed, render_mode=None, delay=0.02):
    env = create_environment(render_mode=render_mode)
    observation, info = env.reset(seed=seed)

    total_reward = 0.0
    start_position = float(observation[0])
    final_position = float(observation[0])
    max_position = float(observation[0])
    reached_goal = False
    final_frame = env.render() if render_mode == "rgb_array" else None

    for step in range(MAX_STEPS):
        action = int(get_action(policy, observation))
        observation, reward, terminated, truncated, info = env.step(action)

        total_reward += reward
        final_position = float(observation[0])
        max_position = max(max_position, final_position)

        if render_mode == "rgb_array":
            final_frame = env.render()
        elif render_mode == "human" and delay > 0:
            time.sleep(delay)

        if terminated:
            reached_goal = True
            break

        if truncated:
            break

    env.close()

    steps_taken = step + 1
    fitness_components = calculate_fitness_components(
        total_reward,
        start_position,
        max_position,
        reached_goal,
    )

    return {
        "seed": seed,
        "total_reward": float(total_reward),
        "start_position": start_position,
        "final_position": final_position,
        "max_position": max_position,
        "reached_goal": reached_goal,
        "steps_taken": steps_taken,
        "fitness_components": fitness_components,
        "final_frame": final_frame,
    }


def save_screenshot(frame, output_path):
    if frame is None:
        raise ValueError("No rendered frame was captured.")

    plt.imsave(output_path, frame)


def print_summary(result):
    fitness_components = result["fitness_components"]

    print("Best solution visualization summary")
    print(f"Environment: {ENV_NAME}")
    print(f"Seed: {result['seed']}")
    print(f"Actions: {ACTION_MEANINGS}")
    print(f"Total reward: {result['total_reward']:.2f}")
    print(f"Start position: {result['start_position']:.4f}")
    print(f"Final position: {result['final_position']:.4f}")
    print(f"Max position: {result['max_position']:.4f}")
    print(f"Reached goal: {result['reached_goal']}")
    print(f"Steps taken: {result['steps_taken']}")
    print(f"Goal progress: {fitness_components['goal_progress']:.4f}")
    print(f"Progress bonus: {fitness_components['progress_bonus']:.2f}")
    print(f"Goal bonus: {fitness_components['goal_bonus']:.2f}")
    print(f"Fitness: {fitness_components['fitness']:.2f}")


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Visualize the saved best MountainCar-v0 policy without retraining."
        )
    )
    parser.add_argument(
        "--mode",
        choices=["human", "screenshot"],
        default="human",
        help=(
            "human opens a live render window. screenshot saves the final rendered "
            "frame to an image file."
        ),
    )
    parser.add_argument(
        "--policy",
        default=DEFAULT_POLICY_FILE,
        help="Path to the saved best policy .npy file.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help="Environment reset seed used for the visualization.",
    )
    parser.add_argument(
        "--output",
        default=DEFAULT_SCREENSHOT_FILE,
        help="Output image path used in screenshot mode.",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=0.02,
        help="Delay between rendered steps in human mode.",
    )

    return parser.parse_args()


def main():
    args = parse_args()
    policy = load_policy(args.policy)

    if args.mode == "human":
        result = run_policy(
            policy,
            seed=args.seed,
            render_mode="human",
            delay=args.delay,
        )
    else:
        result = run_policy(policy, seed=args.seed, render_mode="rgb_array")
        save_screenshot(result["final_frame"], args.output)
        print(f"Saved screenshot: {args.output}")

    print_summary(result)


if __name__ == "__main__":
    main()
