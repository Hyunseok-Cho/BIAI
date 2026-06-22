import gymnasium as gym
import numpy as np
import random
import csv
import matplotlib.pyplot as plt
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "DATA"

ENV_NAME = "MountainCar-v0"

POPULATION_SIZE = 80
GENERATIONS = 60
ELITE_SIZE = 6
TOURNAMENT_SIZE = 5
INITIAL_MUTATION_RATE = 0.05
FINAL_MUTATION_RATE = 0.01
CROSSOVER_RATE = 0.85

POSITION_BINS = 20
VELOCITY_BINS = 20

EPISODES_PER_INDIVIDUAL = 3
MAX_STEPS = 200

POSITION_MIN = -1.2
POSITION_MAX = 0.6
VELOCITY_MIN = -0.07
VELOCITY_MAX = 0.07
GOAL_POSITION = 0.5
STEP_REWARD = -1
PROGRESS_WEIGHT = 500
GOAL_BONUS = 200

ACTION_MEANINGS = {
    0: "accelerate left",
    1: "do nothing",
    2: "accelerate right",
}

RANDOM_SEED = 42
VALIDATION_EPISODES = 10
VALIDATION_SEEDS = [
    RANDOM_SEED + 1000 + seed_index for seed_index in range(VALIDATION_EPISODES)
]
RANDOMIZATION_TRIALS = 20
RANDOMIZATION_SEEDS = [
    RANDOM_SEED + 2000 + seed_index for seed_index in range(RANDOMIZATION_TRIALS)
]

CHAMPIONS_DIR = DATA_DIR / "generation_champions"
TRAINING_RESULTS_FILE = DATA_DIR / "results.csv"
COMPARISON_RESULTS_FILE = DATA_DIR / "generation_comparison.csv"
RANDOMIZATION_RESULTS_FILE = DATA_DIR / "randomization_effects.csv"
FITNESS_PLOT_FILE = DATA_DIR / "fitness_plot.png"
BEST_INDIVIDUAL_FILE = DATA_DIR / "best_individual.npy"
RANDOMIZATION_PLOT_FILE = DATA_DIR / "randomization_effect_plot.png"
GENERATION_REWARD_PLOT_FILE = DATA_DIR / "generation_reward_plot.png"
GENERATION_MAX_POSITION_PLOT_FILE = DATA_DIR / "generation_max_position_plot.png"
GENERATION_STEPS_TO_GOAL_PLOT_FILE = DATA_DIR / "generation_steps_to_goal_plot.png"
GENERATION_SUCCESS_RATE_PLOT_FILE = DATA_DIR / "generation_success_rate_plot.png"
RENDER_FINAL_TEST = False


random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)


def create_environment(render_mode=None):
    return gym.make(ENV_NAME, render_mode=render_mode)


def discretize_observation(observation):
    position = observation[0]
    velocity = observation[1]

    position_index = int(
        (position - POSITION_MIN) / (POSITION_MAX - POSITION_MIN) * POSITION_BINS
    )

    velocity_index = int(
        (velocity - VELOCITY_MIN) / (VELOCITY_MAX - VELOCITY_MIN) * VELOCITY_BINS
    )

    position_index = np.clip(position_index, 0, POSITION_BINS - 1)
    velocity_index = np.clip(velocity_index, 0, VELOCITY_BINS - 1)

    return position_index, velocity_index


def get_action(individual, observation):
    position_index, velocity_index = discretize_observation(observation)
    state_index = position_index * VELOCITY_BINS + velocity_index
    return individual[state_index]


def run_policy_episode(individual, seed=None, render=False):
    if render:
        env = create_environment(render_mode="human")
    else:
        env = create_environment()

    if seed is None:
        observation, info = env.reset()
    else:
        observation, info = env.reset(seed=seed)

    total_reward = 0.0
    start_position = observation[0]
    max_position = observation[0]
    reached_goal = False
    steps_taken = 0
    final_position = observation[0]

    for step in range(MAX_STEPS):
        action = get_action(individual, observation)
        observation, reward, terminated, truncated, info = env.step(action)

        total_reward += reward
        final_position = observation[0]
        max_position = max(max_position, observation[0])
        steps_taken = step + 1

        if terminated:
            reached_goal = True
            break

        if truncated:
            break

    env.close()

    return {
        "total_reward": float(total_reward),
        "start_position": float(start_position),
        "final_position": float(final_position),
        "max_position": float(max_position),
        "reached_goal": reached_goal,
        "steps_taken": steps_taken,
        "steps_to_goal": steps_taken if reached_goal else MAX_STEPS,
    }


def calculate_goal_progress(start_position, max_position):
    distance_to_goal = GOAL_POSITION - start_position

    if distance_to_goal <= 0:
        return 1.0

    progress = (max_position - start_position) / distance_to_goal
    return float(np.clip(progress, 0.0, 1.0))


def calculate_fitness_components(
    total_reward,
    start_position,
    max_position,
    reached_goal,
):
    goal_progress = calculate_goal_progress(start_position, max_position)
    progress_bonus = PROGRESS_WEIGHT * goal_progress
    goal_bonus = GOAL_BONUS if reached_goal else 0
    fitness = total_reward + progress_bonus + goal_bonus

    return {
        "reward_component": float(total_reward),
        "goal_progress": goal_progress,
        "progress_bonus": float(progress_bonus),
        "goal_bonus": float(goal_bonus),
        "fitness": float(fitness),
    }


def calculate_fitness(total_reward, start_position, max_position, reached_goal):
    components = calculate_fitness_components(
        total_reward,
        start_position,
        max_position,
        reached_goal,
    )

    return components["fitness"]


def create_individual():
    chromosome_length = POSITION_BINS * VELOCITY_BINS
    return np.random.randint(0, 3, size=chromosome_length)


def create_population():
    return [create_individual() for _ in range(POPULATION_SIZE)]


def evaluate_individual(individual):
    total_fitness = 0.0

    for _ in range(EPISODES_PER_INDIVIDUAL):
        episode_result = run_policy_episode(individual)
        fitness = calculate_fitness(
            episode_result["total_reward"],
            episode_result["start_position"],
            episode_result["max_position"],
            episode_result["reached_goal"],
        )
        total_fitness += fitness

    return total_fitness / EPISODES_PER_INDIVIDUAL


def evaluate_champion(individual, seeds):
    episode_results = [
        run_policy_episode(individual, seed=seed)
        for seed in seeds
    ]
    fitness_components = [
        calculate_fitness_components(
            result["total_reward"],
            result["start_position"],
            result["max_position"],
            result["reached_goal"],
        )
        for result in episode_results
    ]

    rewards = [result["total_reward"] for result in episode_results]
    max_positions = [result["max_position"] for result in episode_results]
    steps_to_goal = [result["steps_to_goal"] for result in episode_results]
    goal_progress_values = [
        components["goal_progress"] for components in fitness_components
    ]
    progress_bonuses = [
        components["progress_bonus"] for components in fitness_components
    ]
    goal_bonuses = [
        components["goal_bonus"] for components in fitness_components
    ]
    fitness_values = [
        components["fitness"] for components in fitness_components
    ]
    successful_steps = [
        result["steps_to_goal"]
        for result in episode_results
        if result["reached_goal"]
    ]
    successful_episodes = len(successful_steps)

    if successful_steps:
        successful_average_steps = float(np.mean(successful_steps))
    else:
        successful_average_steps = None

    return {
        "validation_episodes": len(episode_results),
        "successful_episodes": successful_episodes,
        "success_rate": successful_episodes / len(episode_results),
        "average_reward": float(np.mean(rewards)),
        "average_max_position": float(np.mean(max_positions)),
        "average_steps_to_goal": float(np.mean(steps_to_goal)),
        "average_goal_progress": float(np.mean(goal_progress_values)),
        "average_progress_bonus": float(np.mean(progress_bonuses)),
        "average_goal_bonus": float(np.mean(goal_bonuses)),
        "average_fitness": float(np.mean(fitness_values)),
        "successful_average_steps_to_goal": successful_average_steps,
    }


def evaluate_population(population):
    fitness_scores = []

    for individual in population:
        fitness = evaluate_individual(individual)
        fitness_scores.append(fitness)

    return fitness_scores


def tournament_selection(population, fitness_scores):
    selected_indices = random.sample(range(len(population)), TOURNAMENT_SIZE)
    best_index = selected_indices[0]

    for index in selected_indices:
        if fitness_scores[index] > fitness_scores[best_index]:
            best_index = index

    return population[best_index].copy()


def crossover(parent1, parent2):
    if random.random() > CROSSOVER_RATE:
        return parent1.copy(), parent2.copy()

    chromosome_length = len(parent1)
    crossover_point = random.randint(1, chromosome_length - 2)

    child1 = np.concatenate((parent1[:crossover_point], parent2[crossover_point:]))
    child2 = np.concatenate((parent2[:crossover_point], parent1[crossover_point:]))

    return child1, child2


def get_mutation_rate(generation):
    if GENERATIONS <= 1:
        return FINAL_MUTATION_RATE

    generation_progress = generation / (GENERATIONS - 1)
    return INITIAL_MUTATION_RATE + (
        FINAL_MUTATION_RATE - INITIAL_MUTATION_RATE
    ) * generation_progress


def mutate(individual, mutation_rate):
    for i in range(len(individual)):
        if random.random() < mutation_rate:
            individual[i] = random.randint(0, 2)

    return individual


def create_next_generation(population, fitness_scores, mutation_rate):
    sorted_indices = np.argsort(fitness_scores)[::-1]

    next_generation = []

    for i in range(ELITE_SIZE):
        elite_index = sorted_indices[i]
        next_generation.append(population[elite_index].copy())

    while len(next_generation) < POPULATION_SIZE:
        parent1 = tournament_selection(population, fitness_scores)
        parent2 = tournament_selection(population, fitness_scores)

        child1, child2 = crossover(parent1, parent2)

        child1 = mutate(child1, mutation_rate)
        child2 = mutate(child2, mutation_rate)

        next_generation.append(child1)

        if len(next_generation) < POPULATION_SIZE:
            next_generation.append(child2)

    return next_generation


def save_generation_champion(individual, generation):
    CHAMPIONS_DIR.mkdir(parents=True, exist_ok=True)
    file_path = CHAMPIONS_DIR / f"generation_{generation:03d}.npy"
    np.save(file_path, individual)


def compare_generation_champions(generation_champions, history):
    comparison_history = []

    print("\nComparing each generation champion...")

    for index, champion in enumerate(generation_champions):
        generation = index + 1
        validation_summary = evaluate_champion(champion, VALIDATION_SEEDS)

        comparison_row = {
            "generation": generation,
            "training_best_fitness": history[index]["best_fitness"],
            "training_average_fitness": history[index]["average_fitness"],
            "validation_average_reward": validation_summary["average_reward"],
            "validation_average_max_position": validation_summary[
                "average_max_position"
            ],
            "validation_average_steps_to_goal": validation_summary[
                "average_steps_to_goal"
            ],
            "validation_average_goal_progress": validation_summary[
                "average_goal_progress"
            ],
            "validation_average_progress_bonus": validation_summary[
                "average_progress_bonus"
            ],
            "validation_average_goal_bonus": validation_summary[
                "average_goal_bonus"
            ],
            "validation_average_fitness": validation_summary["average_fitness"],
            "validation_success_rate": validation_summary["success_rate"],
            "validation_successful_episodes": validation_summary[
                "successful_episodes"
            ],
            "validation_episodes": validation_summary["validation_episodes"],
            "successful_average_steps_to_goal": validation_summary[
                "successful_average_steps_to_goal"
            ],
        }
        comparison_history.append(comparison_row)

        print(
            f"Champion {generation:03d} | "
            f"Reward: {comparison_row['validation_average_reward']:.2f} | "
            f"Max position: "
            f"{comparison_row['validation_average_max_position']:.4f} | "
            f"Progress bonus: "
            f"{comparison_row['validation_average_progress_bonus']:.2f} | "
            f"Success rate: "
            f"{comparison_row['validation_success_rate'] * 100:.1f}%"
        )

    return comparison_history


def analyze_randomization_effects(individual, seeds):
    randomization_history = []

    print("\nChecking randomization effects on reward and bonus...")

    for trial_index, seed in enumerate(seeds, start=1):
        episode_result = run_policy_episode(individual, seed=seed)
        fitness_components = calculate_fitness_components(
            episode_result["total_reward"],
            episode_result["start_position"],
            episode_result["max_position"],
            episode_result["reached_goal"],
        )

        row = {
            "trial": trial_index,
            "seed": seed,
            "total_reward": episode_result["total_reward"],
            "start_position": episode_result["start_position"],
            "final_position": episode_result["final_position"],
            "max_position": episode_result["max_position"],
            "goal_progress": fitness_components["goal_progress"],
            "progress_bonus": fitness_components["progress_bonus"],
            "goal_bonus": fitness_components["goal_bonus"],
            "fitness": fitness_components["fitness"],
            "reached_goal": episode_result["reached_goal"],
            "steps_to_goal": episode_result["steps_to_goal"],
        }
        randomization_history.append(row)

        print(
            f"Seed {seed} | "
            f"Reward: {row['total_reward']:.2f} | "
            f"Progress bonus: {row['progress_bonus']:.2f} | "
            f"Goal bonus: {row['goal_bonus']:.2f} | "
            f"Fitness: {row['fitness']:.2f}"
        )

    return randomization_history


def train():
    DATA_DIR.mkdir(exist_ok=True)
    population = create_population()

    history = []
    generation_champions = []

    best_individual = None
    best_fitness = -float("inf")

    for generation in range(GENERATIONS):
        fitness_scores = evaluate_population(population)

        generation_best = max(fitness_scores)
        generation_average = sum(fitness_scores) / len(fitness_scores)
        generation_best_index = fitness_scores.index(generation_best)
        generation_best_individual = population[generation_best_index].copy()
        mutation_rate = get_mutation_rate(generation)

        generation_champions.append(generation_best_individual)
        save_generation_champion(generation_best_individual, generation + 1)

        if generation_best > best_fitness:
            best_fitness = generation_best
            best_individual = generation_best_individual.copy()

        history.append(
            {
                "generation": generation + 1,
                "best_fitness": generation_best,
                "average_fitness": generation_average,
                "overall_best_fitness": best_fitness,
                "mutation_rate": mutation_rate,
                "crossover_rate": CROSSOVER_RATE,
            }
        )

        print(
            f"Generation {generation + 1:03d} | "
            f"Best: {generation_best:.2f} | "
            f"Average: {generation_average:.2f} | "
            f"Overall best: {best_fitness:.2f} | "
            f"Mutation: {mutation_rate:.3f}"
        )

        population = create_next_generation(
            population,
            fitness_scores,
            mutation_rate,
        )

    save_results(history)
    save_plot(history)
    comparison_history = compare_generation_champions(generation_champions, history)
    save_generation_comparison_results(comparison_history)
    save_comparison_plots(comparison_history)
    np.save(BEST_INDIVIDUAL_FILE, best_individual)
    randomization_history = analyze_randomization_effects(
        best_individual,
        RANDOMIZATION_SEEDS,
    )
    save_randomization_results(randomization_history)
    save_randomization_plot(randomization_history)

    return best_individual, history, comparison_history, randomization_history


def save_results(history):
    DATA_DIR.mkdir(exist_ok=True)

    with open(TRAINING_RESULTS_FILE, "w", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "generation",
                "best_fitness",
                "average_fitness",
                "overall_best_fitness",
                "mutation_rate",
                "crossover_rate",
            ],
        )

        writer.writeheader()

        for row in history:
            writer.writerow(row)


def save_generation_comparison_results(comparison_history):
    DATA_DIR.mkdir(exist_ok=True)

    with open(COMPARISON_RESULTS_FILE, "w", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "generation",
                "training_best_fitness",
                "training_average_fitness",
                "validation_average_reward",
                "validation_average_max_position",
                "validation_average_steps_to_goal",
                "validation_average_goal_progress",
                "validation_average_progress_bonus",
                "validation_average_goal_bonus",
                "validation_average_fitness",
                "validation_success_rate",
                "validation_successful_episodes",
                "validation_episodes",
                "successful_average_steps_to_goal",
            ],
        )

        writer.writeheader()

        for row in comparison_history:
            writer.writerow(row)


def save_randomization_results(randomization_history):
    DATA_DIR.mkdir(exist_ok=True)

    with open(RANDOMIZATION_RESULTS_FILE, "w", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "trial",
                "seed",
                "total_reward",
                "start_position",
                "final_position",
                "max_position",
                "goal_progress",
                "progress_bonus",
                "goal_bonus",
                "fitness",
                "reached_goal",
                "steps_to_goal",
            ],
        )

        writer.writeheader()

        for row in randomization_history:
            writer.writerow(row)


def save_plot(history):
    DATA_DIR.mkdir(exist_ok=True)

    generations = [row["generation"] for row in history]
    best_values = [row["best_fitness"] for row in history]
    average_values = [row["average_fitness"] for row in history]

    plt.figure()
    plt.plot(generations, best_values, label="Best fitness")
    plt.plot(generations, average_values, label="Average fitness")
    plt.xlabel("Generation")
    plt.ylabel("Fitness")
    plt.title("Genetic Algorithm Progress on MountainCar-v0")
    plt.legend()
    plt.grid(True)
    plt.savefig(FITNESS_PLOT_FILE)
    plt.close()


def save_metric_plot(
    comparison_history,
    value_key,
    ylabel,
    title,
    filename,
    value_scale=1.0,
):
    DATA_DIR.mkdir(exist_ok=True)

    generations = [row["generation"] for row in comparison_history]
    values = [row[value_key] * value_scale for row in comparison_history]

    plt.figure(figsize=(10, 5))
    plt.plot(generations, values, marker="o", markersize=3)
    plt.xlabel("Generation")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()


def save_randomization_plot(randomization_history):
    DATA_DIR.mkdir(exist_ok=True)

    trials = [row["trial"] for row in randomization_history]
    rewards = [row["total_reward"] for row in randomization_history]
    progress_bonuses = [row["progress_bonus"] for row in randomization_history]
    goal_bonuses = [row["goal_bonus"] for row in randomization_history]
    fitness_values = [row["fitness"] for row in randomization_history]

    figure, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.flatten()

    axes[0].plot(trials, rewards, marker="o")
    axes[0].set_title("Reward by Random Seed")
    axes[0].set_ylabel("Total reward")

    axes[1].plot(trials, progress_bonuses, marker="o", color="tab:green")
    axes[1].set_title("Progress Bonus by Random Seed")
    axes[1].set_ylabel("Progress bonus")

    axes[2].plot(trials, goal_bonuses, marker="o", color="tab:orange")
    axes[2].set_title("Goal Bonus by Random Seed")
    axes[2].set_ylabel("Goal bonus")

    axes[3].plot(trials, fitness_values, marker="o", color="tab:red")
    axes[3].set_title("Final Fitness by Random Seed")
    axes[3].set_ylabel("Fitness")

    for axis in axes:
        axis.set_xlabel("Trial")
        axis.grid(True)

    figure.suptitle("Randomization Effects on Reward and Fitness Bonuses")
    figure.tight_layout()
    figure.savefig(RANDOMIZATION_PLOT_FILE)
    plt.close(figure)


def save_comparison_plots(comparison_history):
    save_metric_plot(
        comparison_history,
        "validation_average_reward",
        "Average reward",
        "Validation Reward of Each Generation Champion",
        GENERATION_REWARD_PLOT_FILE,
    )
    save_metric_plot(
        comparison_history,
        "validation_average_max_position",
        "Average max position",
        "Max Position of Each Generation Champion",
        GENERATION_MAX_POSITION_PLOT_FILE,
    )
    save_metric_plot(
        comparison_history,
        "validation_average_steps_to_goal",
        "Average steps to goal",
        "Steps to Goal of Each Generation Champion",
        GENERATION_STEPS_TO_GOAL_PLOT_FILE,
    )
    save_metric_plot(
        comparison_history,
        "validation_success_rate",
        "Success rate (%)",
        "Success Rate of Each Generation Champion",
        GENERATION_SUCCESS_RATE_PLOT_FILE,
        value_scale=100,
    )


def test_individual(individual, render=False):
    result = run_policy_episode(individual, render=render)
    fitness_components = calculate_fitness_components(
        result["total_reward"],
        result["start_position"],
        result["max_position"],
        result["reached_goal"],
    )

    print("Test result")
    print(f"Total reward: {result['total_reward']}")
    print(f"Max position: {result['max_position']:.4f}")
    print(f"Goal progress: {fitness_components['goal_progress']:.4f}")
    print(f"Progress bonus: {fitness_components['progress_bonus']:.2f}")
    print(f"Goal bonus: {fitness_components['goal_bonus']:.2f}")
    print(f"Fitness: {fitness_components['fitness']:.2f}")
    print(f"Reached goal: {result['reached_goal']}")
    print(f"Steps to goal: {result['steps_to_goal']}")

    return result["total_reward"], result["max_position"]


def test_random_policy():
    env = create_environment()
    observation, info = env.reset()

    total_reward = 0
    max_position = observation[0]

    for step in range(MAX_STEPS):
        action = env.action_space.sample()
        observation, reward, terminated, truncated, info = env.step(action)

        total_reward += reward
        max_position = max(max_position, observation[0])

        if terminated or truncated:
            break

    env.close()

    print("Random policy result")
    print(f"Total reward: {total_reward}")
    print(f"Max position: {max_position:.4f}")

    return total_reward, max_position


if __name__ == "__main__":
    print("Testing random policy...")
    test_random_policy()

    print("\nTraining Genetic Algorithm...")
    best_individual, history, comparison_history, randomization_history = train()

    print("\nTesting best evolved individual...")
    test_individual(best_individual, render=RENDER_FINAL_TEST)

    print("\nSaved files:")
    print("- DATA/results.csv")
    print("- DATA/fitness_plot.png")
    print("- DATA/best_individual.npy")
    print("- DATA/generation_champions/generation_XXX.npy")
    print("- DATA/generation_comparison.csv")
    print("- DATA/generation_reward_plot.png")
    print("- DATA/generation_max_position_plot.png")
    print("- DATA/generation_steps_to_goal_plot.png")
    print("- DATA/generation_success_rate_plot.png")
    print("- DATA/randomization_effects.csv")
    print("- DATA/randomization_effect_plot.png")
