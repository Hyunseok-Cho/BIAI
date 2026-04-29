import gymnasium as gym
import numpy as np
import random
import csv
import matplotlib.pyplot as plt


ENV_NAME = "MountainCar-v0"

POPULATION_SIZE = 80
GENERATIONS = 60
ELITE_SIZE = 6
TOURNAMENT_SIZE = 5
MUTATION_RATE = 0.02
CROSSOVER_RATE = 0.9

POSITION_BINS = 20
VELOCITY_BINS = 20

EPISODES_PER_INDIVIDUAL = 3
MAX_STEPS = 200

RANDOM_SEED = 42


random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)


def create_environment(render_mode=None):
    return gym.make(ENV_NAME, render_mode=render_mode)


def discretize_observation(observation):
    position = observation[0]
    velocity = observation[1]

    position_min = -1.2
    position_max = 0.6

    velocity_min = -0.07
    velocity_max = 0.07

    position_index = int(
        (position - position_min) / (position_max - position_min) * POSITION_BINS
    )

    velocity_index = int(
        (velocity - velocity_min) / (velocity_max - velocity_min) * VELOCITY_BINS
    )

    position_index = np.clip(position_index, 0, POSITION_BINS - 1)
    velocity_index = np.clip(velocity_index, 0, VELOCITY_BINS - 1)

    return position_index, velocity_index


def get_action(individual, observation):
    position_index, velocity_index = discretize_observation(observation)
    state_index = position_index * VELOCITY_BINS + velocity_index
    return individual[state_index]


def create_individual():
    chromosome_length = POSITION_BINS * VELOCITY_BINS
    return np.random.randint(0, 3, size=chromosome_length)


def create_population():
    return [create_individual() for _ in range(POPULATION_SIZE)]


def evaluate_individual(individual):
    total_fitness = 0.0

    for _ in range(EPISODES_PER_INDIVIDUAL):
        env = create_environment()
        observation, info = env.reset()

        total_reward = 0
        max_position = observation[0]
        reached_goal = False

        for _ in range(MAX_STEPS):
            action = get_action(individual, observation)
            observation, reward, terminated, truncated, info = env.step(action)

            total_reward += reward
            max_position = max(max_position, observation[0])

            if terminated:
                reached_goal = True
                break

            if truncated:
                break

        env.close()

        fitness = total_reward + 100 * max_position

        if reached_goal:
            fitness += 100

        total_fitness += fitness

    return total_fitness / EPISODES_PER_INDIVIDUAL


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


def mutate(individual):
    for i in range(len(individual)):
        if random.random() < MUTATION_RATE:
            individual[i] = random.randint(0, 2)

    return individual


def create_next_generation(population, fitness_scores):
    sorted_indices = np.argsort(fitness_scores)[::-1]

    next_generation = []

    for i in range(ELITE_SIZE):
        elite_index = sorted_indices[i]
        next_generation.append(population[elite_index].copy())

    while len(next_generation) < POPULATION_SIZE:
        parent1 = tournament_selection(population, fitness_scores)
        parent2 = tournament_selection(population, fitness_scores)

        child1, child2 = crossover(parent1, parent2)

        child1 = mutate(child1)
        child2 = mutate(child2)

        next_generation.append(child1)

        if len(next_generation) < POPULATION_SIZE:
            next_generation.append(child2)

    return next_generation


def train():
    population = create_population()

    history = []

    best_individual = None
    best_fitness = -float("inf")

    for generation in range(GENERATIONS):
        fitness_scores = evaluate_population(population)

        generation_best = max(fitness_scores)
        generation_average = sum(fitness_scores) / len(fitness_scores)
        generation_best_index = fitness_scores.index(generation_best)

        if generation_best > best_fitness:
            best_fitness = generation_best
            best_individual = population[generation_best_index].copy()

        history.append(
            {
                "generation": generation + 1,
                "best_fitness": generation_best,
                "average_fitness": generation_average,
                "overall_best_fitness": best_fitness,
            }
        )

        print(
            f"Generation {generation + 1:03d} | "
            f"Best: {generation_best:.2f} | "
            f"Average: {generation_average:.2f} | "
            f"Overall best: {best_fitness:.2f}"
        )

        population = create_next_generation(population, fitness_scores)

    save_results(history)
    save_plot(history)
    np.save("best_individual.npy", best_individual)

    return best_individual, history


def save_results(history):
    with open("results.csv", "w", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "generation",
                "best_fitness",
                "average_fitness",
                "overall_best_fitness",
            ],
        )

        writer.writeheader()

        for row in history:
            writer.writerow(row)


def save_plot(history):
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
    plt.savefig("fitness_plot.png")
    plt.close()


def test_individual(individual, render=False):
    if render:
        env = create_environment(render_mode="human")
    else:
        env = create_environment()

    observation, info = env.reset()

    total_reward = 0
    max_position = observation[0]

    for step in range(MAX_STEPS):
        action = get_action(individual, observation)
        observation, reward, terminated, truncated, info = env.step(action)

        total_reward += reward
        max_position = max(max_position, observation[0])

        if terminated or truncated:
            break

    env.close()

    print("Test result")
    print(f"Total reward: {total_reward}")
    print(f"Max position: {max_position:.4f}")

    return total_reward, max_position


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
    best_individual, history = train()

    print("\nTesting best evolved individual...")
    test_individual(best_individual, render=False)

    print("\nSaved files:")
    print("- results.csv")
    print("- fitness_plot.png")
    print("- best_individual.npy")