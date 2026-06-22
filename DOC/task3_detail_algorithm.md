# Task 3 Detailed Main Algorithm Explanation

#### Hyunseok Cho
#### Jakub Zajac

This document expands the main algorithm from the Task 3 report. It explains
each block of the Genetic Algorithm in execution order and connects each block
to the actual implementation in `IMPL/mountain_car_ga.py`.

The purpose of this document is not to redefine the whole project. Instead, it
explains how each algorithm step starts, what logic it uses, how it proceeds,
and how the step finishes before the next block starts.

## Algorithm Context

The solution is a Genetic Algorithm for `MountainCar-v0`. Each individual is a
policy table. The policy table stores one action for each discretized state of
the environment.

- Position is divided into `20` bins.
- Velocity is divided into `20` bins.
- Therefore, one policy has `20 * 20 = 400` genes.
- Each gene stores an action:
  - `0`: accelerate left
  - `1`: do nothing
  - `2`: accelerate right

The Genetic Algorithm repeats this cycle:

```text
create policies -> evaluate policies -> rank policies -> preserve best policies
-> select parents -> crossover -> mutation -> create next generation
```

The sections below explain each main algorithm block in detail.

## Step 1. Initialize

### Block Goal

The initialization block prepares all constants, random seeds, and environment
settings that the rest of the algorithm depends on.

### How The Step Starts

The program starts by defining the Gymnasium environment name and the Genetic
Algorithm parameters. These values control population size, number of
generations, mutation behavior, crossover behavior, and the discretization of
the continuous MountainCar state.

### Relevant Code

```python
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
```

The environment is created through a helper function:

```python
def create_environment(render_mode=None):
    return gym.make(ENV_NAME, render_mode=render_mode)
```

The program also fixes the base random seed:

```python
RANDOM_SEED = 42

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
```

### Detailed Logic

This step defines the rules of the experiment before any training begins.
`POPULATION_SIZE = 80` means that every generation contains 80 candidate
policies. `GENERATIONS = 60` means that the evolutionary loop will evaluate and
evolve the population 60 times.

The constants `POSITION_BINS` and `VELOCITY_BINS` define how the continuous
observation from MountainCar is converted into a table index. MountainCar gives
the agent two continuous values: position and velocity. A normal array cannot
directly use a continuous value like `-0.4831` as an index, so the state is
converted into discrete bins.

The reward and bonus constants define the scoring logic. The environment itself
gives `-1` per step, but the fitness function also adds a progress bonus and a
goal bonus. This is important because many weak policies fail at the 200-step
limit and receive a very similar raw reward. The extra fitness components help
the Genetic Algorithm distinguish weak failures from policies that move closer
to the goal.

### How The Step Finishes

Initialization finishes when all parameters are available globally and the
program is ready to create the first random population.

## Step 2. Create Population

### Block Goal

This block creates the first generation of candidate solutions. At this point,
the algorithm does not know which actions are good, so the first population is
random.

### How The Step Starts

The `train()` function begins by calling `create_population()`.

### Relevant Code

```python
def create_individual():
    chromosome_length = POSITION_BINS * VELOCITY_BINS
    return np.random.randint(0, 3, size=chromosome_length)


def create_population():
    return [create_individual() for _ in range(POPULATION_SIZE)]
```

Inside `train()`:

```python
def train():
    population = create_population()
```

### Detailed Logic

An individual is one complete policy table. Since there are 20 position bins and
20 velocity bins, the chromosome length is:

```text
20 * 20 = 400 genes
```

Each gene is randomly assigned one of three possible actions:

```text
0 = accelerate left
1 = do nothing
2 = accelerate right
```

For example, a very small policy table could look like this:

```text
[2, 2, 1, 0, 1, 2, ...]
```

This means:

- if the car is in state index `0`, use action `2`,
- if the car is in state index `1`, use action `2`,
- if the car is in state index `2`, use action `1`,
- if the car is in state index `3`, use action `0`.

The real policy table has 400 entries, not just four. The first generation
contains 80 of these random policy tables.

### Loop Walk-through

The population creation loop is written as a list comprehension:

```python
return [create_individual() for _ in range(POPULATION_SIZE)]
```

It behaves like a normal loop that repeats 80 times.

Example flow:

| Loop index | What happens | Result |
|---:|---|---|
| `i = 0` | `create_individual()` creates the first random 400-gene policy. | Stored as `population[0]`. |
| `i = 1` | `create_individual()` creates the second random policy. | Stored as `population[1]`. |
| `i = 2` | `create_individual()` creates the third random policy. | Stored as `population[2]`. |

The same process continues until `i = 79`. After that, the population has 80
individuals and the loop stops.

### How The Step Finishes

This step finishes when `population` contains 80 randomly generated policies.
The algorithm can now evaluate how well each policy behaves in the environment.

## Step 3. Evaluate Policies

### Block Goal

This block tests every individual in the current population by running it inside
the MountainCar environment.

### How The Step Starts

At the beginning of each generation, `train()` calls `evaluate_population()`.

### Relevant Code

```python
for generation in range(GENERATIONS):
    fitness_scores = evaluate_population(population)
```

The population evaluation function calls `evaluate_individual()` for every
policy:

```python
def evaluate_population(population):
    fitness_scores = []

    for individual in population:
        fitness = evaluate_individual(individual)
        fitness_scores.append(fitness)

    return fitness_scores
```

Each individual is evaluated over multiple episodes:

```python
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
```

### Detailed Logic

The algorithm evaluates all 80 policies in the population. Each policy is not
tested only once. It is tested `EPISODES_PER_INDIVIDUAL = 3` times, and the
average fitness is used.

### Loop Walk-through: Population Evaluation

The first loop in this block is:

```python
for individual in population:
    fitness = evaluate_individual(individual)
    fitness_scores.append(fitness)
```

Example flow:

| Loop index | What happens | Result |
|---:|---|---|
| `i = 0` | The algorithm evaluates `population[0]`. | The average fitness is appended as `fitness_scores[0]`. |
| `i = 1` | The algorithm evaluates `population[1]`. | The average fitness is appended as `fitness_scores[1]`. |
| `i = 2` | The algorithm evaluates `population[2]`. | The average fitness is appended as `fitness_scores[2]`. |

The loop continues until all 80 policies have been evaluated. At the end,
`fitness_scores` also has 80 values, so each policy has one matching fitness
score.

### Loop Walk-through: Episodes Per Individual

Inside `evaluate_individual()`, each policy is tested three times:

```python
for _ in range(EPISODES_PER_INDIVIDUAL):
    episode_result = run_policy_episode(individual)
    fitness = calculate_fitness(...)
    total_fitness += fitness
```

Example flow for one individual:

| Loop index | What happens | Result |
|---:|---|---|
| `i = 0` | Run the policy in episode 1 and calculate fitness. | Add episode 1 fitness to `total_fitness`. |
| `i = 1` | Run the same policy again in episode 2. | Add episode 2 fitness to `total_fitness`. |
| `i = 2` | Run the same policy again in episode 3. | Add episode 3 fitness to `total_fitness`. |

After the third episode, the function returns:

```text
average fitness = total_fitness / 3
```

This averaging is important because MountainCar can reset with different initial
conditions. If a policy is lucky in one episode, that one result should not
dominate the entire selection process. Averaging over multiple episodes gives a
more stable estimate of the policy quality.

The episode itself is executed in `run_policy_episode()`:

```python
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
```

The episode loop chooses an action, applies the action, and records the result:

```python
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
```

### Loop Walk-through: Environment Step Loop

Inside `run_policy_episode()`, the policy can interact with the environment for
up to `MAX_STEPS = 200` steps.

Example flow:

| Loop index | What happens | Result |
|---:|---|---|
| `step = 0` | The initial observation is converted to a state index, and the policy chooses an action. | The environment returns the next observation and reward `-1`. |
| `step = 1` | The new observation is converted again, and the next action is selected. | `total_reward` becomes `-2`, and `max_position` is updated if needed. |
| `step = 2` | The same action-selection process repeats for the third environment step. | `total_reward` becomes `-3`, unless the episode ended earlier. |

The loop continues until one of these conditions happens:

- the car reaches the goal, so `terminated` becomes `True`,
- the environment reaches the step limit, so `truncated` becomes `True`,
- the loop reaches `MAX_STEPS`.

The episode returns the values needed for fitness calculation:

```python
return {
    "total_reward": float(total_reward),
    "start_position": float(start_position),
    "final_position": float(final_position),
    "max_position": float(max_position),
    "reached_goal": reached_goal,
    "steps_taken": steps_taken,
    "steps_to_goal": steps_taken if reached_goal else MAX_STEPS,
}
```

### Action Selection Logic

The policy table cannot use raw continuous observations directly. The
observation is first discretized:

```python
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
```

Then the two-dimensional state bin is converted into one chromosome index:

```python
def get_action(individual, observation):
    position_index, velocity_index = discretize_observation(observation)
    state_index = position_index * VELOCITY_BINS + velocity_index
    return individual[state_index]
```

Example:

```text
position_index = 8
velocity_index = 12
state_index = 8 * 20 + 12 = 172
```

If `individual[172] = 2`, the policy chooses action `2`, which means
`accelerate right`.

### How The Step Finishes

This block finishes when every individual has been evaluated and
`fitness_scores` contains one average fitness value for each policy in the
population.

## Step 4. Calculate Fitness

### Block Goal

This block converts the raw episode result into one numerical fitness score.
The Genetic Algorithm uses this score to rank policies.

### How The Step Starts

Fitness calculation starts after one episode has produced:

- total reward,
- start position,
- maximum position reached,
- whether the goal was reached.

These values are passed into `calculate_fitness()`.

### Relevant Code

```python
def calculate_goal_progress(start_position, max_position):
    distance_to_goal = GOAL_POSITION - start_position

    if distance_to_goal <= 0:
        return 1.0

    progress = (max_position - start_position) / distance_to_goal
    return float(np.clip(progress, 0.0, 1.0))
```

```python
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
```

```python
def calculate_fitness(total_reward, start_position, max_position, reached_goal):
    components = calculate_fitness_components(
        total_reward,
        start_position,
        max_position,
        reached_goal,
    )

    return components["fitness"]
```

### Detailed Logic

The final fitness formula is:

```text
fitness = total_reward + progress_bonus + goal_bonus
```

The first component is `total_reward`. In MountainCar, the environment gives
`-1` for each step. The code accumulates this inside the episode loop:

```python
total_reward += reward
```

Since the environment reward is `-1` per step, the total reward is:

```text
total_reward = -1 * number_of_steps
```

Examples:

| Episode result | Steps | Total reward |
|---|---:|---:|
| Goal reached quickly | 110 | -110 |
| Goal reached slowly | 180 | -180 |
| Goal not reached | 200 | -200 |

The second component is `progress_bonus`. It measures how far the car moved
toward the goal compared with the distance it needed to travel from the start
position.

```text
goal_progress = (max_position - start_position) / (goal_position - start_position)
progress_bonus = 500 * goal_progress
```

The third component is `goal_bonus`. It is added only if the car actually
reaches the goal:

```text
goal_bonus = 200 if reached_goal else 0
```

Example:

```text
total_reward = -150
goal_progress = 1.0
progress_bonus = 500
goal_bonus = 200

fitness = -150 + 500 + 200 = 550
```

Another example for a failed but improving policy:

```text
total_reward = -200
goal_progress = 0.70
progress_bonus = 350
goal_bonus = 0

fitness = -200 + 350 + 0 = 150
```

This is why the progress bonus matters. Two failed policies may both receive
`-200` reward, but the policy that reaches a higher maximum position receives a
higher fitness score.

### How The Step Finishes

This block finishes when each policy receives one numerical fitness value. That
fitness value is stored in `fitness_scores` and used by the ranking and
selection blocks.

## Step 5. Save Generation Champion

### Block Goal

This block identifies the best policy in the current generation and saves it as
that generation's champion.

### How The Step Starts

After `evaluate_population()` returns all fitness scores, the algorithm finds
the highest score in the current generation.

### Relevant Code

```python
generation_best = max(fitness_scores)
generation_average = sum(fitness_scores) / len(fitness_scores)
generation_best_index = fitness_scores.index(generation_best)
generation_best_individual = population[generation_best_index].copy()
mutation_rate = get_mutation_rate(generation)

generation_champions.append(generation_best_individual)
save_generation_champion(generation_best_individual, generation + 1)
```

The champion is saved to disk:

```python
def save_generation_champion(individual, generation):
    CHAMPIONS_DIR.mkdir(parents=True, exist_ok=True)
    file_path = CHAMPIONS_DIR / f"generation_{generation:03d}.npy"
    np.save(file_path, individual)
```

### Detailed Logic

`generation_best` stores the highest fitness value from the current generation.
`generation_best_index` finds which individual produced that score. The
algorithm then copies that policy into `generation_best_individual`.

The copy operation is important because the population will later be used to
create children for the next generation. Saving a copy protects the champion
from later changes.

The champion is stored in two ways:

1. In memory, inside `generation_champions`.
2. On disk, inside `DATA/generation_champions/generation_XXX.npy`.

For example, the champion from generation 7 is saved as:

```text
DATA/generation_champions/generation_007.npy
```

### How The Step Finishes

This block finishes when the best policy of the current generation has been
copied and saved. The algorithm can now compare it against the best policy found
across all previous generations.

## Step 6. Update Overall Best

### Block Goal

This block keeps track of the best policy found during the whole training run,
not only the current generation.

### How The Step Starts

At the beginning of training, no best policy exists yet:

```python
best_individual = None
best_fitness = -float("inf")
```

After each generation champion is found, the algorithm checks whether it is
better than the current overall best.

### Relevant Code

```python
if generation_best > best_fitness:
    best_fitness = generation_best
    best_individual = generation_best_individual.copy()
```

The history row records both current generation performance and overall best
performance:

```python
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
```

### Detailed Logic

The current generation champion is not always the best policy of the whole
training run. A later generation can sometimes have a lower champion score
because crossover and mutation introduce randomness.

For that reason, the algorithm uses two different values:

- `generation_best`: the best score in the current generation.
- `best_fitness`: the best score seen across all generations so far.

Example:

| Generation | Generation best | Overall best after update |
|---:|---:|---:|
| 1 | 120 | 120 |
| 2 | 180 | 180 |
| 3 | 160 | 180 |
| 4 | 240 | 240 |

In generation 3, the generation champion is worse than the previous overall
best, so `best_individual` is not replaced.

### How The Step Finishes

This block finishes when `best_individual` and `best_fitness` correctly reflect
the best policy found so far. The generation statistics are also stored in
`history`.

## Step 7. Preserve Elites

### Block Goal

This block copies the strongest policies directly into the next generation.
This is called elitism.

### How The Step Starts

After the current generation has been evaluated and recorded, the algorithm
starts building the next generation by calling `create_next_generation()`.

### Relevant Code

```python
def create_next_generation(population, fitness_scores, mutation_rate):
    sorted_indices = np.argsort(fitness_scores)[::-1]

    next_generation = []

    for i in range(ELITE_SIZE):
        elite_index = sorted_indices[i]
        next_generation.append(population[elite_index].copy())
```

### Detailed Logic

`np.argsort(fitness_scores)` returns the indices that would sort the population
from lowest fitness to highest fitness. The `[::-1]` reverses the order, so the
best fitness comes first.

`ELITE_SIZE = 6`, so the top 6 policies are copied directly into
`next_generation`.

### Loop Walk-through

The elite loop is:

```python
for i in range(ELITE_SIZE):
    elite_index = sorted_indices[i]
    next_generation.append(population[elite_index].copy())
```

Example flow:

| Loop index | What happens | Result |
|---:|---|---|
| `i = 0` | Read `sorted_indices[0]`, which is the best policy index. | Copy the best policy into `next_generation[0]`. |
| `i = 1` | Read `sorted_indices[1]`, which is the second-best policy index. | Copy the second-best policy into `next_generation[1]`. |
| `i = 2` | Read `sorted_indices[2]`, which is the third-best policy index. | Copy the third-best policy into `next_generation[2]`. |

The loop continues until `i = 5`. At that point, 6 elite policies have been
copied without crossover or mutation.

This prevents the algorithm from losing strong policies. Without elitism, a
good policy could disappear if it is not selected as a parent, or if mutation
damages its children.

Example:

```text
fitness_scores = [100, 250, 80, 300, 220]
sorted_indices = [3, 1, 4, 0, 2]
```

If `ELITE_SIZE = 2`, the algorithm copies individuals `3` and `1` directly into
the next generation.

### How The Step Finishes

This block finishes when the top elite policies have been copied into
`next_generation`. The remaining population slots will be filled with children
created by selection, crossover, and mutation.

## Step 8. Select Parents

### Block Goal

This block selects parent policies that will be used to create new child
policies.

### How The Step Starts

After elites are copied, the algorithm repeatedly selects two parents while the
next generation is still smaller than `POPULATION_SIZE`.

### Relevant Code

```python
def tournament_selection(population, fitness_scores):
    selected_indices = random.sample(range(len(population)), TOURNAMENT_SIZE)
    best_index = selected_indices[0]

    for index in selected_indices:
        if fitness_scores[index] > fitness_scores[best_index]:
            best_index = index

    return population[best_index].copy()
```

Parent selection is used inside `create_next_generation()`:

```python
parent1 = tournament_selection(population, fitness_scores)
parent2 = tournament_selection(population, fitness_scores)
```

### Detailed Logic

The algorithm uses tournament selection. Instead of always choosing the top two
policies, it randomly selects a small group of candidates and picks the best one
inside that group.

The tournament size is:

```python
TOURNAMENT_SIZE = 5
```

Example:

```text
Randomly selected candidates: [12, 44, 8, 51, 3]
Their fitness values:         [130, 220, 170, 90, 300]
Selected parent index:        3
```

Candidate `3` wins because it has the highest fitness among the sampled
candidates.

Tournament selection creates a useful balance:

- Better policies have a higher chance of becoming parents.
- Weaker policies can still sometimes be selected if they are sampled into an
  easier tournament.

This keeps selection pressure while preserving some diversity.

### Loop Walk-through

The tournament comparison loop is:

```python
for index in selected_indices:
    if fitness_scores[index] > fitness_scores[best_index]:
        best_index = index
```

Example with five sampled candidates:

```text
selected_indices = [12, 44, 8, 51, 3]
fitness values   = [130, 220, 170, 90, 300]
```

| Loop index | Candidate checked | What happens |
|---:|---:|---|
| `i = 0` | `index = 12` | This is the initial best candidate, so `best_index = 12`. |
| `i = 1` | `index = 44` | Fitness `220` is higher than `130`, so `best_index` becomes `44`. |
| `i = 2` | `index = 8` | Fitness `170` is lower than `220`, so `best_index` stays `44`. |

The loop continues for `index = 51` and `index = 3`. At the end, candidate `3`
wins because its fitness is `300`, the highest value in the tournament.

### How The Step Finishes

This block finishes when two parent policies have been selected and copied. The
algorithm then passes those parents into the crossover block.

## Step 9. Apply Crossover

### Block Goal

This block combines two parent policies to create two child policies.

### How The Step Starts

After two parents are selected, the algorithm calls `crossover(parent1,
parent2)`.

### Relevant Code

```python
def crossover(parent1, parent2):
    if random.random() > CROSSOVER_RATE:
        return parent1.copy(), parent2.copy()

    chromosome_length = len(parent1)
    crossover_point = random.randint(1, chromosome_length - 2)

    child1 = np.concatenate((parent1[:crossover_point], parent2[crossover_point:]))
    child2 = np.concatenate((parent2[:crossover_point], parent1[crossover_point:]))

    return child1, child2
```

### Detailed Logic

The crossover rate is:

```python
CROSSOVER_RATE = 0.85
```

This means crossover is applied with 85% probability. If crossover is not
applied, both parents are copied directly as children.

When crossover is applied, the algorithm chooses one random crossover point
inside the chromosome. The first child receives the first part of parent 1 and
the second part of parent 2. The second child receives the first part of parent
2 and the second part of parent 1.

Example:

```text
parent1 = [2, 2, 2, 2 | 0, 0]
parent2 = [1, 1, 1, 1 | 2, 2]

child1  = [2, 2, 2, 2 | 2, 2]
child2  = [1, 1, 1, 1 | 0, 0]
```

In the actual project, the chromosome has 400 genes, so the crossover point can
split a much larger policy table.

### How The Step Finishes

This block finishes when two child policies are produced. These children are not
added to the next generation immediately; they first pass through mutation.

## Step 10. Apply Mutation

### Block Goal

This block randomly changes some actions in the child policies. Mutation allows
the Genetic Algorithm to explore new behavior that may not exist in the current
parents.

### How The Step Starts

Before creating children in each generation, the algorithm calculates the
current mutation rate.

### Relevant Code

```python
def get_mutation_rate(generation):
    if GENERATIONS <= 1:
        return FINAL_MUTATION_RATE

    generation_progress = generation / (GENERATIONS - 1)
    return INITIAL_MUTATION_RATE + (
        FINAL_MUTATION_RATE - INITIAL_MUTATION_RATE
    ) * generation_progress
```

Each child is then mutated:

```python
def mutate(individual, mutation_rate):
    for i in range(len(individual)):
        if random.random() < mutation_rate:
            individual[i] = random.randint(0, 2)

    return individual
```

Inside `create_next_generation()`:

```python
child1 = mutate(child1, mutation_rate)
child2 = mutate(child2, mutation_rate)
```

### Detailed Logic

The mutation rate is adaptive. It starts higher and gradually decreases:

```text
initial mutation rate = 0.05
final mutation rate   = 0.01
```

At the beginning of training, more mutation helps exploration. The algorithm is
still searching for useful behavior, so it is helpful to try more random action
changes.

Near the end of training, less mutation helps preserve strong policies. The
algorithm has already discovered better patterns, so it should avoid changing
too many useful genes.

Mutation is applied gene by gene. For every gene in a child policy:

```text
if random value < current mutation rate:
    replace this action with random action 0, 1, or 2
```

### Loop Walk-through

The mutation loop is:

```python
for i in range(len(individual)):
    if random.random() < mutation_rate:
        individual[i] = random.randint(0, 2)
```

Example with `mutation_rate = 0.05`:

| Loop index | What happens | Result |
|---:|---|---|
| `i = 0` | Generate a random value for gene `0`. Suppose it is `0.72`. | `0.72 > 0.05`, so gene `0` is not changed. |
| `i = 1` | Generate a random value for gene `1`. Suppose it is `0.02`. | `0.02 < 0.05`, so gene `1` is replaced with a random action. |
| `i = 2` | Generate a random value for gene `2`. Suppose it is `0.31`. | `0.31 > 0.05`, so gene `2` is not changed. |

The loop continues until all 400 genes have been checked. Most genes stay the
same, but a small number of genes are randomly changed.

Example with mutation rate `0.05`:

```text
400 genes * 0.05 = about 20 changed genes
```

Example with mutation rate `0.01`:

```text
400 genes * 0.01 = about 4 changed genes
```

### How The Step Finishes

This block finishes when both children have been checked for mutation. The
mutated children can now be added to the next generation.

## Step 11. Create Next Generation

### Block Goal

This block fills the next generation until it again contains 80 policies.

### How The Step Starts

The block starts after elites have already been copied. The remaining slots are
filled by repeatedly selecting parents, applying crossover, applying mutation,
and appending children.

### Relevant Code

```python
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
```

### Detailed Logic

At this stage, `next_generation` already contains the 6 elites. The algorithm
needs to create the remaining 74 policies.

Each loop creates up to two children:

1. Select parent 1 by tournament selection.
2. Select parent 2 by tournament selection.
3. Apply single-point crossover.
4. Apply adaptive mutation to both children.
5. Add child 1.
6. Add child 2 only if there is still space.

The final `if len(next_generation) < POPULATION_SIZE` check prevents the
population from becoming larger than 80. This matters because children are
created in pairs, but the number of remaining slots may be odd.

### Loop Walk-through

At the start of this block, 6 elite policies are already stored in
`next_generation`.

Example flow:

| While iteration | Start length | What happens | End length |
|---:|---:|---|---:|
| `i = 0` | `6` | Select two parents, create two children, mutate them, append both. | `8` |
| `i = 1` | `8` | Repeat parent selection, crossover, mutation, and append two children. | `10` |
| `i = 2` | `10` | Repeat the same reproduction process again. | `12` |

The loop continues until the length reaches `80`. Then the condition
`len(next_generation) < POPULATION_SIZE` becomes false and the function returns
the completed next generation.

### How The Step Finishes

This block finishes when `next_generation` contains exactly
`POPULATION_SIZE = 80` policies. The function returns the completed next
generation to the training loop.

## Step 12. Repeat

### Block Goal

This block repeats the evolutionary process for the configured number of
generations.

### How The Step Starts

The repetition is controlled by the main generation loop in `train()`.

### Relevant Code

```python
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

    population = create_next_generation(
        population,
        fitness_scores,
        mutation_rate,
    )
```

### Detailed Logic

The loop runs `GENERATIONS = 60` times. Each loop iteration is one generation.

The order inside one generation is:

1. Evaluate the current population.
2. Find the best and average fitness.
3. Save the generation champion.
4. Update the overall best individual if needed.
5. Record generation statistics.
6. Create the next generation.

### Loop Walk-through

The main generation loop is:

```python
for generation in range(GENERATIONS):
    ...
```

Because `GENERATIONS = 60`, the loop variable goes from `0` to `59`. The report
and output files display generation numbers as `generation + 1`.

Example flow:

| Loop value | Displayed generation | What happens |
|---:|---:|---|
| `generation = 0` | Generation `1` | Evaluate the first random population, save `generation_001.npy`, and create generation 2. |
| `generation = 1` | Generation `2` | Evaluate the new population created from generation 1, save `generation_002.npy`, and create generation 3. |
| `generation = 2` | Generation `3` | Evaluate the next evolved population, save `generation_003.npy`, and create generation 4. |

The same cycle continues until `generation = 59`, which is displayed as
generation `60`.

This means that every generation is evaluated before it is used to produce the
next generation. The algorithm does not mutate or crossover policies before
they receive a fitness score. Fitness always comes first, and reproduction uses
the scores from the already evaluated population.

Implementation note: the current code calls `create_next_generation()` at the
bottom of every loop iteration, including the final iteration. The extra
population created after the last evaluated generation is not used for saved
results because `best_individual`, `history`, and generation champions are
already recorded before that call.

### How The Step Finishes

This block finishes when all 60 generations have been evaluated. At that point,
the algorithm has a complete training history, saved generation champions, and a
best individual found across the whole run.

## Step 13. Save Final Best Solution

### Block Goal

This block saves the training outputs after the evolutionary loop finishes.

### How The Step Starts

After the `for generation in range(GENERATIONS)` loop ends, the algorithm calls
output functions and saves the best policy.

### Relevant Code

```python
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
```

The best policy is saved here:

```python
np.save(BEST_INDIVIDUAL_FILE, best_individual)
```

### Detailed Logic

The final best solution is the best policy found across all generations, not
only the champion of the final generation. This is why `best_individual` is
updated throughout training and saved after the loop.

The output files serve different purposes:

- `DATA/results.csv` stores training fitness values for each generation.
- `DATA/fitness_plot.png` visualizes best and average training fitness.
- `DATA/generation_champions/generation_XXX.npy` stores each generation champion.
- `DATA/generation_comparison.csv` compares all generation champions using fixed
  validation seeds.
- `DATA/generation_*_plot.png` files visualize reward, max position, steps to
  goal, and success rate for generation champions.
- `DATA/best_individual.npy` stores the final best solution.
- `DATA/randomization_effects.csv` and `DATA/randomization_effect_plot.png`
  show how the final best policy behaves under different random seeds.

### Loop Walk-through: Saving Rows

Several output functions write rows to CSV files. For example, `save_results()`
writes one row per generation:

```python
for row in history:
    writer.writerow(row)
```

Example flow:

| Loop index | Row written | Meaning |
|---:|---|---|
| `i = 0` | `history[0]` | Save the metrics for generation 1. |
| `i = 1` | `history[1]` | Save the metrics for generation 2. |
| `i = 2` | `history[2]` | Save the metrics for generation 3. |

The same pattern is used for comparison and randomization CSV outputs. Each row
is written once, so the saved file directly matches the in-memory history list.

### How The Step Finishes

This block finishes when the final policy and all analysis files have been
written to disk. The algorithm then returns the main results from `train()`.

```python
return best_individual, history, comparison_history, randomization_history
```

## Step 14. Validate And Visualize

### Block Goal

This block checks whether the saved policies behave well under controlled
validation conditions and allows the best solution to be viewed graphically.

### How The Step Starts

Validation starts after training finishes. The code compares generation
champions using the same validation seeds.

### Relevant Code: Generation Champion Comparison

```python
def compare_generation_champions(generation_champions, history):
    comparison_history = []

    print("\nComparing each generation champion...")

    for index, champion in enumerate(generation_champions):
        generation = index + 1
        validation_summary = evaluate_champion(champion, VALIDATION_SEEDS)
```

Each champion is evaluated with the same validation seeds:

```python
VALIDATION_EPISODES = 10
VALIDATION_SEEDS = [
    RANDOM_SEED + 1000 + seed_index for seed_index in range(VALIDATION_EPISODES)
]
```

The code stores validation metrics:

```python
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
    "validation_average_fitness": validation_summary["average_fitness"],
    "validation_success_rate": validation_summary["success_rate"],
}
```

### Loop Walk-through: Generation Champion Comparison

The comparison loop is:

```python
for index, champion in enumerate(generation_champions):
    generation = index + 1
    validation_summary = evaluate_champion(champion, VALIDATION_SEEDS)
```

Example flow:

| Loop index | Champion checked | What happens |
|---:|---|---|
| `index = 0` | `generation_champions[0]` | Validate the champion from generation 1 using the fixed validation seeds. |
| `index = 1` | `generation_champions[1]` | Validate the champion from generation 2 using the same validation seeds. |
| `index = 2` | `generation_champions[2]` | Validate the champion from generation 3 using the same validation seeds. |

The loop continues until every saved generation champion has been evaluated.
Because all champions use the same validation seeds, their results can be
compared more fairly.

### Relevant Code: Randomization Effect Check

```python
RANDOMIZATION_TRIALS = 20
RANDOMIZATION_SEEDS = [
    RANDOM_SEED + 2000 + seed_index for seed_index in range(RANDOMIZATION_TRIALS)
]
```

```python
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
```

### Loop Walk-through: Seed Generation

The validation seed list uses this formula:

```python
RANDOM_SEED + 1000 + seed_index
```

With `RANDOM_SEED = 42`, the first three validation seeds are:

| Loop index | Calculation | Seed |
|---:|---|---:|
| `seed_index = 0` | `42 + 1000 + 0` | `1042` |
| `seed_index = 1` | `42 + 1000 + 1` | `1043` |
| `seed_index = 2` | `42 + 1000 + 2` | `1044` |

The randomization seed list uses this formula:

```python
RANDOM_SEED + 2000 + seed_index
```

With `RANDOM_SEED = 42`, the first three randomization seeds are:

| Loop index | Calculation | Seed |
|---:|---|---:|
| `seed_index = 0` | `42 + 2000 + 0` | `2042` |
| `seed_index = 1` | `42 + 2000 + 1` | `2043` |
| `seed_index = 2` | `42 + 2000 + 2` | `2044` |

### Loop Walk-through: Randomization Trials

The randomization analysis loop is:

```python
for trial_index, seed in enumerate(seeds, start=1):
    episode_result = run_policy_episode(individual, seed=seed)
    fitness_components = calculate_fitness_components(...)
```

Example flow:

| Loop index | Seed used | What happens |
|---:|---:|---|
| `trial_index = 1` | `2042` | Run the final best policy once with seed `2042`, then store reward, bonuses, and fitness. |
| `trial_index = 2` | `2043` | Run the same policy with seed `2043`, then store the same metrics. |
| `trial_index = 3` | `2044` | Run the same policy with seed `2044`, then store the same metrics. |

The loop continues until 20 randomization trials have been recorded.

### Relevant Code: Visual Program

The separate visualization script loads the saved best policy:

```python
def load_policy(policy_path):
    path = Path(policy_path)

    if not path.exists():
        raise FileNotFoundError(
            f"Policy file '{policy_path}' was not found. "
            "Run IMPL/mountain_car_ga.py first to create DATA/best_individual.npy."
        )

    return np.load(path)
```

Then it runs that policy in either a live window or screenshot mode:

```python
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
```

### Detailed Logic

Validation has two main roles.

First, generation champion comparison checks how each generation's best policy
performs under the same validation conditions. During training, each generation
is evaluated under random episode conditions. For comparison, using fixed
validation seeds makes the result more consistent. This allows us to compare
generation 1, generation 2, generation 3, and so on using the same set of
starting conditions.

Second, randomization effect analysis checks the final best solution under
different seeds. A seed changes the environment reset randomness, such as the
initial state of the episode. If the final solution performs well across many
seeds, it is more stable. If it only performs well on one seed, it may be less
general.

The visualization script serves a different purpose. It does not retrain the
Genetic Algorithm. Instead, it loads `DATA/best_individual.npy` and runs the
saved policy directly. This allows us to demonstrate the final learned behavior
with a graphical MountainCar window or save a screenshot for the report.

Example commands:

```powershell
python IMPL/visualize_best_solution.py
```

```powershell
python IMPL/visualize_best_solution.py --mode screenshot --seed 2042 --output DATA/best_solution_screenshot.png
```

### How The Step Finishes

This block finishes when validation files, comparison plots, randomization
plots, and optional visualization outputs have been produced. At this point, the
algorithm has not only trained a policy, but also generated evidence for how the
policy performs across generations and under different random seeds.

## Summary Of The Complete Flow

The full process can be summarized as follows:

1. Define the environment, GA parameters, bins, rewards, and random seeds.
2. Create 80 random policy tables.
3. Run every policy in MountainCar.
4. Calculate fitness using reward, progress bonus, and goal bonus.
5. Save the best policy from the current generation.
6. Update the overall best policy if the current champion is stronger.
7. Copy elite policies directly to the next generation.
8. Select parents using tournament selection.
9. Combine parents with single-point crossover.
10. Mutate child policies with an adaptive mutation rate.
11. Fill the next generation back to 80 policies.
12. Repeat the process for 60 generations.
13. Save the final best policy and result files.
14. Validate, compare, plot, and visualize the learned solution.

The key idea is that the algorithm combines preservation and exploration.
Elitism preserves the strongest policies, while tournament selection, crossover,
and mutation explore new policy combinations. The improved fitness function
guides the search toward policies that move farther toward the goal, reach the
goal more reliably, and use fewer steps.
