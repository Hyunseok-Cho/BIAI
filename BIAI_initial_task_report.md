# BIAI Initial Task Report

#### Hyunseok Cho
#### Jakub Zając

## 1. Selection of Topic

For the initial task, our group selected the following topic:

**Solving the MountainCar-v0 environment using a Genetic Algorithm**

The purpose of this topic is to become familiar with the basic structure and operation of Genetic Algorithms. We selected the `MountainCar-v0` environment from Gymnasium because it is a relatively simple control problem, but it still requires the agent to learn a strategy instead of performing only one direct action.

In this environment, a small car is placed in a valley between two hills. The goal is to drive the car up the right hill and reach the flag. The car does not have enough power to reach the goal directly, so it must move left and right to build momentum.

The environment provides two observation values:

- position of the car
- velocity of the car

The agent can choose one of three actions:

- `0` — accelerate to the left
- `1` — do nothing
- `2` — accelerate to the right

The goal of the agent is to reach the target position as quickly as possible.

---

## 2. Preparation of a Work Plan

The work plan for the initial task is divided into several steps.

| Step | Task | Description |
|---|---|---|
| 1 | Select the environment | Choose a Gymnasium environment suitable for testing a Genetic Algorithm. |
| 2 | Understand the environment | Analyze the observation space, action space, reward system, and goal of `MountainCar-v0`. |
| 3 | Define the policy representation | Represent one candidate solution as a chromosome. In our case, one chromosome represents a policy table. |
| 4 | Create the initial population | Generate a group of random policies. |
| 5 | Evaluate individuals | Run each policy in the environment and calculate its fitness score. |
| 6 | Select parents | Choose better-performing individuals for reproduction. |
| 7 | Apply crossover | Combine parts of two parent chromosomes to create new individuals. |
| 8 | Apply mutation | Randomly change some actions in the chromosome to maintain diversity. |
| 9 | Repeat for several generations | Continue the evolutionary process to improve the population. |
| 10 | Test the best solution | Run the best evolved policy and check whether it reaches the goal. |

The planned implementation was completed in Python using Gymnasium. The program trains a population of policies over multiple generations and saves the best individual. The best policy was also tested visually using rendering, and it was able to reach the goal in the environment.

The implementation produces the following output files:

- `results.csv` — numerical results from training
- `fitness_plot.png` — graph of best and average fitness over generations
- `best_individual.npy` — saved best evolved policy

---

## 3. Basic Explanation of the Chosen Algorithm

The chosen algorithm is a **Genetic Algorithm**.

A Genetic Algorithm is an optimization method inspired by natural evolution. It works with a population of candidate solutions. Each candidate solution is evaluated using a fitness function. Better solutions have a higher chance of being selected as parents and passing their information to the next generation.

The main components of a Genetic Algorithm are:

### Population

The population is a group of candidate solutions.  
In this project, each individual in the population represents one policy for controlling the Mountain Car.

### Chromosome

A chromosome is the encoded form of one solution.  
In our implementation, the chromosome is a table of actions. The continuous observation values of the environment are divided into discrete bins, and each bin stores one action.

For example, if the position and velocity are divided into bins, each combination of position and velocity corresponds to one action:

```text
0 = accelerate left
1 = do nothing
2 = accelerate right
```

Therefore, one chromosome represents a complete decision-making strategy for the agent.

### Fitness Function

The fitness function measures how good each individual is.

In the MountainCar-v0 environment, the agent receives -1 reward at each time step. This means that reaching the goal faster gives a better result. However, many random policies fail and receive very similar rewards.

Because of this, the fitness function also includes a bonus based on the maximum position reached by the car. This helps the Genetic Algorithm distinguish between policies that completely fail and policies that move closer to the goal.

The fitness function is based on:

```total reward + maximum position bonus + goal bonus```
### Selection

Selection is the process of choosing better individuals as parents.
In our implementation, tournament selection is used. A few individuals are chosen randomly, and the best one among them becomes a parent.

### Crossover

Crossover combines two parent chromosomes to create new children.
Part of the first parent and part of the second parent are joined together. This allows good features from different parents to be combined.

### Mutation

Mutation randomly changes some values in a chromosome.
This prevents the population from becoming too similar and allows the algorithm to explore new possible solutions.

### Generations

After selection, crossover, and mutation, a new population is created.
This process is repeated for many generations. Over time, the population should improve, and better policies should appear.

## 4. Current Progress

The initial task has been completed.

So far, we have:

* selected the topic,
* selected the MountainCar-v0 environment,
* prepared a work plan,
* implemented a Genetic Algorithm in Python,
* trained policies over several generations,
* generated a fitness graph,
* saved the best evolved individual,
* tested the best policy visually using rendering.

The final evolved policy was able to reach the goal in the Mountain Car environment.

This confirms that the Genetic Algorithm was able to improve the agent's behavior over generations.