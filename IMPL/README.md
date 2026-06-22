# BIAI MountainCar Genetic Algorithm

This project solves the `MountainCar-v0` Gymnasium environment with a Genetic
Algorithm. Each individual is a discrete policy table, and the algorithm evolves
a population of policies over multiple generations.

## Environment Reward And Actions

The agent chooses one action at every time step:

- `0`: accelerate left
- `1`: do nothing
- `2`: accelerate right

The environment gives a reward of `-1` for each action/time step. This means a
policy receives a better total reward when it reaches the goal in fewer steps.
Many unsuccessful policies still receive a similar reward near `-200`, so the
fitness function also gives a stronger bonus for moving farther toward the goal.

## Fitness Improvements

The current fitness function is:

```text
fitness = total_reward + progress_bonus + goal_bonus
```

`progress_bonus` measures how much of the distance from the start position to
the goal was covered by the best position reached in the episode. This gives
more importance to policies that move the car farther toward the goal, even
before they can solve the environment completely.

The Genetic Algorithm also uses an adaptive mutation rate. Mutation starts high
to support exploration and gradually decreases to preserve good solutions in
later generations. The crossover rate was adjusted to balance recombination with
exploration.

## Setup

Create and activate a virtual environment:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Install the required packages from the project root:

```powershell
pip install -r IMPL/requirements.txt
```

## Run

Start training with:

```powershell
python IMPL/mountain_car_ga.py
```

The script trains the Genetic Algorithm, saves the best final policy, saves the
best champion from each generation, and compares those generation champions with
the same validation seeds. All generated result files are saved in `DATA/`.

## Visualize Best Solution

After training has created `DATA/best_individual.npy`, run the saved best policy
in a live MountainCar window with:

```powershell
python IMPL/visualize_best_solution.py
```

Save a screenshot of the final rendered frame with:

```powershell
python IMPL/visualize_best_solution.py --mode screenshot --seed 2042 --output DATA/best_solution_screenshot.png
```

## Output Files

- `DATA/results.csv`: training fitness values for each generation.
- `DATA/fitness_plot.png`: best and average training fitness over generations.
- `DATA/best_individual.npy`: best policy found during the whole training run.
- `DATA/generation_champions/generation_XXX.npy`: best policy from each generation.
- `DATA/generation_comparison.csv`: validation metrics for each generation champion.
- `DATA/generation_reward_plot.png`: average reward of each generation champion.
- `DATA/generation_max_position_plot.png`: average maximum position reached.
- `DATA/generation_steps_to_goal_plot.png`: average steps to goal, with failed runs
  counted as `MAX_STEPS`.
- `DATA/generation_success_rate_plot.png`: success rate of each generation champion.
- `DATA/randomization_effects.csv`: reward, progress bonus, goal bonus, and fitness
  for the best solution under different random seeds.
- `DATA/randomization_effect_plot.png`: plot showing how random seeds affect reward
  and fitness bonus components.
- `DATA/best_solution_screenshot.png`: screenshot generated from the saved best
  policy visualization.

## What to Check

Use `DATA/fitness_plot.png` to confirm that the training process improves over
time. Then use the generation comparison plots to check whether the best
solution from each generation actually performs better under the same validation
conditions. `DATA/generation_comparison.csv` contains the exact numerical values
used in those plots. Use `DATA/randomization_effect_plot.png` to check whether
the final solution is stable across different random seeds.
