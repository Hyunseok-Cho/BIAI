# BIAI MountainCar Genetic Algorithm

This project solves the `MountainCar-v0` Gymnasium environment with a Genetic
Algorithm. Each individual is a discrete policy table, and the algorithm evolves
a population of policies over multiple generations.

## Setup

Create and activate a virtual environment:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Install the required packages:

```powershell
pip install -r requirements.txt
```

## Run

Start training with:

```powershell
python mountain_car_ga.py
```

The script trains the Genetic Algorithm, saves the best final policy, saves the
best champion from each generation, and compares those generation champions with
the same validation seeds.

## Output Files

- `results.csv`: training fitness values for each generation.
- `fitness_plot.png`: best and average training fitness over generations.
- `best_individual.npy`: best policy found during the whole training run.
- `generation_champions/generation_XXX.npy`: best policy from each generation.
- `generation_comparison.csv`: validation metrics for each generation champion.
- `generation_reward_plot.png`: average reward of each generation champion.
- `generation_max_position_plot.png`: average maximum position reached.
- `generation_steps_to_goal_plot.png`: average steps to goal, with failed runs
  counted as `MAX_STEPS`.
- `generation_success_rate_plot.png`: success rate of each generation champion.

## What to Check

Use `fitness_plot.png` to confirm that the training process improves over time.
Then use the generation comparison plots to check whether the best solution from
each generation actually performs better under the same validation conditions.
`generation_comparison.csv` contains the exact numerical values used in those
plots.
