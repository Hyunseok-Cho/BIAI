# BIAI Initial Task Report

#### Hyunseok Cho
#### Jakub Zając

## Project Topic

**Genetic Algorithm for solving the MountainCar-v0 environment**

The selected topic for the initial task is the implementation of a Genetic Algorithm to optimize a simple control policy for the `MountainCar-v0` environment from Gymnasium.

The purpose of this task is to understand how Genetic Algorithms work and how they can be applied to an optimization problem.

---

## Environment Description

The selected environment is `MountainCar-v0`.

In this environment, a car is placed in a valley between two hills. The goal is to drive the car up the right hill and reach the flag. However, the car engine is not strong enough to reach the goal directly. Therefore, the agent has to move left and right to build momentum.

### Observation Space

The environment gives two values as observation:

- car position
- car velocity

Example observation:

```text
[position, velocity]