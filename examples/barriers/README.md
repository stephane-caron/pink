Here we will have the examples of using barriers, please go over [this note](https://simeon-ned.com/blog/2024/cbf/) for more info.

# Barrier Examples

- [Arm: UR5](#arm-ur5): with joints and end effector limits
- [Quadruped: Go2](#go2-squat): Go 2 squatting with floating base position limits
- [Dual Arms: Yumi](#yumi-end-effector-self-collision-avoidance): self-collision avoidance with spheres
- [Dual Arms: Iiwa](#iiwa-whole-body-collision-avoidance): whole-body self-collision avoidance

## Arm UR5

A UR5 arm tracking a moving target while stopping in front of virtual wall:

https://github.com/domrachev03/pink/assets/28687492/f30ba7a1-98a3-44cb-ab52-23f99e42714c

| Task | Cost |
|------|------|
| End-effector | (50,1) |
| Posture | $10^{-3}$ |

| Barrier | Gain |
|------|------|
| End-effector | $10^{2}$ |

## Go2 squat

Go2 quadruped squating with base position is constrained by z and y coordinates:

https://github.com/domrachev03/pink/assets/28687492/78281f44-3676-4d4d-9619-768b951a15a2

| Task | Cost |
|------|------|
| Base | (50, 1) |
| Legs | 200 |
| Posture | $10^{-3}$ |

| Barrier | Gain |
|------|------|
| End-effector | $10^{2}$ |

## Yumi end-effector self-collision avoidance

Yumi two-armed manimpulator with constraint on minimal distance between frames, defined by end-effectors

https://github.com/domrachev03/pink/assets/28687492/f8c4bc8d-63e3-4bf7-a34f-e7ede43c0438

| Task | Cost |
|------|------|
| End-effector | (50,10) |
| Posture | $10^{-3}$ |

| Barrier | Gain |
|------|------|
| Body Spherical | $10^{2}$ |

## Iiwa whole-body collision avoidance

Two iiwas with custom collision geometry with some barely feasible tasks for end-effectors.

https://github.com/domrachev03/pink/assets/28687492/d64163b6-399f-4bbf-ac50-1135fa69c2da

| Task | Cost |
|------|------|
| End-effector | (50,10) |
| Posture | $10^{-3}$ |

| Barrier | Gain |
|------|------|
| Self Collision Avoidance | $10$ |
