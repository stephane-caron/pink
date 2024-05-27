Here we will have the examples of using barriers, please go over [notes](https://simeon-ned.com/blog/2024/cbf/) for more info. 

# Barrier Examples


- [Arm: UR5](#arm-ur5): with joints and end effector limits
- [Quadruped: Go2](#go2-squat): Go 2 squatting with floating base position limits
- [Dual Arms: Yumi](#dual-arm-yumi): self collisions with spheres

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

https://github.com/domrachev03/pink/assets/26837717/701bdfbe-0dba-4f9d-80e2-c018475f38d6



| Task | Cost |
|------|------|
| Base | (50, 1) |
| Legs | 200 |
| Posture | $10^{-3}$ |

| Barrier | Gain |
|------|------|
| End-effector | $10^{2}$ |





