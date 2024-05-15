Here we will have the examples of using barriers, please go over [notes](NOTES.md) for more info. 

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
| Configuration | $1$ |

## Go2 squat

Go2 quadruped squating with base position is constrained by z and y coordinates:

https://github.com/domrachev03/pink/assets/26837717/6d9d3aef-2853-4ae7-a95a-7c317fd65625

| Task | Cost |
|------|------|
| Base | (50, 1) |
| Legs | 200 |
| Posture | $10^{-3}$ |

| Barrier | Gain |
|------|------|
| End-effector | $10^{2}$ |
| Configuration | $50$ |


## Dual Arm Yumi

A dual arm YuMi randomly swinging arms while avoiding self collisions:

https://github.com/domrachev03/pink/assets/28687492/97397fa1-575d-4a1f-81e9-dad85a376cf6

| Task | Cost |
|------|------|
| Left end-effector | (50,1) |
| Right end-effector | (50,1) |
| Posture | $10^{-3}$ |


| Barrier | Gain |
|------|------|
| Spheres Collision  | 100 |
| Configuration | $1$ |



