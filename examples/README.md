# Examples

The following examples include *tasks* and *limits*:

- [Arm: UR5](#arm-ur5)
- [Flying dual-arm UR3](#flying-dual-arm-ur3)
- [Humanoid: Draco 3](#humanoid-draco-3)
- [Mobile: Stretch](#mobile-stretch)
- [Wheeled biped: Upkie](#wheeled-biped-upkie)

Check out the [barriers](barriers/) sub-directory for more examples including *control barrier functions*.

## Arm: UR5

A UR5 arm tracking a moving target:

https://github.com/stephane-caron/pink/assets/1189580/d0d6aae9-326b-45fe-8cd3-013f29f7343a

| Task | Cost |
|------|------|
| End-effector | 1 |
| Posture | $10^{-3}$ |

## Flying dual-arm UR3

A pair of UR3 arms on a mobile body tracking moving targets:

https://github.com/stephane-caron/pink/assets/1189580/ef3f2571-6188-4b14-ae3f-b22428b11f5c

| Task | Cost |
|------|------|
| Mobile base | 1 |
| Left end-effector | 1 |
| Right end-effector | 1 |
| Posture | $10^{-3}$ |

## Humanoid: Draco 3

A Draco 3 humanoid moving its right hand laterally while standing. This model includes a closed kinematic chain, implemented in the example with a ``JointCouplingTask``:

https://github.com/stephane-caron/pink/assets/1189580/db6acda8-82a4-4f4d-9acf-1fc3d831e222

| Task | Cost |
|------|------|
| Left foot | (1, 1) |
| Left knee coupling | 100 |
| Posture | $10^{-1}$ |
| Right foot | (1, 1) |
| Right knee coupling | 100 |
| Right wrist | (4, 4) |
| Torso | (1, 0) |

## Mobile: Stretch

Move a Stretch RE1 with a fixed fingertip target around the origin:

https://github.com/stephane-caron/pink/assets/1189580/711c4b92-6234-41bd-945b-e6c043f6b2e6

| Task | Position cost | Orientation cost |
|------|---------------|------------------|
| Mobile base | $0.1$ | 1 |
| Fingertip | 1 | $10^{-4}$ |

## Wheeled biped: Upkie

An Upkie wheeled biped rolling without slipping:

https://github.com/user-attachments/assets/18ae0b68-21a2-44ec-af48-1d8ab4a7e658

| Task | Position cost | Orientation cost |
|------|---------------|------------------|
| Floating base | $1$ | $1$ |
| Left wheel rolling | $10$ | - |
| Right wheel rolling | $10$ | - |
| Left wheel position | $1$ | $0$ |
| Right wheel position | $1$ | $0$ |
