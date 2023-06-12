# Examples

## Flying dual-arm UR3

A pair of UR3 arms on a mobile body tracking moving targets:

https://github.com/tasts-robots/pink/assets/1189580/60e904e2-3e8a-45f0-b2ad-4ac691f1551a

| Task | Cost |
|------|------|
| Mobile base | 1 |
| Left end-effector | 1 |
| Right end-effector | 1 |
| Posture | $10^{-3}$ |

## Mobile Stretch

Move a Stretch RE1 with a fixed fingertip target around the origin:

https://user-images.githubusercontent.com/1189580/231776286-7dbba695-1e34-408c-936e-80122b7f1148.mp4

| Task | Position cost | Orientation cost |
|------|---------------|------------------|
| Mobile base | $0.1$ | 1 |
| Fingertip | 1 | $10^{-4}$ |

## UR5 arm
