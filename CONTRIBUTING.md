# 👷 Contributing

If any of these sound interesting, open an [issue](https://github.com/tasts-robots/pink/issues) and say you're on it!

The first thing you'll be asked to do is go through the doc and examples, checking that you understand what's going on. You'll need to have this little bit of experience with inverse kinematics for these tasks to make sense.

## Easy

- Check out the [documentation](https://tasts-robots.org/doc/pink/) and ask questions if anything is unclear
- Try out the [examples](examples) and report any issue

## Medium

- Write your own example
- Find a use case that is not covered and write a unit test for it
- Benchmark the performance of the following design choice depending on the number and types of tasks:
    - The current ``Configuration`` interface, which does ``pin.computeJointJacobians`` + ``pin.getFrameJacobian``
    - The alternative using only ``pin.computeFrameJacobian``
- Import the [`COMTask`](https://github.com/stephane-caron/pymanoid/blob/1b16932ee3216782fc4b66abafcf8099c19d6d8e/pymanoid/tasks.py#L119) from pymanoid
- Import the [`MinCAMTask`](https://github.com/stephane-caron/pymanoid/blob/1b16932ee3216782fc4b66abafcf8099c19d6d8e/pymanoid/tasks.py#L394) from pymanoid
- Import the [`MinVelTask`](https://github.com/stephane-caron/pymanoid/blob/1b16932ee3216782fc4b66abafcf8099c19d6d8e/pymanoid/tasks.py#L424) from pymanoid

## Advanced

- Import the [`COMAccelTask`](https://github.com/stephane-caron/pymanoid/blob/1b16932ee3216782fc4b66abafcf8099c19d6d8e/pymanoid/tasks.py#L163) from pymanoid
- Import the [`MinAccelTask`](https://github.com/stephane-caron/pymanoid/blob/1b16932ee3216782fc4b66abafcf8099c19d6d8e/pymanoid/tasks.py#L357) from pymanoid
- Import the [`PendulumModeTask`](https://github.com/stephane-caron/pymanoid/blob/1b16932ee3216782fc4b66abafcf8099c19d6d8e/pymanoid/tasks.py#L455) from pymanoid
- Solve the question of numerical instability incurred by almost-unfeasible targets where LM damping does not kick in
