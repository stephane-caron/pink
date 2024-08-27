# 👷 Contributing

This project's goal is to make it easy to do inverse kinematics on all kinds of robot models. All contributions are welcome, for example here are some ways to help:

- Try out the [examples](examples) and report any issue
- Propose improvements or ask questions about the [documentation](https://stephane-caron.github.io/pink/)
- Pick something you want to do with one of the many [robot descriptions](https://github.com/robot-descriptions/robot_descriptions.py) and write a new example
- Find a use case that is not covered and write a unit test for it
- Benchmark the performance of the following design choice depending on the number and types of tasks:
    - The current `Configuration` interface, which does `pin.computeJointJacobians` + `pin.getFrameJacobian`
    - The alternative using only `pin.computeFrameJacobian`

There are also a number of tasks from Pymanoid that are not in Pink yet. If you happen to need one of them for your application, check out the reference implementation and add it to Pink:

- [`COMTask`](https://github.com/stephane-caron/pymanoid/blob/1b16932ee3216782fc4b66abafcf8099c19d6d8e/pymanoid/tasks.py#L119)
- [`MinCAMTask`](https://github.com/stephane-caron/pymanoid/blob/1b16932ee3216782fc4b66abafcf8099c19d6d8e/pymanoid/tasks.py#L394)
- [`MinVelTask`](https://github.com/stephane-caron/pymanoid/blob/1b16932ee3216782fc4b66abafcf8099c19d6d8e/pymanoid/tasks.py#L424)
- [`COMAccelTask`](https://github.com/stephane-caron/pymanoid/blob/1b16932ee3216782fc4b66abafcf8099c19d6d8e/pymanoid/tasks.py#L163)
- [`MinAccelTask`](https://github.com/stephane-caron/pymanoid/blob/1b16932ee3216782fc4b66abafcf8099c19d6d8e/pymanoid/tasks.py#L357)
- [`PendulumModeTask`](https://github.com/stephane-caron/pymanoid/blob/1b16932ee3216782fc4b66abafcf8099c19d6d8e/pymanoid/tasks.py#L455)

Finally, there is one question that has been open for a long time and is still a pain in practice:

- Solve the question of numerical instability incurred by almost-unfeasible targets where LM damping does not kick in

If any of those sound interesting, open an [issue](https://github.com/stephane-caron/pink/issues) and say you're on it!
