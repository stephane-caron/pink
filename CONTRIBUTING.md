# 👷 Contributing

If any of these sound interesting, open an [issue](https://github.com/tasts-robots/pink/issues) and say you're on it!

The first thing you'll be asked to do is go through the doc and examples, checking that you understand what's going on. You'll need to have this little bit of experience with inverse kinematics for these tasks to make sense.

## Easy

- Check out the [documentation](https://tasts-robots.org/doc/pink/) and ask questions if anything is unclear
- Try out the [examples](examples) and report any issue
- Add a [robot model](https://tasts-robots.org/doc/pink/extras.html#module-pink.models) to the extras so that it can load by ``pink.models.build_from_urdf("/home/.../my_robot_description")``

## Medium

- Write your own example
- Find a use case that is not covered and write a unit test for it
- Benchmark the performance of the following design choice depending on the number and types of tasks:
    - The current ``Configuration`` interface, which does ``pin.computeJointJacobians`` + ``pin.getFrameJacobian``
    - The alternative using only ``pin.computeFrameJacobian``

## Research-grade

- Solve the question of numerical instability incurred by almost-unfeasible targets where LM damping does not kick in
