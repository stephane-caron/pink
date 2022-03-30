# 👷 Contributing

There are many ways you can contribute to Pink, all of them welcome! Here are some ideas of increasing difficulty:

- Check out the [documentation](https://scaron.info/doc/pink/) and report mistakes or ask questions
- Try out the [examples](examples) and report any issue
- Suggests improvements to simplify the API
- Add your own [robot model](https://scaron.info/doc/pink/extras.html#module-pink.models) to the extras so that it can load by ``pink.models.build_from_urdf("/home/.../my_robot_description")``
- Write your own example and open a pull request
- Find a use case that is not covered and write a unit test for it
- Benchmark the performance of the current Configuration interface (``pin.computeJointJacobians`` + ``pin.getFrameJacobian``) compared to using ``pin.computeFrameJacobian``, e.g. depending on the number and locations of tasks
- Solve the question of numerical instability incurred by almost-unfeasible targets where LM damping does not kick in

If you are interested in helping out, open an issue so we can track progress. If you already have a PR, open it so we can review it :)
