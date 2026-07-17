:github_url: https://github.com/stephane-caron/pink/tree/main/doc/inverse-kinematics.rst

******************
Inverse kinematics
******************

The main function solve inverse kinematics is :func:`.solve_ik`. Here is for
instance how it appears in a closed-loop inverse kinematics:

.. code:: python

    rate = RateLimiter(frequency=100.0)
    while True:
        # [...] <- update task targets here
        velocity = solve_ik(configuration, tasks, rate.dt, solver=solver)
        configuration.integrate_inplace(velocity, rate.dt)
        rate.sleep()

See the ``examples/`` folder in the repository for complete use cases.

.. autofunction:: pink.solve_ik.solve_ik

It is also possible to ask Pink to only build the underlying inverse kinematics
problem via the :func:`.build_ik` function:

.. autofunction:: pink.solve_ik.build_ik
