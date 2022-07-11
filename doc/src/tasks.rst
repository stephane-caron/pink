:github_url: https://github.com/tasts-robots/pink/tree/master/doc/src/tasks.rst

.. _Tasks:

*****
Tasks
*****

Kinematic tasks
===============

**TODO:** we describe below the core concepts and notations, but there is a bit
more dimensional analysis in Pink's position and orientation costs
(respectively in cost / m and cost / rad) that is not written down.

Let us consider the task of bringing a point :math:`p`, located on one of
the robot’s links, to a goal position :math:`p^*`, both point coordinates
being expressed in the world frame. When the robot is in configuration
:math:`q`, the (position) `residual
<https://en.wikipedia.org/wiki/Residual_(numerical_analysis)>`__ of this task
is:

.. math::

    r(q) = p^* - p(q)

The goal of the task is to bring this residual to zero. Next, from forward
kinematics we know how to compute the Jacobian matrix of :math:`p`:

.. math::

    J(q) = \frac{\partial p}{\partial q}(q),

which maps joint velocities :math:`\dot{q}` to end-point velocities :math:`pd`
via :math:`J(q) \dot{q} = pd`. Suppose that we apply a velocity
:math:`\dot{q}` over a small duration :math:`\delta t`. The new residual after
:math:`\delta t` is :math:`r' = r - pd \delta t`. Our goal is to
cancel it, that is :math:`r' = \boldsymbol{0} \Leftrightarrow pd \delta t
= r`, which leads us to define the *velocity residual*:

.. math::

    v(q, \delta t) := \frac{r(q)}{\delta t} = \frac{p^* - p(q)}{\delta t}

The best option is then to select :math:`\dot{q}` such that:

.. math::

    J(q) \dot{q} = pd = v(q, \delta t)

If the Jacobian were invertible, we could take :math:`\dot{q} = J^{-1} v`.
However, that’s usually not the case (think of a point task where :math:`J`
has three rows and one column per DOF). The best solution that we can get in
the least-square sense is the solution to:

.. math::

    \underset{\dot{q}}{\textrm{minimize}}\ \| J \dot{q} - v \|^2,

and is given by the `pseudo-inverse
<https://en.wikipedia.org/wiki/Generalized_inverse>`__ :math:`J^\dagger` of
:math:`J`: :math:`\dot{q} = J^\dagger v`. By writing this equivalently as
:math:`(J^\top J) q = J^\top v`, we see that this approach is
exactly the `Gauss-Newton algorithm
<https://en.wikipedia.org/wiki/Gauss%E2%80%93Newton_algorithm>`__. (There is a
sign difference compared with the Gauss-Newton update rule, which comes from
our use of the end-effector Jacobian :math:`\partial p / \partial q`
rather than the residual Jacobian :math:`\partial r / \partial q`.)

Gain of a task
--------------

For this solution to work, the time step :math:`\delta t` should be
sufficiently small, so that the variations of the Jacobian term between
:math:`q` and :math:`q + \dot{q} \delta t` can be neglected. The total
variation is

.. math::

    J(q + \dot{q} \delta t) \dot{q} - J(q) \dot{q}
    = \delta t\,\dot{q}^\top H(q) \dot{q}

where :math:`H(q)` is the *Hessian* matrix of the task. This matrix is
more expensive to compute than :math:`J`. Rather than checking that the
variation above is small enough, a `common practice
<http://www.roboticsproceedings.org/rss07/p21.pdf>`__ is to multiply the
velocity residual by a `proportional gain
<https://en.wikipedia.org/wiki/Proportional_control>`__ :math:`K_p \in [0, 1]`:

.. math::

    J(q) \dot{q} = K_p\, v

For example, :math:`K_p = 0.5` means that the system will (try at best to) cut
the residual by half at each time step :math:`\delta t`. Adding this gain does
not change the exponential convergence to the solution :math:`r =
\boldsymbol{0}`, and helps avoid *overshooting* of the real solution. When you
observe instabilities in your IK tracking, reducing task gains is usually a
good idea.

Tasks in Pink
=============

.. automodule:: pink.tasks.task
    :members:

Body task
---------

.. automodule:: pink.tasks.body_task
    :members:

Posture task
------------

.. automodule:: pink.tasks.posture_task
    :members:
