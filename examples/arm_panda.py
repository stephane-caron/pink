#!/usr/bin/env python3
# /// script
# dependencies = ["daqp", "loop-rate-limiters", "viser", "pin-pink",
# "qpsolvers", "robot_descriptions"]
# ///

"""Example of a 7-dof robot manipulator and viser visualizer to tune the different task costs."""

import qpsolvers
import viser
from loop_rate_limiters import RateLimiter
from robot_descriptions.loaders.pinocchio import load_robot_description
from scipy.spatial.transform import Rotation

import pink
from pink import solve_ik
from pink.tasks import DampingTask, FrameTask, PostureTask
from pink.utils import custom_configuration_vector
from pink.visualization import start_viser_visualizer

if __name__ == "__main__":
    robot = load_robot_description("panda_description", root_joint=None)

    viz = start_viser_visualizer(robot)
    viewer = viz.viewer
    frame_name = "panda_hand_tcp"

    end_effector_task = FrameTask(
        frame_name,
        position_cost=1.0,  # [cost] / [m]
        orientation_cost=1.0,  # [cost] / [rad]
        lm_damping=1.0,
    )
    posture_task = PostureTask(
        cost=1e-3,  # [cost] / [rad]
    )
    damping_task = DampingTask(
        cost=1e-3,  # [cost] / [rad/s]
    )
    tasks = [end_effector_task, posture_task, damping_task]

    q_ref = custom_configuration_vector(
        robot,
        panda_joint1=0.0,
        panda_joint2=-0.785398,
        panda_joint3=0.0,
        panda_joint4=-2.35619,
        panda_joint5=0.0,
        panda_joint6=1.5708,
        panda_joint7=0.785398,
    )

    configuration = pink.Configuration(robot.model, robot.data, q_ref)

    configuration.update(q_ref)

    pose_home = configuration.get_transform_frame_to_world(
        end_effector_task.frame
    ).np

    end_effector_task.set_target_from_configuration(configuration)
    posture_task.set_target_from_configuration(configuration)

    viz.display(configuration.q)

    # Select QP solver
    solver = qpsolvers.available_solvers[0]
    if "daqp" in qpsolvers.available_solvers:
        solver = "daqp"

    viz.viewer.scene.add_grid(
        "/grid",
        width=2,
        height=2,
        position=(0.0, 0.0, 0.0),
    )

    # === Interactive handle to move the end-effector target in the scene ===
    transform_handle = viz.viewer.scene.add_transform_controls(
        "/end_effector_target",
        position=pose_home[:3, 3],
        wxyz=Rotation.from_matrix(pose_home[:3, :3]).as_quat(
            scalar_first=True
        ),
        scale=0.3,
        line_width=3.0,
    )

    @transform_handle.on_update
    def _(handle: viser.TransformControlsEvent) -> None:
        end_effector_target = end_effector_task.transform_target_to_world
        end_effector_target.translation = handle.target.position
        end_effector_target.rotation = Rotation.from_quat(
            handle.target.wxyz,
            scalar_first=True,
        ).as_matrix()

    # === GUI elemets: We add sliders to tune the different tasks ===
    frame_task_position_cost_slider = viewer.gui.add_slider(
        "Frame Task Position Cost",
        min=0.0,
        max=10.0,
        initial_value=end_effector_task.position_cost
        if isinstance(end_effector_task.position_cost, (float, int))
        else end_effector_task.position_cost[0],
        step=0.1,
    )
    frame_task_orientation_cost_slider = viewer.gui.add_slider(
        "Frame Task Orientation Cost",
        min=0.0,
        max=10.0,
        initial_value=end_effector_task.orientation_cost
        if isinstance(end_effector_task.orientation_cost, (float, int))
        else end_effector_task.orientation_cost[0],
        step=0.1,
    )
    damping_cost_slider = viewer.gui.add_slider(
        "Damping Cost",
        min=0.0,
        max=1.0,
        initial_value=damping_task.cost
        if isinstance(damping_task.cost, (float, int))
        else 0.0,
        step=0.01,
    )
    posture_cost_slider = viewer.gui.add_slider(
        "Posture Cost",
        min=0.0,
        max=1.0,
        initial_value=posture_task.cost
        if isinstance(posture_task.cost, (float, int))
        else 0.0,
        step=0.01,
    )
    reset_button = viewer.gui.add_button("Reset Configuration")

    @frame_task_position_cost_slider.on_update
    def _(event: viser.GuiEvent[viser.GuiSliderHandle]):
        end_effector_task.set_position_cost(event.target.value)

    @frame_task_orientation_cost_slider.on_update
    def _(event: viser.GuiEvent[viser.GuiSliderHandle]):
        end_effector_task.set_orientation_cost(event.target.value)

    @damping_cost_slider.on_update
    def _(event: viser.GuiEvent[viser.GuiSliderHandle]):
        damping_task.cost = event.target.value

    @posture_cost_slider.on_update
    def _(event: viser.GuiEvent[viser.GuiSliderHandle]):
        posture_task.cost = event.target.value

    @reset_button.on_click
    def _(_: viser.GuiEvent[viser.GuiButtonHandle]):
        configuration.update(q_ref)
        end_effector_task.set_target_from_configuration(configuration)
        posture_task.set_target_from_configuration(configuration)
        transform_handle.position = configuration.get_transform_frame_to_world(
            end_effector_task.frame
        ).np[:3, 3]
        transform_handle.wxyz = Rotation.from_matrix(
            configuration.get_transform_frame_to_world(
                end_effector_task.frame
            ).np[:3, :3]
        ).as_quat(scalar_first=True)

    # === End of GUI elements ===

    rate = RateLimiter(frequency=200.0, warn=True)
    dt = rate.period

    while True:
        velocity = solve_ik(configuration, tasks, dt, solver=solver)
        configuration.integrate_inplace(velocity, dt)

        viz.display(configuration.q)
        rate.sleep()
