#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 Stéphane Caron
#
# /// script
# dependencies = ["daqp", "loop-rate-limiters", "viser", "pin-pink",
# "qpsolvers", "robot_descriptions"]
# ///

"""Manipulability comparison example: two Panda IK solutions side-by-side.

This example visualizes two IK solutions simultaneously:
- Robot 1 (full opacity): Uses ManipulabilityTask for optimization
- Robot 2 (gray, 50% opacity): Without ManipulabilityTask

Both robots track the same end-effector target. Manipulability values for
both configurations are logged to the console for comparison.
"""

from typing import List, Literal, Optional, Tuple

import numpy as np
import pinocchio as pin
import qpsolvers
import viser
import viser.uplot
from loop_rate_limiters import RateLimiter
from pinocchio import visualize
from robot_descriptions.loaders.pinocchio import load_robot_description
from scipy.spatial.transform import Rotation, Slerp
from viser_shapes import add_grid

import pink
from pink import solve_ik
from pink.limits.configuration_limit import ConfigurationLimit
from pink.tasks import DampingTask, FrameTask, ManipulabilityTask, PostureTask
from pink.utils import custom_configuration_vector
from pink.visualization import start_viser_visualizer


def start_configuration_comparison_visualizer(
    robot: pin.RobotWrapper,
    configurations: List[Tuple[np.ndarray, Tuple[float, float, float, float]]],
    labels: Optional[List[str]] = None,
    target_pose: Optional[pin.SE3] = None,
    open: bool = True,  # pylint: disable=redefined-builtin
) -> "viser.ViserServer":
    """Visualize multiple robot configurations side-by-side.

    Creates a viser visualization showing multiple robot configurations
    overlaid, each with a different color/opacity. Useful for comparing
    IK solutions, trajectories, or the effects of different task parameters.

    Args:
        robot: Pinocchio robot wrapper with its model and data.
        configurations: List of (q, rgba) tuples where q is a joint
            configuration vector and rgba is a color tuple (r, g, b, a)
            with values in [0, 1]. The alpha channel controls opacity.
        labels: Optional list of labels for each configuration. If provided,
            must have the same length as configurations.
        target_pose: Optional target end-effector pose to display as a frame.
        open: If set (default), open Viser in a new Web browser tab.

    Returns:
        The viser server instance. Call server.stop() to shut down, or
        use in a context manager.

    Example:
        >>> configs = [
        ...     (q_baseline, (0.5, 0.5, 0.5, 0.5)),  # Gray, semi-transparent
        ...     (q_improved, (0.2, 0.8, 0.2, 1.0)),  # Green, opaque
        ... ]
        >>> server = start_configuration_comparison_visualizer(
        ...     robot, configs, labels=["baseline", "improved"]
        ... )
    """
    import viser
    from scipy.spatial.transform import Rotation

    # Create viser server
    server = viser.ViserServer(port=8080)

    # Create a visualizer for each configuration
    visualizers = []
    for i, (q, rgba) in enumerate(configurations):
        label = labels[i] if labels else f"config_{i}"

        viz = visualize.ViserVisualizer(
            robot.model, robot.collision_model, robot.visual_model
        )
        viz.initViewer(viewer=server, open=(open and i == 0))
        viz.loadViewerModel(
            rootNodeName=f"robot_{label}",
            visual_color=list(rgba),
        )
        viz.display(q)
        visualizers.append(viz)

    # Add target pose indicator if provided
    if target_pose is not None:
        target_quat = Rotation.from_matrix(target_pose.rotation).as_quat(
            scalar_first=True
        )
        server.scene.add_frame(
            "/target_pose",
            position=tuple(target_pose.translation),
            wxyz=tuple(target_quat),
            axes_length=0.15,
            axes_radius=0.008,
        )

    # Add a ground grid for reference
    server.scene.add_grid(
        "/grid",
        width=2.0,
        height=2.0,
        cell_size=0.1,
        plane="xy",
        position=(0.0, 0.0, 0.0),
    )

    return server


TARGET_POSES = [
    pin.SE3(pose)
    for pose in [
        np.array(
            [
                [1, 1.63397e-07, 8e-06, 0.443996],
                [1.63397e-07, -1, 7.78654e-16, 0.233728],
                [8e-06, 1.3064e-12, -1, 0.0351002],
                [0, 0, 0, 1],
            ]
        ),
        np.array(
            [
                [1, 1.63397e-07, 8e-06, 0.443996],
                [1.63397e-07, -1, 7.78654e-16, -0.233728],
                [8e-06, 1.3064e-12, -1, 0.0351002],
                [0, 0, 0, 1],
            ]
        ),
        np.array(
            [
                [0.999993, -0.00367112, -8.00165e-06, 0.245269],
                [0.00367063, 0.999821, 0.0185543, -0.425038],
                [-6.01146e-05, -0.0185542, 0.999828, 0.809132],
                [0, 0, 0, 1],
            ]
        ),
        np.array(
            [
                [0.430884, 1.63397e-07, 0.902407, 0.548952],
                [7.04065e-08, -1, 1.4745e-07, 3.95522e-08],
                [0.902407, 1.30774e-12, -0.430884, 0.695672],
                [0, 0, 0, 1],
            ]
        ),
    ]
]


def interpolate_poses(pose_1: pin.SE3, pose_2: pin.SE3, t: float):
    """Interpolate between two SE3 poses using linear translation and SLERP rotation.

    Args:
        pose_1: Starting pose at t=0.
        pose_2: Ending pose at t=1.
        t: Interpolation parameter in [0, 1].

    Returns:
        Interpolated SE3 pose.
    """
    assert 0 <= t <= 1, "t should be in [0,1]"
    translation = (1 - t) * pose_1.translation + t * pose_2.translation
    slerp = Slerp(
        [0, 1],
        Rotation.concatenate(
            [
                Rotation.from_matrix(pose_1.rotation),
                Rotation.from_matrix(pose_2.rotation),
            ]
        ),
    )
    rotation = slerp(t).as_matrix()
    return pin.SE3(rotation, translation)


class TargetPoseHandle:
    poses: List[pin.SE3]
    index: int
    interpolation_time: float
    hold_time: float
    _elapsed: float
    _phase: Literal["interpolating", "holding"]

    def __init__(
        self,
        poses: List[pin.SE3],
        interpolation_time: float = 3.0,
        hold_time: float = 6.0,
    ):
        self.poses = poses
        self.interpolation_time = interpolation_time
        self.hold_time = hold_time
        self.reset()

    def reset(self, start_pose: pin.SE3 = None):
        self._elapsed = 0.0
        if start_pose is not None:
            self.poses.insert(0, start_pose)
            self.index = 1
            self._phase = "interpolating"
        else:
            self.index = 0
            self._phase = "holding"

    def current_pose(self) -> pin.SE3:
        return self.poses[self.index]

    def previous_pose(self) -> pin.SE3:
        return self.poses[self.index - 1]

    def get_next_pose(self, dt: float) -> pin.SE3:
        self._elapsed += dt
        if self._phase == "interpolating":
            t = min(self._elapsed / self.interpolation_time, 1.0)
            pose = interpolate_poses(
                self.previous_pose(), self.current_pose(), t
            )
            if self._elapsed >= self.interpolation_time:
                self._phase = "holding"
                self._elapsed = 0.0
            return pose
        else:  # holding
            if self._elapsed >= self.hold_time:
                self.index = (self.index + 1) % len(self.poses)
                self._phase = "interpolating"
                self._elapsed = 0.0
            return self.current_pose()


if __name__ == "__main__":
    robot = load_robot_description("panda_description", root_joint=None)
    q_locked = custom_configuration_vector(
        robot,
        panda_joint1=0.0,
        panda_joint2=-0.785398,
        panda_joint3=0.0,
        panda_joint4=-2.35619,
        panda_joint5=0.0,
        panda_joint6=1.5708,
        panda_joint7=0.785398,
    )
    joint_to_lock = ["panda_finger_joint1", "panda_finger_joint2"]
    list_of_joints_to_lock_by_id = [
        robot.model.getJointId(joint_name) for joint_name in joint_to_lock
    ]
    robot.model, further_models = pin.buildReducedModel(
        robot.model,
        [robot.visual_model, robot.collision_model],
        list_of_joints_to_lock_by_id,
        q_locked,
    )
    robot.visual_model, robot.collision_model = further_models
    robot.rebuildData()

    # Create primary visualizer (full opacity) for manipulability-optimized robot
    viz = start_viser_visualizer(robot, open=False)
    viewer = viz.viewer
    frame_name = "panda_hand_tcp"

    target_pose_handler = TargetPoseHandle(poses=TARGET_POSES)

    # Create ghost visualizer (50% opacity, gray) for non-manipulability robot
    viz_ghost = visualize.ViserVisualizer(
        robot.model, robot.collision_model, robot.visual_model
    )
    viz_ghost.initViewer(viewer=viewer, open=False)
    viz_ghost.loadViewerModel(
        rootNodeName="robot_ghost",
        visual_color=[0.7, 0.7, 0.9, 0.5],  # blue with 50% opacity
    )

    # Tasks for robot with manipulability optimization
    end_effector_task = FrameTask(
        frame_name,
        position_cost=1.0,  # [cost] / [m]
        orientation_cost=1.0,  # [cost] / [rad]
        lm_damping=1.0,
    )
    damping_task = DampingTask(
        cost=0.1,  # [cost] / [rad]
    )

    manipulability_task = ManipulabilityTask(
        frame=frame_name,
        cost=0.3,  # [cost] / [m^6]
        lm_damping=0.001,
        gain=1.0,
        manipulability_rate=1.0,
    )

    # collision_barrier = SelfCollisionBarrier(
    #     n_collision_pairs=len(robot.collision_model.collisionPairs),
    #     gain=10.0,
    #     safe_displacement_gain=1.0,
    #     d_min=0.05,
    # )

    tasks_with_manip = [
        end_effector_task,
        damping_task,
        manipulability_task,
    ]

    # Tasks for robot without manipulability optimization
    end_effector_task_ghost = FrameTask(
        frame_name,
        position_cost=1.0,  # [cost] / [m]
        orientation_cost=1.0,  # [cost] / [rad]
        lm_damping=1.0,
    )
    damping_task_ghost = DampingTask(
        cost=0.05,  # [cost] / [rad]
    )
    posture_task_ghost = PostureTask(
        cost=0.05,  # [cost] / [rad]
    )

    config_limit = ConfigurationLimit(robot.model, 0.99)
    tasks_no_manip = [
        end_effector_task_ghost,
        damping_task_ghost,
        posture_task_ghost,
    ]

    # Initial configuration (same for both robots)
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

    # Create two separate configurations
    configuration_manip = pink.Configuration(
        robot.model,
        robot.data,
        q_ref,
        collision_model=robot.collision_model,
        collision_data=robot.collision_data,
    )
    configuration_no_manip = pink.Configuration(
        robot.model,
        robot.data,
        q_ref.copy(),
        collision_model=None,
        collision_data=None,
    )

    # Set initial configuration for both
    configuration_manip.update(q_ref)
    configuration_no_manip.update(q_ref.copy())

    # Get home pose from the initial configuration
    pose_home = configuration_manip.get_transform_frame_to_world(
        end_effector_task.frame
    ).np

    lm_damping_slider = viewer.gui.add_slider(
        "Manipulability Damping",
        min=0.0,
        max=0.001,
        initial_value=manipulability_task.lm_damping,
        step=1e-4,
    )
    manipulability_rate_slider = viewer.gui.add_slider(
        "Manipulability Rate",
        min=-10.0,
        max=10.0,
        initial_value=manipulability_task.manipulability_rate,
        step=1.0,
    )
    manipulability_cost_slider = viewer.gui.add_slider(
        "Manipulability Cost",
        min=0.0,
        max=1.0,
        initial_value=float(
            manipulability_task._cost_float
        ),  # TODO: properly handle cost types in the task class
        step=0.01,
    )
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
        else damping_task.cost[0],
        step=0.01,
    )

    @lm_damping_slider.on_update
    def _(event: viser.GuiEvent[viser.GuiSliderHandle]):
        manipulability_task.lm_damping = event.target.value

    @manipulability_rate_slider.on_update
    def _(event: viser.GuiEvent[viser.GuiSliderHandle]):
        manipulability_task.manipulability_rate = event.target.value

    @manipulability_cost_slider.on_update
    def _(event: viser.GuiEvent[viser.GuiSliderHandle]):
        manipulability_task.cost = event.target.value

    @frame_task_position_cost_slider.on_update
    def _(event: viser.GuiEvent[viser.GuiSliderHandle]):
        end_effector_task.set_position_cost(event.target.value)
        end_effector_task_ghost.set_position_cost(event.target.value)

    @frame_task_orientation_cost_slider.on_update
    def _(event: viser.GuiEvent[viser.GuiSliderHandle]):
        end_effector_task.set_orientation_cost(event.target.value)
        end_effector_task_ghost.set_orientation_cost(event.target.value)

    @damping_cost_slider.on_update
    def _(event: viser.GuiEvent[viser.GuiSliderHandle]):
        damping_task.cost = event.target.value
        damping_task_ghost.cost = (
            event.target.value * 0.5
        )  # less damping for ghost

    # Set task targets to home configuration
    for task in tasks_with_manip:
        if isinstance(task, ManipulabilityTask) or isinstance(
            task, DampingTask
        ):
            continue
        task.set_target_from_configuration(configuration_manip)

    for task in tasks_no_manip:
        if isinstance(task, DampingTask):
            continue
        task.set_target_from_configuration(configuration_no_manip)

    # Display initial configurations
    viz.display(configuration_manip.q)
    viz_ghost.display(configuration_no_manip.q)

    # Select QP solver
    solver = qpsolvers.available_solvers[0]
    if "daqp" in qpsolvers.available_solvers:
        solver = "daqp"

    # Single transform handle controlling both robots' end-effector targets
    transform_handle = viz.viewer.scene.add_transform_controls(
        "/end_effector_target",
        position=pose_home[:3, 3],
        wxyz=Rotation.from_matrix(pose_home[:3, :3]).as_quat(
            scalar_first=True
        ),
        scale=0.3,
        line_width=3.0,
    )

    auto_driving = False
    activate_auto_drive = viz.viewer.gui.add_checkbox(
        label="Auto-drive target",
        initial_value=auto_driving,
    )

    # Store raw target from UI (updated by callback)
    raw_target = {
        "position": pose_home[:3, 3].copy(),
        "rotation": pose_home[:3, :3].copy(),
    }

    @activate_auto_drive.on_update
    def _(gui_event: viser.GuiEvent[viser.GuiCheckboxHandle]) -> None:
        global auto_driving
        auto_driving = gui_event.target.value
        if auto_driving:
            print("Auto-driving target through predefined poses...")
            target_pose_handler.reset(
                start_pose=configuration_manip.get_transform_frame_to_world(
                    frame_name
                )
            )
        else:
            print("Manual control enabled. Drag the transform handle.")

    @transform_handle.on_update
    def _(handle: viser.TransformControlsEvent) -> None:
        if auto_driving:
            return

        pose = end_effector_task.transform_target_to_world
        pose.translation = np.array(handle.target.position)
        pose.rotation = Rotation.from_quat(
            handle.target.wxyz,
            scalar_first=True,
        ).as_matrix()
        pose_ghost = end_effector_task_ghost.transform_target_to_world
        pose_ghost.translation = np.array(handle.target.position)
        pose_ghost.rotation = Rotation.from_quat(
            handle.target.wxyz,
            scalar_first=True,
        ).as_matrix()

    add_grid(viewer)

    rate = RateLimiter(frequency=200.0, warn=False)
    dt = rate.period
    t = 0.0  # [s]

    # Real-time manipulability plotting setup
    num_timesteps = int(5.0 / dt)  # 5 second window at 200Hz = 1000 samples

    # Rolling buffers for time and manipulability values
    time_data = np.zeros(num_timesteps, dtype=np.float64)
    for i in range(num_timesteps):
        time_data[i] = (i - num_timesteps) * dt
    manip_data_opt = np.zeros(num_timesteps, dtype=np.float64)
    manip_data_base = np.zeros(num_timesteps, dtype=np.float64)
    error_data_opt = np.zeros(num_timesteps, dtype=np.float64)
    error_data_base = np.zeros(num_timesteps, dtype=np.float64)

    uplot_manip = viewer.gui.add_uplot(
        data=(time_data, manip_data_opt, manip_data_base),
        series=(
            viser.uplot.Series(label="time"),
            viser.uplot.Series(
                label="Optimized",
                stroke="#2ecc71",  # green
                width=2,
            ),
            viser.uplot.Series(
                label="Baseline",
                stroke="#e74c3c",  # red
                width=2,
                dash=(4.0, 4.0),
            ),
        ),
        title="Manipulability Comparison",
        scales={
            "x": viser.uplot.Scale(time=False, auto=True),
            "y": viser.uplot.Scale(range=(0.0, 0.015)),
        },
        legend=viser.uplot.Legend(show=True),
        aspect=2.5,
    )

    uplot_error = viewer.gui.add_uplot(
        data=(time_data, error_data_opt, error_data_base),
        series=(
            viser.uplot.Series(label="time"),
            viser.uplot.Series(
                label="Optimized",
                stroke="#2ecc71",  # green
                width=2,
            ),
            viser.uplot.Series(
                label="Baseline",
                stroke="#e74c3c",  # red
                width=2,
                dash=(4.0, 4.0),
            ),
        ),
        title="Tracking Error Comparison",
        scales={
            "x": viser.uplot.Scale(time=False, auto=True),
            "y": viser.uplot.Scale(auto=True),
        },
        legend=viser.uplot.Legend(show=True),
        aspect=2.5,
    )

    print("Starting manipulability comparison...")
    print("Drag the transform handle to move the end-effector target.")
    print(
        "Watch how the two robots diverge due to manipulability optimization.\n"
    )

    while True:
        # Update target for manipulability-optimized robot
        if auto_driving:
            target_pose = target_pose_handler.get_next_pose(dt)
            end_effector_task.transform_target_to_world = target_pose
            end_effector_task_ghost.transform_target_to_world = target_pose
            # Reset the handle
            transform_handle.position = target_pose.translation
            transform_handle.wxyz = Rotation.from_matrix(
                target_pose.rotation
            ).as_quat(scalar_first=True)

        # Solve IK for both configurations
        try:
            velocity_manip = solve_ik(
                configuration_manip,
                tasks_with_manip,
                dt,
                solver=solver,
                limits=[config_limit],
                # barriers=[collision_barrier],
                safety_break=False,
            )
        except Exception as e:
            print(
                f"IK solver failed for manipulability-optimized robot at time {t:.2f}s: {e}"
            )
            raise e
            velocity_manip = np.zeros_like(configuration_manip.q)

        try:
            velocity_no_manip = solve_ik(
                configuration_no_manip, tasks_no_manip, dt, solver=solver
            )
        except Exception as e:
            print(f"IK solver failed for baseline robot at time {t:.2f}s: {e}")
            velocity_no_manip = np.zeros_like(configuration_no_manip.q)

        # Integrate velocities
        configuration_manip.integrate_inplace(velocity_manip, dt)
        configuration_no_manip.integrate_inplace(velocity_no_manip, dt)

        manip_with = manipulability_task.compute_manipulability(
            configuration_manip
        )
        manip_without = manipulability_task.compute_manipulability(
            configuration_no_manip
        )

        error_with = np.linalg.norm(
            end_effector_task.compute_error(configuration_manip)
        )
        error_without = np.linalg.norm(
            end_effector_task_ghost.compute_error(configuration_no_manip)
        )

        viz.display(configuration_manip.q)
        viz_ghost.display(configuration_no_manip.q)

        time_data = np.roll(time_data, -1)
        time_data[-1] = t
        manip_data_opt = np.roll(manip_data_opt, -1)
        manip_data_opt[-1] = manip_with
        manip_data_base = np.roll(manip_data_base, -1)
        manip_data_base[-1] = manip_without
        uplot_manip.data = (time_data, manip_data_opt, manip_data_base)

        error_data_opt = np.roll(error_data_opt, -1)
        error_data_opt[-1] = error_with
        error_data_base = np.roll(error_data_base, -1)
        error_data_base[-1] = error_without
        uplot_error.data = (time_data, error_data_opt, error_data_base)

        rate.sleep()
        t += dt
