#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# SPDX-License-Identifier: Apache-2.0
# Copyright 2025

"""Example demonstrating MinimalDisplacementGoalTask with comparison.

This example shows how MinimalDisplacementGoalTask affects the robot's behavior
by comparing two scenarios:
1. Without minimal displacement task: robot may take large joint motions
2. With minimal displacement task: robot prefers smaller joint changes
"""

import numpy as np
import qpsolvers
import time

import meshcat_shapes
import pink
from pink import solve_ik
from pink.tasks import FrameTask, PostureTask, MinimalDisplacementGoalTask
from pink.utils import custom_configuration_vector
from pink.visualization import start_meshcat_visualizer

try:
    from loop_rate_limiters import RateLimiter
except ModuleNotFoundError as exc:
    raise ModuleNotFoundError(
        "Examples use loop rate limiters, "
        "try `[conda|pip] install loop-rate-limiters`"
    ) from exc

try:
    from robot_descriptions.loaders.pinocchio import load_robot_description
except ModuleNotFoundError:
    raise ModuleNotFoundError(
        "Examples need robot_descriptions, "
        "try `[conda|pip] install robot_descriptions`"
    )


def run_scenario(robot, viz, use_minimal_displacement=True, duration=5.0):
    """Run IK scenario with or without minimal displacement task.
    
    Args:
        robot: Robot model
        viz: Visualizer 
        use_minimal_displacement: Whether to include minimal displacement task
        duration: How long to run the scenario in seconds
        
    Returns:
        tuple: (final_configuration, total_displacement, max_joint_change)
    """
    print(f"\n{'='*60}")
    if use_minimal_displacement:
        print("SCENARIO: WITH MinimalDisplacementGoalTask")
        print("Expected: Smaller joint movements, smoother motion")
    else:
        print("SCENARIO: WITHOUT MinimalDisplacementGoalTask") 
        print("Expected: Potentially larger joint movements")
    print(f"{'='*60}")

    # Set up initial configuration
    q_initial = custom_configuration_vector(
        robot,
        j2s6s200_joint_1=0.0,
        j2s6s200_joint_2=0.8,
        j2s6s200_joint_3=0.5,
        j2s6s200_joint_4=0.0,
        j2s6s200_joint_5=0.5,
        j2s6s200_joint_6=0.0,
    )
    configuration = pink.Configuration(robot.model, robot.data, q_initial.copy())
    
    # Create tasks
    end_effector_task = FrameTask(
        "j2s6s200_end_effector",
        position_cost=10.0,  # High cost for precise positioning
        orientation_cost=5.0,
    )
    
    posture_task = PostureTask(
        cost=1e-4,  # Very low cost, just for regularization
    )
    
    tasks = [end_effector_task, posture_task]
    
    # Add minimal displacement task if requested
    if use_minimal_displacement:
        minimal_displacement_task = MinimalDisplacementGoalTask(
            cost=1.0,  # Moderate cost for displacement minimization
            is_secondary=True  # Secondary objective
        )
        minimal_displacement_task.set_initial_configuration(configuration)
        tasks.append(minimal_displacement_task)
        print("✓ Added MinimalDisplacementGoalTask with cost=1.0")
    
    # Set initial targets
    for task in tasks:
        if hasattr(task, 'set_target_from_configuration'):
            task.set_target_from_configuration(configuration)
    
    # Select QP solver
    solver = qpsolvers.available_solvers[0]
    if "daqp" in qpsolvers.available_solvers:
        solver = "daqp"
    
    print(f"Using solver: {solver}")
    
    # Tracking variables
    joint_positions_history = [configuration.q.copy()]
    total_displacement = 0.0
    max_joint_change = 0.0
    
    rate = RateLimiter(frequency=100.0, warn=False)
    dt = rate.period
    t = 0.0
    
    print(f"Initial joint positions: {configuration.q}")
    
    while t < duration:
        # Create a challenging target trajectory: figure-8 motion
        phase = 2.0 * np.pi * t / duration
        target_transform = configuration.get_transform_frame_to_world("j2s6s200_end_effector").copy()
        
        # Figure-8 motion in Y-Z plane
        target_transform.translation[0] += 0.05 * np.sin(phase)
        target_transform.translation[1] += 0.15 * np.sin(2 * phase)  # Figure-8 
        target_transform.translation[2] += 0.1 * np.sin(phase)
        
        # Update end effector target
        end_effector_task.set_target(target_transform)
        
        # Store previous configuration for displacement calculation
        q_prev = configuration.q.copy()
        
        # Solve IK and integrate
        try:
            velocity = solve_ik(configuration, tasks, dt, solver=solver)
            configuration.integrate_inplace(velocity, dt)
            
            # Calculate displacement metrics
            joint_change = np.abs(configuration.q - q_prev)
            max_joint_change = max(max_joint_change, np.max(joint_change))
            total_displacement += np.sum(joint_change)
            joint_positions_history.append(configuration.q.copy())
            
            # Visualize (at reduced rate to avoid overwhelming)
            if int(t * 100) % 5 == 0:  # Update visualization every 5 steps
                viz.display(configuration.q)
                
        except Exception as e:
            print(f"IK solver failed at t={t:.2f}: {e}")
            break
        
        rate.sleep()
        t += dt
    
    # Calculate final metrics
    final_displacement = np.linalg.norm(configuration.q - q_initial)
    joint_variance = np.var(np.array(joint_positions_history), axis=0)
    
    print(f"\nRESULTS:")
    print(f"Final configuration: {configuration.q}")
    print(f"Total displacement from initial: {final_displacement:.4f} rad")
    print(f"Cumulative joint movement: {total_displacement:.4f} rad")
    print(f"Max single joint change: {max_joint_change:.4f} rad")
    print(f"Joint variance (smoothness): {np.mean(joint_variance):.6f}")
    
    return configuration, total_displacement, max_joint_change, joint_variance


if __name__ == "__main__":
    # Load robot
    robot = load_robot_description("gen2_description", root_joint=None)
    
    # Start visualizer
    viz = start_meshcat_visualizer(robot)
    viewer = viz.viewer
    
    # Add visualization elements
    meshcat_shapes.frame(viewer["target"], opacity=0.7, tube_radius=0.01)
    
    print("MinimalDisplacementGoalTask Comparison Example")
    print("This example compares robot behavior with and without minimal displacement task")
    
    # Run without minimal displacement task
    print("\nPress Enter to start first scenario (WITHOUT minimal displacement)...")
    input()
    
    config1, displacement1, max_change1, variance1 = run_scenario(
        robot, viz, use_minimal_displacement=False, duration=5.0
    )
    
    # Wait before second scenario
    print("\nPress Enter to start second scenario (WITH minimal displacement)...")
    input()
    
    config2, displacement2, max_change2, variance2 = run_scenario(
        robot, viz, use_minimal_displacement=True, duration=5.0
    )
    
    # Compare results
    print(f"\n{'='*60}")
    print("COMPARISON SUMMARY")
    print(f"{'='*60}")
    print(f"                              WITHOUT    WITH     IMPROVEMENT")
    print(f"Total displacement:           {displacement1:7.4f}  {displacement2:7.4f}  {((displacement1-displacement2)/displacement1*100):6.1f}%")
    print(f"Max joint change:             {max_change1:7.4f}  {max_change2:7.4f}  {((max_change1-max_change2)/max_change1*100):6.1f}%")
    print(f"Average joint variance:       {np.mean(variance1):7.6f}  {np.mean(variance2):7.6f}  {((np.mean(variance1)-np.mean(variance2))/np.mean(variance1)*100):6.1f}%")
    
    print(f"\n🎯 MinimalDisplacementGoalTask Effects:")
    if displacement2 < displacement1:
        print(f"   ✓ Reduced total joint movement by {((displacement1-displacement2)/displacement1*100):.1f}%")
    if max_change2 < max_change1:
        print(f"   ✓ Reduced maximum joint change by {((max_change1-max_change2)/max_change1*100):.1f}%") 
    if np.mean(variance2) < np.mean(variance1):
        print(f"   ✓ Improved motion smoothness by {((np.mean(variance1)-np.mean(variance2))/np.mean(variance1)*100):.1f}%")
    
    print(f"\n💡 The MinimalDisplacementGoalTask acts as a 'lazy' regularizer,")
    print(f"   preferring solutions that require less joint movement while")
    print(f"   still achieving the primary end-effector positioning task.")
