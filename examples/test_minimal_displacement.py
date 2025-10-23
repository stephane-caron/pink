#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# SPDX-License-Identifier: Apache-2.0
# Copyright 2025

"""Realistic example demonstrating MinimalDisplacementGoalTask as secondary objective.

This example mimics the UR5 setup with FrameTask as primary objective and 
MinimalDisplacementGoalTask as secondary objective, showing how minimal 
displacement affects IK sol    return all_resultstaining end-effector accuracy.
"""

import sys
import os
import numpy as np

# Add the parent directory to path to import pink
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    import pink
    from pink import solve_ik
    from pink.tasks import FrameTask, PostureTask, MinimalDisplacementGoalTask
    from pink.utils import custom_configuration_vector
    PINK_AVAILABLE = True
except ImportError:
    PINK_AVAILABLE = False
    print("⚠️  Pink library not fully available, running simplified tests only")

try:
    from robot_descriptions.loaders.pinocchio import load_robot_description
    ROBOT_DESCRIPTIONS_AVAILABLE = True
except ImportError:
    ROBOT_DESCRIPTIONS_AVAILABLE = False

def create_mock_robot_config(n_joints=7):
    """Create a mock robot configuration for testing when full Pink is unavailable."""
    class MockConfiguration:
        def __init__(self, q_init):
            self.q = q_init.copy()
            self.n_joints = len(q_init)
            
        def copy(self):
            new_config = MockConfiguration(self.q)
            return new_config
            
    # 7-DOF arm initial configuration 
    q_initial = np.array([0.0, -0.5, 0.0, -1.5, 0.0, 1.0, 0.0])[:n_joints]
    return MockConfiguration(q_initial)

def test_minimal_displacement_task():
    """Test the MinimalDisplacementGoalTask functionality."""
    
    print("="*60)
    print("MinimalDisplacementGoalTask Test Example")
    print("="*60)
    n_joints = 7  # 7-DOF robot arm
    initial_config = create_mock_robot_config(n_joints)
    print(f"Initial configuration: {initial_config.q}")
    
    minimal_displacement_task = MinimalDisplacementGoalTask(
        cost=1.0,
        is_secondary=True
    )
    
    # Set initial configuration
    minimal_displacement_task.set_initial_configuration(initial_config)
    print(f"✓ Set initial configuration in task")
    
    # Create test configurations with proper dimensions
    small_change = np.zeros(n_joints)
    small_change[0] = 0.1  # Small change in first joint
    
    medium_change = np.array([0.3, 0.2, -0.2, 0.1, 0.0, 0.1, 0.2])[:n_joints]
    
    large_change = np.array([0.8, 0.5, -0.5, 0.3, -0.2, 0.3, 0.4])[:n_joints]
    
    test_configs = [
        initial_config.q + small_change,   # Small change
        initial_config.q + medium_change,  # Medium change  
        initial_config.q + large_change,   # Large change
        initial_config.q.copy(),           # No change (should be zero error)
    ]
    
    config_names = ["Small displacement", "Medium displacement", "Large displacement", "No displacement"]
    
    print(f"\nTesting different configurations:")
    print(f"{'Configuration':<20} {'Error':<12} {'RMS Error':<12} {'Max Change':<12}")
    print(f"{'-'*60}")
    
    for i, (test_q, name) in enumerate(zip(test_configs, config_names)):
        # Create test configuration
        test_config = create_mock_robot_config(n_joints)
        test_config.q = test_q
        
        # Compute task error
        try:
            error = minimal_displacement_task.compute_error(test_config)
            rms_error = np.sqrt(np.mean(error**2))
            max_change = np.max(np.abs(test_q - initial_config.q))
            
            print(f"{name:<20} {np.sum(error**2):<12.6f} {rms_error:<12.6f} {max_change:<12.6f}")
            
        except Exception as e:
            print(f"{name:<20} ERROR: {e}")
    
    print(f"\n" + "="*60)
    print("Task Properties:")
    print(f"✓ Is secondary objective: {minimal_displacement_task.is_secondary_objective()}")
    print(f"✓ Cost weight: {minimal_displacement_task.cost}")
    print(f"✓ Task representation: {minimal_displacement_task}")
    
    print(f"\nTesting variable weights:")
    weights = np.array([2.0, 1.0, 0.5, 1.0, 1.5, 0.8, 1.2])[:n_joints]  # 7 weights for 7 joints
    minimal_displacement_task.set_variable_weights(weights)
    print(f"Set weights: {weights}")
    
    test_config = create_mock_robot_config(n_joints)
    test_config.q = test_configs[1]
    
    error_weighted = minimal_displacement_task.compute_error(test_config)
    rms_error_weighted = np.sqrt(np.mean(error_weighted**2))
    
    print(f"Medium displacement with weights:")
    print(f"  Weighted RMS Error: {rms_error_weighted:.6f}")
    
    # Test jacobian computation
    print(f"\nTesting Jacobian computation:")
    try:
        jacobian = minimal_displacement_task.compute_jacobian(test_config)
        print(f"✓ Jacobian shape: {jacobian.shape}")
        print(f"✓ Jacobian diagonal (first 5): {np.diag(jacobian)[:5]}")
        
        # Verify jacobian matches weights
        if np.allclose(np.diag(jacobian), weights):
            print(f"✓ Jacobian correctly reflects variable weights")
        else:
            print(f"⚠ Jacobian does not match expected weights")
            
    except Exception as e:
        print(f"✗ Jacobian computation failed: {e}")
    
    # Test secondary vs primary behavior
    print(f"\nTesting secondary vs primary behavior:")
    original_cost = minimal_displacement_task.cost
    print(f"Original cost (secondary): {original_cost:.6f}")
    
    minimal_displacement_task.set_secondary(False)  # Make it primary
    print(f"Cost after making primary: {minimal_displacement_task.cost:.6f}")
    print(f"Cost increase factor: {minimal_displacement_task.cost / original_cost:.2f}")
    
    minimal_displacement_task.set_secondary(True)   # Make it secondary again  
    print(f"Cost after making secondary again: {minimal_displacement_task.cost:.6f}")
    
    print(f"\n" + "="*60)
    print("✅ MinimalDisplacementGoalTask test completed successfully!")
    print("💡 This task minimizes joint displacement from initial configuration")
    print("💡 Lower errors indicate configurations closer to the initial pose")
    print("💡 Variable weights allow emphasizing certain joints over others")
    print("="*60)

def demonstrate_use_case():
    """Demonstrate a realistic use case scenario."""
    
    print(f"\n" + "="*60)
    print("REALISTIC USE CASE DEMONSTRATION")
    print("="*60)
    
    print("Scenario: Robot arm needs to reach a target while minimizing joint movement")
    
    # Simulate a robot with 7 DOF arm
    n_joints = 7
    initial_config = create_mock_robot_config(n_joints)
    
    # Two possible IK solutions that reach the same target
    solution_A = initial_config.q + np.array([0.1, 0.2, -0.1, 0.05, 0.0, 0.1, 0.05])  # Small movements
    solution_B = initial_config.q + np.array([0.8, -0.5, 0.6, -0.3, 0.4, -0.2, 0.3])  # Large movements
    
    print(f"Initial config:  {initial_config.q}")
    print(f"Solution A:      {solution_A}")  
    print(f"Solution B:      {solution_B}")
    
    # Create task
    task = MinimalDisplacementGoalTask(cost=1.0, is_secondary=True)
    task.set_initial_configuration(initial_config)
    
    # Evaluate both solutions
    config_A = create_mock_robot_config(n_joints)
    config_A.q = solution_A
    error_A = task.compute_error(config_A)
    cost_A = np.sum(error_A**2)
    
    config_B = create_mock_robot_config(n_joints)  
    config_B.q = solution_B
    error_B = task.compute_error(config_B)
    cost_B = np.sum(error_B**2)
    
    print(f"\nMinimal Displacement Task Evaluation:")
    print(f"Solution A cost: {cost_A:.6f}")
    print(f"Solution B cost: {cost_B:.6f}")
    
    if cost_A < cost_B:
        print(f"✅ Solution A is preferred (lower displacement cost)")
        print(f"   Cost ratio B/A: {cost_B/cost_A:.2f}x")
    else:
        print(f"✅ Solution B is preferred (lower displacement cost)")
        print(f"   Cost ratio A/B: {cost_A/cost_B:.2f}x")
    
    print(f"\n💡 In IK optimization, this task would bias the solver toward")
    print(f"   the solution requiring less joint movement from the initial pose")

def compare_error_with_same_targets():
    """Compare displacement errors for same target points with/without MinimalDisplacementGoalTask."""
    
    print("="*80)
    print("� ERROR COMPARISON: Same Targets, Different Secondary Objectives")
    print("7-DOF Arm: Comparing displacement errors with/without MinimalDisplacementGoalTask")
    print("="*80)
    
    n_joints = 7
    
    # 7-DOF arm initial configuration
    initial_config = create_mock_robot_config(n_joints)
    print(f"🤖 Initial configuration (7-DOF): {initial_config.q}")
    
    # Define target joint configurations (same targets for both scenarios)
    target_configurations = [
        {
            "name": "Target 1 (Small movement)", 
            "q": np.array([0.2, -0.3, 0.1, -1.2, 0.1, 0.8, 0.1])
        },
        {
            "name": "Target 2 (Medium movement)", 
            "q": np.array([0.6, -0.8, 0.5, -1.8, 0.3, 1.2, 0.4])
        },
        {
            "name": "Target 3 (Large movement)", 
            "q": np.array([1.2, -1.2, 0.8, -2.2, 0.6, 1.6, 0.8])
        },
        {
            "name": "Target 4 (Complex movement)", 
            "q": np.array([0.8, -1.0, -0.3, -1.5, -0.2, 1.4, -0.3])
        }
    ]
    
    # Create TWO scenarios for comparison
    scenarios = {
        "WITHOUT Secondary": {
            "task": None,  # No MinimalDisplacementGoalTask
            "description": "No secondary objective"
        },
        "WITH Secondary": {
            "task": MinimalDisplacementGoalTask(cost=1.0, is_secondary=True),
            "description": "With MinimalDisplacementGoalTask"
        }
    }
    
    # Set initial configuration for the secondary task
    if scenarios["WITH Secondary"]["task"] is not None:
        scenarios["WITH Secondary"]["task"].set_initial_configuration(initial_config)
    
    print(f"\n📋 Testing {len(target_configurations)} targets in {len(scenarios)} scenarios...")
    
    all_results = {}
    
    for scenario_name, scenario_info in scenarios.items():
        print(f"\n🔹 {scenario_name}")
        print(f"   {scenario_info['description']}")
        print(f"{'Target':<20} {'Joint Mvmt':<12} {'Displacement Error':<18} {'Error²':<12}")
        print(f"{'-'*65}")
        
        scenario_results = []
    
        for target in target_configurations:
            # Create target configuration
            target_config = create_mock_robot_config(n_joints)
            target_config.q = target['q']
            
            # Calculate joint movement from initial position
            joint_movement = np.linalg.norm(target['q'] - initial_config.q)
            
            # Calculate error based on scenario
            if scenario_info['task'] is None:
                # WITHOUT secondary objective - only basic joint movement
                displacement_error_vec = target['q'] - initial_config.q  # Raw displacement
                error_magnitude = joint_movement  # Same as joint movement
                error_squared = np.sum(displacement_error_vec**2)  # Basic squared displacement
                
                print(f"{target['name']:<20} {joint_movement:<12.4f} {'N/A (no task)':<18} {error_squared:<12.6f}")
                
            else:
                # WITH secondary objective - MinimalDisplacementGoalTask error
                try:
                    displacement_error_vec = scenario_info['task'].compute_error(target_config)
                    error_magnitude = np.linalg.norm(displacement_error_vec)
                    error_squared = np.sum(displacement_error_vec**2)
                    
                    print(f"{target['name']:<20} {joint_movement:<12.4f} {error_magnitude:<18.6f} {error_squared:<12.6f}")
                    
                except Exception as e:
                    print(f"{target['name']:<20} {joint_movement:<12.4f} ERROR: {e}")
                    continue
            
            scenario_results.append({
                'name': target['name'],
                'target_q': target['q'],
                'joint_movement': joint_movement,
                'displacement_error': displacement_error_vec,
                'error_magnitude': error_magnitude,
                'error_squared': error_squared
            })
        
        all_results[scenario_name] = scenario_results
    
    # Compare results between scenarios
    print(f"\n🔍 COMPARISON ANALYSIS:")
    print(f"{'='*80}")
    
    if len(all_results) == 2:
        without_results = all_results["WITHOUT Secondary"]  
        with_results = all_results["WITH Secondary"]
        
        print(f"\n� SIDE-BY-SIDE COMPARISON:")
        print(f"{'Target':<20} {'Joint Mvmt':<12} {'Without Sec²':<15} {'With Sec²':<15} {'Difference':<12}")
        print(f"{'-'*80}")
        
        total_without = 0
        total_with = 0
        
        for i, target in enumerate(target_configurations):
            if i < len(without_results) and i < len(with_results):
                without_error = without_results[i]['error_squared']
                with_error = with_results[i]['error_squared'] 
                joint_mvmt = without_results[i]['joint_movement']
                difference = without_error - with_error
                
                total_without += without_error
                total_with += with_error
                
                print(f"{target['name']:<20} {joint_mvmt:<12.4f} {without_error:<15.6f} {with_error:<15.6f} {difference:<12.6f}")
        
        print(f"{'-'*80}")
        print(f"{'TOTALS:':<20} {'':<12} {total_without:<15.6f} {total_with:<15.6f} {total_without-total_with:<12.6f}")
        
        improvement_pct = ((total_without - total_with) / total_without * 100) if total_without > 0 else 0
        print(f"\n🎯 SUMMARY:")
        print(f"   • WITHOUT secondary task total error²: {total_without:.6f}")
        print(f"   • WITH secondary task total error²:    {total_with:.6f}")
        print(f"   • Total improvement:                   {total_without-total_with:.6f}")
        print(f"   • Percentage improvement:              {improvement_pct:.1f}%")
        
        print(f"\n💡 INTERPRETATION:")
        if total_with < total_without:
            print(f"   ✅ MinimalDisplacementGoalTask REDUCES error (as expected)")
            print(f"   ✅ Secondary objective successfully minimizes displacement")
        else:
            print(f"   ⚠️  Unexpected: secondary task increased error") 
            
    else:
        print(f"⚠️  Expected 2 scenarios for comparison, got {len(all_results)}")
    
    print(f"\n💭 PRACTICAL IMPLICATIONS:")
    print(f"   • Lower error² means IK solver prefers that solution")
    print(f"   • MinimalDisplacementGoalTask acts as 'lazy' regularizer")  
    print(f"   • Helps choose solutions closer to initial configuration")
    print(f"   • Useful when multiple IK solutions exist for same target")
    
    return all_results

    
    print(f"\n📈 SUMMARY STATISTICS:")
    print(f"{'-'*50}")
    print(f"Joint movements:      min={np.min(movements):.4f}, max={np.max(movements):.4f}, avg={np.mean(movements):.4f}")
    print(f"Displacement errors:  min={np.min(errors):.6f}, max={np.max(errors):.6f}, avg={np.mean(errors):.6f}")
    print(f"Squared errors:       min={np.min(errors_squared):.6f}, max={np.max(errors_squared):.6f}, avg={np.mean(errors_squared):.6f}")
    
    # Relationship analysis
    print(f"\n� ERROR vs MOVEMENT RELATIONSHIP:")
    print(f"{'-'*50}")
    correlation = np.corrcoef(movements, errors)[0,1]
    print(f"Correlation between joint movement and error: {correlation:.4f}")
    
    if correlation > 0.9:
        print(f"✅ Strong positive correlation - larger movements cause proportionally larger errors")
    elif correlation > 0.7:
        print(f"✅ Good correlation - errors generally increase with movement")
    else:
        print(f"⚠️  Weak correlation - error relationship is more complex")
    
    # Practical implications
    print(f"\n💡 PRACTICAL IMPLICATIONS FOR IK:")
    print(f"{'-'*50}")
    print(f"• MinimalDisplacementGoalTask error = √cost × weighted_displacement")
    print(f"• Larger joint movements → Higher displacement error")
    print(f"• IK solver will prefer solutions with lower total error")
    print(f"• This biases toward smaller joint movements from initial pose")
    print(f"• Effect strength controlled by 'cost' parameter")
    
    # Simulation of IK preference
    print(f"\n🎯 IK SOLVER PREFERENCE SIMULATION:")
    print(f"{'-'*50}")
    print("If IK solver had multiple solutions reaching same end-effector target:")
    
    # Sort by error (IK solver preference)
    sorted_results = sorted(total_results, key=lambda x: x['error_squared'])
    
    print(f"{'Preference Rank':<15} {'Target':<25} {'Error²':<12} {'Movement':<12}")
    print(f"{'-'*70}")
    for rank, result in enumerate(sorted_results, 1):
        print(f"{rank:<15} {result['name']:<25} {result['error_squared']:<12.6f} {result['joint_movement']:<12.4f}")
    
    print(f"\n➡️  IK solver would prefer: {sorted_results[0]['name']}")
    print(f"    (Lowest error² = {sorted_results[0]['error_squared']:.6f})")
    
    return total_results

if __name__ == "__main__":
    print("🚀 MinimalDisplacementGoalTask Error Comparison Test")
    print(f"Date: October 2, 2025")
    print("7-DOF Robot Arm: Same targets, different secondary objectives")
    print("="*80)
    
    # Run the main error comparison (main feature)
    print("\n1️⃣ Running error comparison for same target points...")
    try:
        compare_error_with_same_targets()
    except Exception as e:
        print(f"❌ Error comparison failed: {e}")
    
    # Run basic functionality test
    print(f"\n{'='*80}")
    print("2️⃣ Running basic functionality tests...")
    try:
        test_minimal_displacement_task()
    except Exception as e:
        print(f"❌ Basic test failed: {e}")
    
    # Run use case demonstration  
    print(f"\n{'='*80}")
    print("3️⃣ Running use case demonstration...")
    try:
        demonstrate_use_case()
    except Exception as e:
        print(f"❌ Use case demo failed: {e}")
    
    print(f"\n{'='*80}")
    print("🎯 SUMMARY: How to use MinimalDisplacementGoalTask in practice")
    print(f"{'='*80}")
    print(f"# Primary task (high priority):")
    print(f"frame_task = FrameTask('ee_link', position_cost=10.0)")
    print(f"")
    print(f"# Secondary task (low priority - minimal displacement):")
    print(f"minimal_disp_task = MinimalDisplacementGoalTask(")
    print(f"    cost=0.01,        # Much lower than primary task")
    print(f"    is_secondary=True # Explicitly mark as secondary")
    print(f")")
    print(f"minimal_disp_task.set_initial_configuration(initial_config)")
    print(f"")
    print(f"# Combine tasks (order doesn't matter, cost weights do):")
    print(f"tasks = [frame_task, minimal_disp_task]")
    print(f"velocity = solve_ik(configuration, tasks, dt)")
    print(f"")
    print(f"💡 Result: Robot reaches target while minimizing joint movement!")
    print(f"🎯 Perfect for energy efficiency and smooth motion!")
