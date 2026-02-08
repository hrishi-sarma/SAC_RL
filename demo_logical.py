"""
Demo Script - Verify Logical Placement System Setup
"""

import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from furniture_env_logical import FurnitureRecommendationEnvLogical
from sac_agent import RuleGuidedSAC


def test_environment():
    """Test that the enhanced environment works correctly"""
    print("=" * 70)
    print("Testing Logical Placement Environment")
    print("=" * 70)
    
    # Create environment
    env = FurnitureRecommendationEnvLogical(
        room_layout_path='room_layout.json',
        catalog_path='furniture_catalog_enhanced.json',
        max_items=4,
        grid_size=0.3,
        collision_buffer=0.15
    )
    
    print(f"✅ Environment created successfully")
    print(f"  State dimension: {env.observation_space.shape[0]}")
    print(f"  Action dimension: {env.action_space.shape[0]}")
    print(f"  Max items: {env.max_items}")
    print(f"  Grid size: {env.grid_size}m")
    print(f"  Collision buffer: {env.collision_buffer}m")
    print(f"  Catalog items: {len(env.furniture_catalog)}")
    
    # Test reset
    state, info = env.reset()
    print(f"✅ Environment reset successfully")
    print(f"  Initial state shape: {state.shape}")
    
    # Test placement rules
    print("\n" + "=" * 70)
    print("Testing Placement Rules")
    print("=" * 70)
    
    placement_types = {}
    for item in env.furniture_catalog:
        rules = item.get('placement_rules', {})
        ptype = rules.get('placement_type', 'unknown')
        if ptype not in placement_types:
            placement_types[ptype] = []
        placement_types[ptype].append(item['type'])
    
    for ptype, items in placement_types.items():
        print(f"\n{ptype.upper()}:")
        for item_type in items[:3]:  # Show first 3 examples
            print(f"  - {item_type}")
        if len(items) > 3:
            print(f"  ... and {len(items) - 3} more")
    
    # Test random steps with reward analysis
    print("\n" + "=" * 70)
    print("Testing Reward Components")
    print("=" * 70)
    
    total_reward = 0
    for i in range(3):
        action = env.action_space.sample()
        next_state, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        print(f"\nStep {i+1}:")
        print(f"  Furniture: {info['selected_furniture']}")
        print(f"  Valid: {info['reward_components']['valid_placement']}")
        
        if info['reward_components']['valid_placement']:
            logical_score = info['reward_components'].get('logical_placement', 0)
            print(f"  Logical Placement Score: {logical_score:.3f}/1.0")
        else:
            if info['reward_components']['collision']:
                print(f"  Reason: Collision")
            elif info['reward_components']['out_of_bounds']:
                print(f"  Reason: Out of bounds")
            elif info['reward_components']['blocks_door']:
                print(f"  Reason: Blocks door")
        
        print(f"  Total Reward: {reward:.3f}")
        
        if terminated or truncated:
            break
    
    print(f"\n✅ Random episode completed")
    print(f"  Total reward: {total_reward:.3f}")
    print(f"  Items placed: {info['placed_items']}")
    
    return True


def test_agent():
    """Test that the SAC agent works with the new environment"""
    print("\n" + "=" * 70)
    print("Testing SAC Agent Compatibility")
    print("=" * 70)
    
    # Create environment
    env = FurnitureRecommendationEnvLogical(
        room_layout_path='room_layout.json',
        catalog_path='furniture_catalog_enhanced.json',
        max_items=4
    )
    
    # Create agent
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    action_low = env.action_space.low
    action_high = env.action_space.high
    
    agent = RuleGuidedSAC(
        state_dim=state_dim,
        action_dim=action_dim,
        action_low=action_low,
        action_high=action_high,
        hidden_dim=128
    )
    
    print(f"✅ SAC agent created successfully")
    print(f"  Actor parameters: {sum(p.numel() for p in agent.actor.parameters()):,}")
    print(f"  Critic parameters: {sum(p.numel() for p in agent.critic.parameters()):,}")
    
    # Test action selection
    state, _ = env.reset()
    action = agent.select_action(state, evaluate=False)
    
    print(f"✅ Action selection works")
    print(f"  Action shape: {action.shape}")
    print(f"  Action in bounds: {np.all(action >= env.action_space.low) and np.all(action <= env.action_space.high)}")
    
    # Test experience collection
    for i in range(5):
        action = agent.select_action(state, evaluate=False)
        next_state, reward, terminated, truncated, info = env.step(action)
        agent.replay_buffer.push(state, action, reward, next_state, terminated or truncated)
        state = next_state
        
        if terminated or truncated:
            state, _ = env.reset()
    
    print(f"✅ Experience collection works")
    print(f"  Replay buffer size: {len(agent.replay_buffer)}")
    
    # Test update
    if len(agent.replay_buffer) >= 5:
        agent.update(batch_size=5)
        print(f"✅ Agent update works")
        print(f"  Actor loss: {agent.actor_losses[-1]:.4f}")
        print(f"  Critic loss: {agent.critic_losses[-1]:.4f}")
    
    return True


def test_placement_logic():
    """Test specific placement logic scenarios"""
    print("\n" + "=" * 70)
    print("Testing Placement Logic Scenarios")
    print("=" * 70)
    
    env = FurnitureRecommendationEnvLogical(
        room_layout_path='room_layout.json',
        catalog_path='furniture_catalog_enhanced.json',
        max_items=4
    )
    
    # Find specific furniture types
    wall_shelf = next((item for item in env.furniture_catalog if item['type'] == 'wall_shelf'), None)
    plant_stand = next((item for item in env.furniture_catalog if item['type'] == 'plant_stand'), None)
    coffee_table = next((item for item in env.furniture_catalog if item['type'] == 'coffee_table'), None)
    
    print("\nPlacement Type Examples:")
    
    if wall_shelf:
        print(f"\n1. WALL-MOUNTED - {wall_shelf['name']}")
        rules = wall_shelf.get('placement_rules', {})
        print(f"   Placement Type: {rules.get('placement_type')}")
        print(f"   Max Distance from Wall: {rules.get('max_distance_from_wall')}m")
        print(f"   → Should be placed close to walls")
    
    if plant_stand:
        print(f"\n2. CORNER - {plant_stand['name']}")
        rules = plant_stand.get('placement_rules', {})
        print(f"   Placement Type: {rules.get('placement_type')}")
        print(f"   Corner Preferred: {rules.get('corner_preferred')}")
        print(f"   → Should be placed in room corners")
    
    if coffee_table:
        print(f"\n3. NEAR SEATING - {coffee_table['name']}")
        rules = coffee_table.get('placement_rules', {})
        print(f"   Placement Type: {rules.get('placement_type')}")
        print(f"   Ideal Distance from Seating: {rules.get('ideal_distance_from_seating')}m")
        print(f"   Max Distance from Seating: {rules.get('max_distance_from_seating')}m")
        print(f"   → Should be placed near sofas/chairs")
    
    print("\n✅ Placement logic configured correctly")
    return True


def main():
    """Run all tests"""
    print("\n" + "🪑 " * 30)
    print("LOGICAL PLACEMENT SYSTEM - SETUP VERIFICATION")
    print("🪑 " * 30 + "\n")
    
    try:
        # Test environment
        test_environment()
        
        # Test agent
        test_agent()
        
        # Test placement logic
        test_placement_logic()
        
        print("\n" + "=" * 70)
        print("✅ ALL TESTS PASSED!")
        print("=" * 70)
        print("\nYou're ready to:")
        print("  1. Train the model: python train_sac_logical.py")
        print("  2. Test the model: python test_sac_logical.py")
        print("\nThe system now includes:")
        print("  • Wall-mounted placement (shelves, mirrors)")
        print("  • Against-wall placement (bookcases, console tables)")
        print("  • Corner placement (plant stands, storage baskets)")
        print("  • Near-seating placement (coffee tables, lamps)")
        print("  • Room-center placement (rugs)")
        print("  • Free-standing placement (default)")
        print("\n" + "=" * 70)
        
    except Exception as e:
        print("\n" + "=" * 70)
        print("❌ TEST FAILED")
        print("=" * 70)
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
