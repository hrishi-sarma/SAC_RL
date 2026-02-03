"""
Quick Demo Script - Verify Environment Setup
"""

import numpy as np
from furniture_env import FurnitureRecommendationEnv
from sac_agent import RuleGuidedSAC


def test_environment():
    """Test that the environment works correctly"""
    print("=" * 60)
    print("Testing Furniture Recommendation Environment")
    print("=" * 60)
    
    # Create environment
    env = FurnitureRecommendationEnv(
        room_layout_path='room_layout.json',
        catalog_path='furniture_catalog.json',
        max_items=4
    )
    
    print(f"✓ Environment created successfully")
    print(f"  State dimension: {env.observation_space.shape[0]}")
    print(f"  Action dimension: {env.action_space.shape[0]}")
    print(f"  Max items: {env.max_items}")
    
    # Test reset
    state, info = env.reset()
    print(f"✓ Environment reset successfully")
    print(f"  Initial state shape: {state.shape}")
    
    # Test random steps
    total_reward = 0
    for i in range(3):
        action = env.action_space.sample()
        next_state, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        print(f"✓ Step {i+1} completed")
        print(f"  Reward: {reward:.3f}")
        print(f"  Valid placement: {info['reward_components']['valid_placement']}")
        
        if terminated or truncated:
            break
    
    print(f"\n✓ Random episode completed")
    print(f"  Total reward: {total_reward:.3f}")
    print(f"  Items placed: {info['placed_items']}")
    
    return True


def test_agent():
    """Test that the SAC agent works correctly"""
    print("\n" + "=" * 60)
    print("Testing Rule-Guided SAC Agent")
    print("=" * 60)
    
    # Create simple environment
    env = FurnitureRecommendationEnv(
        room_layout_path='room_layout.json',
        catalog_path='furniture_catalog.json',
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
        hidden_dim=128  # Smaller for demo
    )
    
    print(f"✓ SAC agent created successfully")
    print(f"  Actor parameters: {sum(p.numel() for p in agent.actor.parameters()):,}")
    print(f"  Critic parameters: {sum(p.numel() for p in agent.critic.parameters()):,}")
    
    # Test action selection
    state, _ = env.reset()
    action = agent.select_action(state, evaluate=False)
    
    print(f"✓ Action selection works")
    print(f"  Action shape: {action.shape}")
    print(f"  Action bounds valid: {np.all(action >= env.action_space.low) and np.all(action <= env.action_space.high)}")
    
    # Test experience collection
    for i in range(5):
        action = agent.select_action(state, evaluate=False)
        next_state, reward, terminated, truncated, info = env.step(action)
        agent.replay_buffer.push(state, action, reward, next_state, terminated or truncated)
        state = next_state
        
        if terminated or truncated:
            state, _ = env.reset()
    
    print(f"✓ Experience collection works")
    print(f"  Replay buffer size: {len(agent.replay_buffer)}")
    
    # Test update (with small batch)
    if len(agent.replay_buffer) >= 5:
        agent.update(batch_size=5)
        print(f"✓ Agent update works")
        print(f"  Actor loss: {agent.actor_losses[-1]:.4f}")
        print(f"  Critic loss: {agent.critic_losses[-1]:.4f}")
    
    return True


def test_data_files():
    """Test that JSON data files are valid"""
    print("\n" + "=" * 60)
    print("Testing Data Files")
    print("=" * 60)
    
    import json
    
    # Test room layout
    with open('room_layout.json', 'r') as f:
        room_data = json.load(f)
    
    print(f"✓ Room layout loaded")
    print(f"  Room type: {room_data['room_info']['room_type']}")
    print(f"  Dimensions: {room_data['room_info']['dimensions']['length']}m × {room_data['room_info']['dimensions']['width']}m")
    print(f"  Existing furniture: {len(room_data['existing_furniture'])} items")
    print(f"  Budget: ${room_data['constraints']['budget_remaining']}")
    
    # Test catalog
    with open('furniture_catalog.json', 'r') as f:
        catalog_data = json.load(f)
    
    print(f"✓ Furniture catalog loaded")
    print(f"  Total items: {len(catalog_data['furniture_items'])}")
    
    categories = {}
    for item in catalog_data['furniture_items']:
        cat = item['category']
        categories[cat] = categories.get(cat, 0) + 1
    
    print(f"  Categories:")
    for cat, count in categories.items():
        print(f"    - {cat}: {count} items")
    
    return True


def main():
    """Run all tests"""
    print("\n" + "🏠" * 30)
    print("FURNITURE RECOMMENDATION SYSTEM - SETUP VERIFICATION")
    print("🏠" * 30 + "\n")
    
    try:
        # Test data files first
        test_data_files()
        
        # Test environment
        test_environment()
        
        # Test agent
        test_agent()
        
        print("\n" + "=" * 60)
        print("✓ ALL TESTS PASSED!")
        print("=" * 60)
        print("\nYou're ready to:")
        print("  1. Train the model: python train_sac.py")
        print("  2. Test the model: python test_sac.py")
        print("\n" + "=" * 60)
        
    except Exception as e:
        print("\n" + "=" * 60)
        print("✗ TEST FAILED")
        print("=" * 60)
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
