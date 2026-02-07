"""
Test script for improved model (relaxed constraints)
"""

from test_sac_semantic_FIXED import FurnitureRecommendationTesterSemantic
import glob
import os

def find_improved_model():
    """Find the improved/relaxed model"""
    # Look for improved models
    models = glob.glob('models_improved/sac_improved_epfinal.pt')
    if models:
        return models[0]
    
    # Otherwise get latest checkpoint
    models = glob.glob('models_improved/sac_improved_ep*.pt')
    if models:
        # Sort by episode number
        models = [m for m in models if 'final' not in m]
        if models:
            try:
                episodes = [int(m.split('ep')[1].split('.')[0]) for m in models]
                latest_idx = episodes.index(max(episodes))
                return models[latest_idx]
            except (ValueError, IndexError):
                return models[-1]
    
    # Fallback: try semantic models
    models = glob.glob('models_semantic/sac_semantic_epfinal.pt')
    if models:
        print("⚠️  No improved model found, using semantic model")
        return models[0]
    
    models = glob.glob('models_semantic/sac_semantic_ep*.pt')
    if models:
        print("⚠️  No improved model found, using semantic checkpoint")
        try:
            models = [m for m in models if 'final' not in m]
            episodes = [int(m.split('ep')[1].split('.')[0]) for m in models]
            latest_idx = episodes.index(max(episodes))
            return models[latest_idx]
        except (ValueError, IndexError):
            return models[-1]
    
    return None

def main():
    model_path = find_improved_model()
    
    if not model_path:
        print("❌ No model found!")
        print("Please train first:")
        print("  python train_improved.py")
        print("  OR")
        print("  python train_sac_semantic.py")
        return
    
    print(f"✅ Found model: {model_path}")
    
    # Use relaxed constraints for testing
    tester = FurnitureRecommendationTesterSemantic(
        room_layout_path='room_layout.json',
        catalog_path='furniture_catalog.json',
        model_path=model_path,
        max_items=4,
        grid_size=0.4,              # Larger grid for more options
        collision_buffer=0.12,       # Smaller buffer to fit more
        wall_proximity=0.60          # Very relaxed wall proximity
    )
    
    print("\n" + "🏠" * 30)
    print("TESTING IMPROVED MODEL (RELAXED CONSTRAINTS)")
    print("🏠" * 30 + "\n")
    
    # Test single episode
    tester.test_episode(visualize=True, verbose=True)
    
    # Batch test
    print("\n" + "=" * 70)
    print("Running batch test...")
    print("=" * 70)
    tester.batch_test(num_episodes=20)
    
    print("\n" + "=" * 70)
    print("✅ TESTING COMPLETE!")
    print("=" * 70)
    print("Check the output files:")
    print("  - recommendation_layout_semantic.png")
    print("  - reward_breakdown_semantic.png")
    print("  - batch_test_results_semantic.png")
    print("=" * 70)

if __name__ == '__main__':
    main()