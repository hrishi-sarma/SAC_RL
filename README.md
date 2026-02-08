# Furniture Recommendation System with Logical Placement

An AI-powered furniture recommendation system using Reinforcement Learning (SAC) with **intelligent placement rules** that understand furniture context and placement logic.

## 🎯 What's New: Logical Placement System

This enhanced version implements **rule-based logical placement** that teaches the AI where different types of furniture should naturally go:

### Placement Types

| Type | Examples | Placement Logic |
|------|----------|----------------|
| **Wall-Mounted** | Shelves, Mirrors | Must be within 0.1m of walls |
| **Against Wall** | Bookcases, Console Tables, Benches | Should be within 0.2-0.3m of walls |
| **Corner** | Plant Stands, Bar Carts, Storage Baskets | Prefers corner zones (1m radius from corners) |
| **Near Seating** | Coffee Tables, Side Tables, Lamps, Ottomans | Should be 0.5-2.0m from sofas/chairs |
| **Room Center** | Area Rugs | Prefers central placement |
| **Free Standing** | Default | Avoids walls (>0.5m clearance) |

### How It Works

Instead of the agent learning placement from scratch, we provide **structured guidance** through:

1. **Metadata in Catalog**: Each furniture item has `placement_rules` specifying its ideal placement
2. **Logical Placement Reward**: Highest-weighted reward component (3.0x) that rewards correct placement
3. **Distance-Based Scoring**: Calculates distance to walls, corners, or seating and rewards appropriately

**Example**: A wall shelf gets maximum reward when placed within 0.1m of a wall, zero reward if placed in the room center.

## 📁 File Structure

```
.
├── furniture_catalog_enhanced.json    # Enhanced catalog with placement rules
├── room_layout.json                   # Room configuration
├── furniture_env_logical.py           # Environment with logical placement
├── sac_agent.py                       # SAC algorithm implementation
├── train_sac_logical.py              # Training script
├── test_sac_logical.py               # Testing & visualization script
├── demo_logical.py                    # Setup verification
└── requirements.txt                   # Python dependencies
```

## 🚀 Quick Start

### 1. Setup Environment

```bash
pip install -r requirements.txt
```

### 2. Verify Setup

```bash
python demo_logical.py
```

This will:
- ✅ Test environment initialization
- ✅ Verify placement rules are loaded
- ✅ Check SAC agent compatibility
- ✅ Show example placement logic

### 3. Train the Model

```bash
python train_sac_logical.py
```

**Training Parameters**:
- Episodes: 1000
- Batch Size: 256
- Grid Size: 0.3m (30cm alignment)
- Collision Buffer: 0.2m (20cm minimum spacing)

**Expected Training Time**: ~30-60 minutes on CPU, ~10-15 minutes on GPU

**What to Expect**:
- Early episodes: Random exploration, many collisions
- Mid training (300-500): Learns to avoid collisions, starts understanding placement rules
- Late training (700-1000): Consistent logical placement, high success rate

### 4. Test the Trained Model

```bash
python test_sac_logical.py
```

**Outputs**:
- `recommendation_layout_logical.png` - Visual layout with placed furniture
- `reward_breakdown_logical.png` - Analysis of reward components
- `batch_test_results_logical.png` - Statistics from 20 test episodes

## 🎨 Understanding the Outputs

### Layout Visualization
- **Purple dashed circles**: Corner zones where corner-preferred items should go
- **Grid lines**: 0.3m grid for organized placement
- **Brown arcs**: Door clearance zones
- **Blue rectangles**: Windows
- **Colored furniture**: Color-coded by category
  - 🔴 Red: Seating
  - 🔵 Teal: Tables
  - 🟦 Blue: Storage
  - 🟠 Orange: Lighting
  - 🟢 Green: Decor

### Reward Breakdown
Shows average values of each reward component:
- **logical_placement** (purple border): The new placement logic reward (should be high!)
- **diversity**: Rewards different furniture types
- **accessibility**: Ensures proper clearance
- **color_harmony**: Matches room color scheme
- **functional_pairing**: Places items near complementary furniture

## 📊 Key Metrics

After training, you should see:
- **Success Rate**: 90-100% (places all 4 items successfully)
- **Logical Placement Score**: 0.7-0.9 (0=poor, 1=perfect placement)
- **Valid Placement Rate**: 95-100% (minimal collisions)
- **Zero Collision Episodes**: 18-20 out of 20 test episodes

## 🔧 Customization

### Adjust Placement Rules

Edit `furniture_catalog_enhanced.json`:

```json
{
  "id": "my_item",
  "placement_rules": {
    "placement_type": "corner",           // Type of placement
    "corner_preferred": true,             // Must be in corner?
    "ideal_distance_from_wall": 0.3,     // Target distance
    "max_distance_from_wall": 0.8,       // Maximum allowed
    "min_distance_from_wall": 0.2        // Minimum allowed
  }
}
```

### Modify Reward Weights

In `furniture_env_logical.py`, adjust the weights in `_calculate_reward()`:

```python
weights = {
    'logical_placement': 3.0,    # Increase for stricter placement rules
    'functional_pairing': 1.0,
    'accessibility': 1.5,
    # ... other components
}
```

### Change Room Configuration

Edit `room_layout.json`:
- Room dimensions
- Existing furniture
- Door/window positions
- Budget constraints

## 🧠 Technical Details

### State Representation (120-dimensional)
- Room features (6)
- Furniture encoding (80) - positions, sizes, categories
- Budget info (2)
- Free space (1)
- Step counter (1)
- Catalog availability mask (20)
- Spatial occupancy (10)

### Action Space (5-dimensional)
1. Furniture selection (catalog index)
2. X position (0-1, normalized)
3. Y position (0-1, normalized)
4. Rotation (0-1, discrete: 0°/90°/180°/270°)
5. Scale (0.8-1.2)

### Reward Components

| Component | Weight | Purpose |
|-----------|--------|---------|
| Logical Placement | 3.0 | **NEW**: Enforces placement rules |
| Accessibility | 1.5 | Maintains walkable space |
| Diversity | 1.5 | Encourages variety |
| Clearance | 1.2 | Pathway clearance |
| Functional Pairing | 1.0 | Items near complementary pieces |
| Visual Balance | 1.0 | Balanced room layout |
| Grid Alignment | 1.0 | Organized placement |
| Parallel Placement | 1.0 | Aligned with existing furniture |
| Color Harmony | 0.8 | Matches color scheme |
| Completeness | 0.8 | Rewards placing all items |
| Size Appropriateness | 0.8 | Right-sized furniture |
| Budget Efficiency | 0.5 | Efficient spending |

## 🆚 Comparison to Previous System

### Before (Generic RL)
- ❌ No understanding of furniture context
- ❌ Wall shelves could be placed anywhere
- ❌ Plant stands in the middle of the room
- ❌ Slow learning (needs to discover placement logic)

### After (Logical Placement)
- ✅ Built-in placement knowledge
- ✅ Wall-mounted items go near walls
- ✅ Corner items prefer corners
- ✅ Faster learning (guided by rules)
- ✅ More realistic, professional layouts

## 🐛 Troubleshooting

### Model doesn't learn logical placement
- **Check**: Are placement rules in the catalog?
- **Check**: Is `logical_placement` weight set to 3.0?
- **Solution**: Increase weight or reduce other weights

### Too many collisions
- **Increase**: `collision_buffer` in training script (try 0.25m)
- **Decrease**: Learning rate (try 1e-4)

### Items not using full room
- **Check**: Grid size (try 0.2m for finer placement)
- **Adjust**: Reward weights to reduce `visual_balance`

## 📈 Advanced Training

For better results:
```bash
# Longer training
python train_sac_logical.py --num_episodes 2000

# Stricter collision rules
python train_sac_logical.py --collision_buffer 0.25

# Finer grid
python train_sac_logical.py --grid_size 0.2
```

## 🤝 Contributing

To add new placement types:

1. Define in catalog: `"placement_type": "your_type"`
2. Implement logic in `_reward_logical_placement()` in `furniture_env_logical.py`
3. Add weight in `_calculate_reward()`

## 📚 References

- **SAC Algorithm**: [Soft Actor-Critic Paper](https://arxiv.org/abs/1801.01290)
- **Grid Alignment**: Inspired by interior design best practices
- **Placement Rules**: Based on residential interior design guidelines

## 📄 License

MIT License - Feel free to use and modify!

## ✨ Future Enhancements

- [ ] Multi-room support
- [ ] Style-specific placement rules (minimalist vs traditional)
- [ ] 3D visualization
- [ ] User preference learning
- [ ] Real-time recommendation API

---

**Made with ❤️ for smarter furniture placement**
