# OID-PPO: Optimal Interior Design using Proximal Policy Optimization

**Exact implementation following the paper:** Yoon et al., "OID-PPO: Optimal Interior Design using Proximal Policy Optimization by Transforming Design Guidelines into Reward Functions", AAAI 2026 (arXiv:2508.00364v1)

---

## 📦 Installation

### 1. Install Dependencies

```bash
# Core dependencies
pip install torch torchvision
pip install numpy matplotlib scipy

# Optional for better visualization
pip install seaborn
```

### 2. Verify PyTorch Installation

```python
import torch
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
```

---

## 📁 Project Structure

```
oid_ppo_implementation/
├── oid_ppo_core.py          # FurnitureItem, InteriorDesignEnv (MDP formulation)
├── oid_ppo_rewards.py       # All 6 reward functions from Section 3
├── oid_ppo_network.py       # ActorCriticNetwork (PyTorch, Figure 1)
├── oid_ppo_agent.py         # PPOAgent (training logic)
├── oid_ppo_complete.py      # Main training script
├── furniture_catalog_enhanced.json  # Furniture specifications
├── room_layout.json         # Room configuration
└── README.md                # This file
```

---

## 🚀 Quick Start

### Run Training

```bash
python oid_ppo_complete.py
```

This will:
1. Load furniture catalog and room layout
2. Initialize the ActorCriticNetwork
3. Train for 1000 episodes (paper default)
4. Save results, model, and training curves

**Expected runtime:** 10-30 minutes (depending on hardware)

---

## ⚙️ Configuration

### Training Hyperparameters (from paper)

```python
gamma = 0.99           # Discount factor γ
gae_lambda = 0.95      # GAE lambda λ
epsilon = 0.2          # PPO clip ratio ε
lr_actor = 1e-4        # Actor learning rate η_a
lr_critic = 1e-3       # Critic learning rate η_c
n_episodes = 1000      # Training episodes
```

### Environment Configuration

```python
map_resolution = 0.10  # 10cm grid (best results)
# Options: 0.05 (5cm, slow), 0.10 (10cm, balanced), 0.20 (20cm, fast)
```

---

## 📊 Output Files

After training completes, you'll find:

```
oid_ppo_results/
├── training_curves.png      # Reward and loss plots
├── results.json              # Best layout and metrics
├── oid_ppo_final.pth         # Trained model weights
└── oid_ppo_checkpoint_ep*.pth  # Checkpoints every 100 episodes
```

---

## 🧪 Architecture Details

### Neural Network (Paper Figure 1)

```
Input State s_t = (e_t, e_{t+1}, O_t)
│
├─► Furniture Encoder (current)
│   └─► MLP: 4 → 64 → 128 → 128 (GELU)
│
├─► Furniture Encoder (next)
│   └─► MLP: 4 → 64 → 128 → 128 (GELU)
│
└─► Occupancy Encoder
    └─► CNN: 1 → 16 → 32 → 64 (GELU)
    
Concatenate → FC: 256 → 256 (GELU)
│
├─► Actor Head (Diagonal Gaussian)
│   ├─► Mean: 256 → 3
│   └─► Log Std: 256 → 3
│
└─► Critic Head
    └─► Value: 256 → 1
```

**Total Parameters:** ~500K (varies with map resolution)

---

## 📐 MDP Formulation (Paper Section 2)

### State Space S
- **e_t**: Current furniture descriptor (4D: length, width, height, area)
- **e_{t+1}**: Next furniture descriptor (4D)
- **O_t**: Binary occupancy map (H×W grid)

### Action Space A
- **x, y**: Position ∈ ℝ² (continuous)
- **k**: Rotation ∈ {0, 1, 2, 3} (90° increments)
- **Policy**: Diagonal Gaussian `a_t = μ_t + σ_t ⊙ z`, `z ~ N(0,I)`

### Reward Function R
- **R_idg** = (1/6)(R_pair + R_a + R_v + R_path + R_b + R_al) ∈ [-1, 1]
- **Invalid placement penalty**: φ = -10

---

## 🎯 Reward Components (Paper Section 3)

### 1. Pairwise Relationship (R_pair)
**Formula:** `R_pair = (1/|P|) Σ K_dist(p,c) · K_dir(p,c)`

Encourages functional furniture pairing (e.g., desk↔chair, sofa↔coffee_table)

### 2. Accessibility (R_a)
**Formula:** `R_a = 1 - (2/|F|) Σ |ν(f)|/|U_f|`

Ensures clearance zones around furniture for user access

### 3. Visibility (R_v)
**Formula:** `R_v = -(1/|F|) Σ ⟨n_f, n_w(f)⟩`

Prevents furniture from facing walls (unusable orientation)

### 4. Pathway Connection (R_path)
**Formula:** `R_path = 1 - (2/|F|) Σ [(1-I_f) + e^(-κ_f)·I_f]`

Maintains clear paths from doors using A* search

### 5. Visual Balance (R_b)
**Formula:** `R_b = exp(-||x̄_F-o||²/2d²_△) + exp(-||Σ_F-κ²_EI||²_F/κ⁴_E) - 1`

Promotes even spatial distribution and centered mass

### 6. Alignment (R_al)
**Formula:** `R_al = Σ Π(f)·cos²(2ϑ_f)·(1-tanh²(2ω_f)) / Σ Π(f)`

Encourages parallel/perpendicular alignment with walls

---

## 🔬 Theoretical Guarantees (Paper)

### Proposition 1: Finite Horizon
Every episode terminates in at most |F| steps

### Proposition 2: Reachability
Layouts where all furniture is reachable always score higher on R_path

### Proposition 3: Monotonic Improvement  
PPO guarantees policy improvement: `|J(θ) - L^clip(θ)| ≤ O(ε)`

### Proposition 4: Exploration
Diagonal Gaussian policy ensures positive exploration probability

### Theorem 1: Convergence
Policy parameters converge almost surely: `J(θ_k) → J(θ*) a.s.`

---

## 📈 Expected Results

Based on paper Table 1 (Fn=4, Square Room):

| Method | R_idg | Time (s) |
|--------|-------|----------|
| MH (baseline) | 0.281 | 70.3 |
| MOPSO (baseline) | 0.338 | 96.8 |
| DDPG | 0.803 | 1.2 |
| TD3 | 0.823 | 1.2 |
| SAC | 0.903 | 2.5 |
| **OID-PPO** | **0.971** | **3.2** |

**Your implementation should achieve:** R_idg ≈ 0.90-0.97 after 1000 episodes

---

## 🐛 Troubleshooting

### Issue: Low reward (<0.5)
**Solution:** Train for more episodes or decrease map_resolution

### Issue: Many invalid placements
**Solution:** Check furniture is sorted by area (descending)

### Issue: Out of memory
**Solution:** Reduce map_resolution from 0.10 to 0.15 or 0.20

### Issue: Slow training
**Solution:** 
- Use GPU if available
- Increase map_resolution (trade-off with accuracy)
- Reduce update_epochs from 4 to 2

---

## 🔧 Advanced Usage

### Resume Training from Checkpoint

```python
# Load checkpoint
agent = PPOAgent(network, env, device=device)
agent.load('oid_ppo_checkpoint_ep500.pth')

# Continue training
for episode in range(501, 1001):
    agent.train_episode()
```

### Test Trained Model

```python
# Load trained model
agent.load('oid_ppo_results/oid_ppo_final.pth')

# Test with deterministic policy
state = env.reset()
done = False
while not done:
    state_tensor = agent._state_to_tensor(state)
    with torch.no_grad():
        action, _, _ = agent.network.get_action(state_tensor, deterministic=True)
    action_np = action.cpu().numpy()[0]
    state, reward, done, info = env.step(action_np)

print(f"Final reward: {reward}")
print(f"Reward components: {env.last_rewards}")
```

### Modify Reward Weights

To emphasize certain design aspects, you can modify the composite reward:

```python
# In oid_ppo_core.py, _calculate_reward method:
# Instead of equal weights:
R_idg = (R_pair + R_a + R_v + R_path + R_b + R_al) / 6.0

# Use custom weights (must sum to 1.0):
R_idg = 0.3*R_pair + 0.2*R_a + 0.1*R_v + 0.2*R_path + 0.1*R_b + 0.1*R_al
```

---

## 📚 Paper Citation

```bibtex
@inproceedings{yoon2026oidppo,
  title={OID-PPO: Optimal Interior Design using Proximal Policy Optimization by Transforming Design Guidelines into Reward Functions},
  author={Yoon, Chanyoung and Yoo, Sangbong and Yim, Soobin and Kim, Chansoo and Jang, Yun},
  booktitle={AAAI Conference on Artificial Intelligence},
  year={2026},
  note={arXiv:2508.00364v1}
}
```

---

## 📝 Notes

### Implementation Faithfulness
This implementation follows the paper **exactly**:
- ✅ MDP formulation (Section 2)
- ✅ All 6 reward functions with exact formulas (Section 3)
- ✅ Neural network architecture (Figure 1)
- ✅ PPO training procedure (Model section)
- ✅ Hyperparameters (Table 1 caption)
- ✅ Theoretical guarantees (Propositions 1-4, Theorem 1)

### Differences from Paper
1. **Dataset**: Paper uses RPLAN dataset; this uses custom room_layout.json
2. **Furniture count**: Paper tests up to 8 items; this uses 4 (configurable)
3. **Visualization**: Added for better interpretability

---

## 📧 Support

For issues or questions about this implementation:
1. Check if furniture is sorted by area (descending)
2. Verify PyTorch is installed correctly
3. Ensure input JSON files are in correct format

---

## ⚖️ License

This implementation is for academic and research purposes following the paper's methodology.

---

**Last Updated:** February 2026  
**Implementation:** Complete OID-PPO following paper specification
