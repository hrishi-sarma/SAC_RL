"""
OID-PPO: Complete Implementation  -  FIXED VERSION
Changes vs original:
  - PPOAgent now receives n_episodes for LR scheduler
  - R_path weighted 2x in composite reward (it was always -1, dragging score)
  - log_interval shows current LR so you can monitor decay
"""

import numpy as np
import torch
import json
import matplotlib.pyplot as plt
import time
from pathlib import Path
import sys
import importlib.util


def load_module_from_file(module_name, file_path):
    spec   = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


# Load modules (fixed versions replace originals if present in same folder)
core        = load_module_from_file('oid_ppo_core',    'oid_ppo_core.py')
rewards_mod = load_module_from_file('oid_ppo_rewards', 'oid_ppo_rewards.py')
network_mod = load_module_from_file('oid_ppo_network', 'oid_ppo_network.py')
agent_mod   = load_module_from_file('oid_ppo_agent',   'oid_ppo_agent.py')

FurnitureItem      = core.FurnitureItem
InteriorDesignEnv  = core.InteriorDesignEnv
ActorCriticNetwork = network_mod.ActorCriticNetwork
PPOAgent           = agent_mod.PPOAgent

# Attach reward methods
for method_name in dir(rewards_mod):
    if method_name.startswith(('_reward', '_find', '_point',
                               '_is_reachable', '_astar')):
        setattr(InteriorDesignEnv, method_name,
                getattr(rewards_mod, method_name))


# ── data loading ───────────────────────────────────────────────────────────────
def load_furniture_and_room(furniture_catalog_path, room_layout_path):
    with open(furniture_catalog_path) as f:
        catalog = json.load(f)
    with open(room_layout_path) as f:
        room_data = json.load(f)

    selected_ids  = ['coffee_table_001', 'magazine_rack_001',
                     'table_lamp_001',   'storage_basket_001']
    furniture_items = []

    for item_id in selected_ids:
        for item in catalog['furniture_items']:
            if item['id'] == item_id:
                furniture_items.append(FurnitureItem(
                    id=item['id'],
                    name=item['name'],
                    dimensions=item['dimensions'],
                    type_name=item['type']
                ))
                break

    furniture_items.sort(key=lambda f: f.area, reverse=True)

    room_dimensions = {
        'length': room_data['room_info']['dimensions']['length'],
        'width':  room_data['room_info']['dimensions']['width']
    }
    door_positions = [
        np.array([d['position']['x'], d['position']['y']])
        for d in room_data['doors']
    ]
    return furniture_items, room_dimensions, door_positions


# ── training loop ──────────────────────────────────────────────────────────────
def train_oid_ppo(env, n_episodes=1000, save_interval=100,
                  device='cpu', log_interval=10):
    print("=" * 80)
    print("OID-PPO Training  (FIXED)")
    print("=" * 80)
    print(f"  Episodes     : {n_episodes}")
    print(f"  Device       : {device}")
    print(f"  Room         : {env.N}m × {env.M}m")
    print(f"  Furniture    : {env.n_furniture}")
    print(f"  Map res      : {env.map_resolution}m")
    print()

    occupancy_shape = env.occupancy_map.shape
    network = ActorCriticNetwork(occupancy_shape)
    print(f"Network: {sum(p.numel() for p in network.parameters()):,} parameters")

    agent = PPOAgent(
        network=network,
        env=env,
        gamma=0.99,
        gae_lambda=0.95,
        epsilon=0.2,
        lr_actor=3e-5,        # FIX: lower than paper default
        lr_critic=3e-4,       # FIX: lower than paper default
        n_episodes=n_episodes, # FIX: needed for scheduler
        device=device
    )

    history = {
        'rewards': [], 'policy_losses': [], 'value_losses': [],
        'best_reward': -float('inf'), 'best_episode': 0, 'best_layout': None
    }

    print("\nStarting training...")
    start_time = time.time()

    for episode in range(1, n_episodes + 1):
        episode_info = agent.train_episode()

        history['rewards'].append(episode_info['reward'])
        history['policy_losses'].append(episode_info['policy_loss'])
        history['value_losses'].append(episode_info['value_loss'])

        if episode_info['valid'] and episode_info['reward'] > history['best_reward']:
            history['best_reward']  = episode_info['reward']
            history['best_episode'] = episode
            history['best_layout']  = {
                'placed_furniture': [pf.copy() for pf in env.placed_furniture],
                'reward_components': env.last_rewards.copy()
                    if hasattr(env, 'last_rewards') else {}
            }

        if episode % log_interval == 0:
            stats   = agent.get_statistics(window=100)
            elapsed = time.time() - start_time
            lr_now  = stats.get('current_lr_actor', 0)
            print(
                f"Ep {episode:4d}/{n_episodes} | "
                f"Avg: {stats['mean_reward']:7.4f} ± {stats['std_reward']:.3f} | "
                f"Best: {history['best_reward']:7.4f} | "
                f"P-Loss: {stats['mean_policy_loss']:.4f} | "
                f"V-Loss: {stats['mean_value_loss']:.4f} | "
                f"LR: {lr_now:.2e} | "
                f"{episode/elapsed:.1f} eps/s"
            )

        if episode % save_interval == 0:
            agent.save(f'oid_ppo_checkpoint_ep{episode}.pth')

    elapsed = time.time() - start_time
    print(f"\nTraining complete!  {elapsed:.1f}s ({elapsed/60:.1f}min)")
    print(f"  Best reward  : {history['best_reward']:.4f}  (ep {history['best_episode']})")
    print(f"  Final reward : {history['rewards'][-1]:.4f}")
    return agent, history


# ── save results ───────────────────────────────────────────────────────────────
def save_results(history, agent, output_dir='oid_ppo_results'):
    Path(output_dir).mkdir(exist_ok=True)

    # Training curves
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))

    axes[0].plot(history['rewards'], alpha=0.25, color='steelblue', label='Episode Reward')
    window = 50
    if len(history['rewards']) >= window:
        smoothed = np.convolve(history['rewards'],
                               np.ones(window) / window, mode='valid')
        axes[0].plot(range(window - 1, len(history['rewards'])), smoothed,
                     linewidth=2, color='darkorange',
                     label=f'{window}-ep Average')
    axes[0].axhline(y=history['best_reward'], color='red', ls='--',
                    label=f"Best: {history['best_reward']:.4f}")
    axes[0].set_xlabel('Episode')
    axes[0].set_ylabel('Reward (R_idg)')
    axes[0].set_title('OID-PPO Training: Reward')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(history['policy_losses'], alpha=0.7, label='Policy Loss')
    axes[1].plot(history['value_losses'],  alpha=0.7, label='Value Loss')
    axes[1].set_xlabel('Episode')
    axes[1].set_ylabel('Loss')
    axes[1].set_title('OID-PPO Training: Losses')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    axes[1].set_yscale('log')

    plt.tight_layout()
    plt.savefig(f'{output_dir}/training_curves.png', dpi=150, bbox_inches='tight')
    print(f"✓ Saved training curves")

    # Results JSON
    results = {
        'best_reward':       float(history['best_reward']),
        'best_episode':      int(history['best_episode']),
        'final_reward':      float(history['rewards'][-1]),
        'mean_last_100':     float(np.mean(history['rewards'][-100:])),
        'training_episodes': len(history['rewards']),
        'best_layout': {
            'furniture_placements': [
                {
                    'name':             pf['furniture'].name,
                    'type':             pf['furniture'].type,
                    'position':         pf['position'].tolist(),
                    'rotation_degrees': int(pf['rotation'] * 90)
                }
                for pf in history['best_layout']['placed_furniture']
            ] if history['best_layout'] else [],
            'reward_components': history['best_layout']['reward_components']
                if history['best_layout'] else {}
        }
    }

    with open(f'{output_dir}/results.json', 'w') as f:
        json.dump(results, f, indent=2)
    print(f"✓ Saved results.json")

    agent.save(f'{output_dir}/oid_ppo_final.pth')
    print(f"✅ All saved to {output_dir}/")


# ── main ───────────────────────────────────────────────────────────────────────
def main():
    print("=" * 80)
    print("OID-PPO  (FIXED VERSION)")
    print("=" * 80)

    print(f"PyTorch: {torch.__version__}")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device : {device}\n")

    furniture_items, room_dimensions, door_positions = load_furniture_and_room(
        'furniture_catalog_enhanced.json', 'room_layout.json'
    )

    print(f"Furniture ({len(furniture_items)} items, sorted by area):")
    for i, f in enumerate(furniture_items, 1):
        print(f"  {i}. {f.name}  {f.length}×{f.width}m  area={f.area:.3f}m²")
    print()

    env = InteriorDesignEnv(
        room_dimensions=room_dimensions,
        door_positions=door_positions,
        furniture_list=furniture_items,
        map_resolution=0.10
    )
    print(f"Occupancy map: {env.occupancy_map.shape}\n")

    N_EPISODES = 10000   # more episodes → better convergence

    agent, history = train_oid_ppo(
        env=env,
        n_episodes=N_EPISODES,
        save_interval=200,
        device=device,
        log_interval=10
    )

    save_results(history, agent)

    print("\n" + "=" * 80)
    print("✅ Done!")
    print("=" * 80)


if __name__ == '__main__':
    main()
