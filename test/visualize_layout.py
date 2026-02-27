"""
OID-PPO Layout Visualizer
Reads results.json + room_layout.json + furniture_catalog_enhanced.json
and produces a clean room layout plot.

Usage:
    python visualize_layout.py
    python visualize_layout.py --results oid_ppo_results/results.json
    python visualize_layout.py --results results.json --room room_layout.json --catalog furniture_catalog_enhanced.json
"""

import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Rectangle, FancyBboxPatch
from pathlib import Path


def load_json(path):
    with open(path) as f:
        return json.load(f)


def get_furniture_dims(catalog, ftype):
    for item in catalog['furniture_items']:
        if item['type'] == ftype:
            return item['dimensions']['length'], item['dimensions']['width']
    return 0.5, 0.5


def rotated_dims(length, width, rotation_deg):
    if rotation_deg % 180 == 90:
        return width, length
    return length, width


def visualize(results_path='oid_ppo_results/results.json',
              room_path='room_layout.json',
              catalog_path='furniture_catalog_enhanced.json',
              output_path='layout_output.png',
              show=True):

    results = load_json(results_path)
    room    = load_json(room_path)
    catalog = load_json(catalog_path)

    room_N = room['room_info']['dimensions']['length']
    room_M = room['room_info']['dimensions']['width']

    placements     = results['best_layout']['furniture_placements']
    reward_comps   = results['best_layout'].get('reward_components', {})
    best_reward    = results.get('best_reward', None)
    best_episode   = results.get('best_episode', '?')
    total_episodes = results.get('training_episodes', '?')

    existing_items = room.get('existing_furniture', [])
    doors          = room.get('doors', [])
    windows        = room.get('windows', [])
    zones          = room.get('available_space', {}).get('priority_zones', [])

    fig, ax = plt.subplots(figsize=(12, 10))

    # Room boundary
    ax.add_patch(Rectangle((0, 0), room_N, room_M,
                            fill=False, edgecolor='black', linewidth=3))

    # Priority zones
    zone_colors = {'high': 'orange', 'medium': 'yellow', 'low': 'lightgreen'}
    for zone in zones:
        ax.add_patch(plt.Circle(
            (zone['center']['x'], zone['center']['y']), zone['radius'],
            facecolor=zone_colors.get(zone.get('priority', 'medium'), 'gray'),
            alpha=0.15, edgecolor='none', zorder=0
        ))

    # Windows
    for win in windows:
        wx = win['position']['x']
        wy = win['position']['y']
        ww = win['dimensions']['width']

        if wy == 0.0:
            ax.add_patch(Rectangle((wx - ww/2, -0.1), ww, 0.2,
                         facecolor='lightblue', edgecolor='blue', linewidth=2, alpha=0.7, zorder=2))
            ax.text(wx, -0.35, 'Window', ha='center', fontsize=8, color='blue')
        elif wx >= room_N:
            ax.add_patch(Rectangle((room_N - 0.1, wy - ww/2), 0.2, ww,
                         facecolor='lightblue', edgecolor='blue', linewidth=2, alpha=0.7, zorder=2))
            ax.text(room_N + 0.15, wy, 'Window', ha='left', va='center', fontsize=8, color='blue')
        elif wy >= room_M:
            ax.add_patch(Rectangle((wx - ww/2, room_M - 0.1), ww, 0.2,
                         facecolor='lightblue', edgecolor='blue', linewidth=2, alpha=0.7, zorder=2))
            ax.text(wx, room_M + 0.15, 'Window', ha='center', fontsize=8, color='blue')
        else:
            ax.add_patch(Rectangle((-0.1, wy - ww/2), 0.2, ww,
                         facecolor='lightblue', edgecolor='blue', linewidth=2, alpha=0.7, zorder=2))
            ax.text(-0.15, wy, 'Window', ha='right', va='center', fontsize=8, color='blue')

    # Doors
    for door in doors:
        dx = door['position']['x']
        dy = door['position']['y']
        cr = door.get('clearance_radius', 1.2)
        ax.add_patch(plt.Circle((dx, dy), cr, fill=False, edgecolor='brown',
                                linestyle='--', linewidth=1.5, alpha=0.5, zorder=1))
        ax.add_patch(plt.Circle((dx, dy), 0.1, color='brown', zorder=10))
        ax.text(dx + 0.15, dy - 0.25, 'Door', ha='left', fontsize=8, color='brown')

    # Existing furniture
    for ef in existing_items:
        ex   = ef['position']['x']
        ey   = ef['position']['y']
        erot = ef.get('rotation', 0) * 90
        el_r, ew_r = rotated_dims(ef['dimensions']['length'], ef['dimensions']['width'], erot)
        ax.add_patch(FancyBboxPatch((ex - el_r/2, ey - ew_r/2), el_r, ew_r,
                     boxstyle='round,pad=0.05', facecolor='lightgray',
                     edgecolor='gray', linewidth=2, alpha=0.7, zorder=5))
        ax.text(ex, ey, f"Existing:\n{ef['type']}", ha='center', va='center', fontsize=8,
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    # New placed furniture
    if not placements:
        ax.text(room_N / 2, room_M / 2,
                'No valid placements.\nFix action scaling and re-train.',
                color='red', fontsize=11, ha='center', va='center',
                fontweight='bold', zorder=10)
    else:
        colors = plt.cm.Set3(np.linspace(0, 1, max(len(placements), 1)))
        for i, pf in enumerate(placements):
            ftype = pf.get('type', 'unknown')
            fname = pf.get('name', ftype)
            pos   = pf.get('position', [0, 0])
            rot   = pf.get('rotation_degrees', 0)
            fx, fy = pos[0], pos[1]
            fl, fw = get_furniture_dims(catalog, ftype)
            fl_r, fw_r = rotated_dims(fl, fw, rot)

            ax.add_patch(FancyBboxPatch((fx - fl_r/2, fy - fw_r/2), fl_r, fw_r,
                         boxstyle='round,pad=0.05', facecolor=colors[i],
                         edgecolor='darkblue', linewidth=2.5, alpha=0.8, zorder=6))
            ax.text(fx, fy, f"NEW ({i+1}):\n{fname}",
                    ha='center', va='center', fontsize=9, fontweight='bold',
                    bbox=dict(boxstyle='round', facecolor='white',
                              edgecolor='darkblue', linewidth=1.5, alpha=0.9))

    # Axes
    ax.set_xlim(-0.5, room_N + 0.5)
    ax.set_ylim(-0.5, room_M + 0.5)
    ax.set_aspect('equal')
    ax.set_xlabel('Length (m)', fontsize=12)
    ax.set_ylabel('Width (m)', fontsize=12)
    score_str = f'{best_reward:.4f}' if isinstance(best_reward, float) else 'N/A'
    ax.set_title(
        f'OID-PPO Furniture Placement\n'
        f'Best R_idg: {score_str}  |  Episode: {best_episode}/{total_episodes}  |  '
        f'Items placed: {len(placements)}',
        fontsize=13, fontweight='bold'
    )
    ax.grid(True, alpha=0.3, linestyle=':', linewidth=0.5)

    # Legend
    legend_elements = [
        patches.Patch(facecolor='lightgray', edgecolor='gray', label='Existing Furniture'),
        patches.Patch(facecolor='lightblue', edgecolor='darkblue', label='New Placement'),
        patches.Circle((0, 0), 0.1, facecolor='brown', label='Door'),
        patches.Rectangle((0, 0), 1, 0.3, facecolor='lightblue', edgecolor='blue', label='Window'),
        patches.Circle((0, 0), 0.1, fill=False, edgecolor='brown', linestyle='--', label='Door Clearance'),
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=10)

    # Reward components text box
    if reward_comps:
        label_map = {'R_pair':'Pairwise','R_a':'Accessibility','R_v':'Visibility',
                     'R_path':'Pathway','R_b':'Balance','R_al':'Alignment'}
        lines = ['Reward Components:']
        for k, v in reward_comps.items():
            lines.append(f"  {label_map.get(k, k)}: {v:.3f}")
        ax.text(0.01, 0.01, '\n'.join(lines), transform=ax.transAxes,
                fontsize=8, verticalalignment='bottom', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f'Layout saved to: {output_path}')

    if show:
        plt.show()
    plt.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Visualize OID-PPO layout')
    parser.add_argument('--results', default='oid_ppo_results/results.json')
    parser.add_argument('--room',    default='room_layout.json')
    parser.add_argument('--catalog', default='furniture_catalog_enhanced.json')
    parser.add_argument('--output',  default='layout_output.png')
    parser.add_argument('--no-show', action='store_true')
    args = parser.parse_args()

    missing = [p for p in [args.results, args.room, args.catalog] if not Path(p).exists()]
    if missing:
        print('ERROR: Missing files:')
        for m in missing: print(f'  {m}')
        exit(1)

    visualize(results_path=args.results, room_path=args.room,
              catalog_path=args.catalog, output_path=args.output,
              show=not args.no_show)