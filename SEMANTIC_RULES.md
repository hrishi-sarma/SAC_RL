# 📋 Semantic Rules Reference Card

## Quick Reference for Furniture Placement Constraints

---

## 🏠 Wall-Mounted Items

### Types: wall_shelf, mirror, decorative_ladder

```
Rule: MUST be near walls
Threshold: 0.35m from any wall

Additional Constraints:
- Mirrors: Avoid windows (0.8m) and doors (1.0m)
- Wall shelves: Avoid windows (0.8m) and doors (0.8m)
- Decorative ladders: Very close to walls (0.3m)

✓ Valid:   [Wall]----[Item]
✗ Invalid: [Wall]----------[Item]  (too far)
✗ Invalid: [Window]--[Mirror]  (too close to window)
```

**Reward Components:**
- semantic_correctness: +1.5 if near wall
- wall_proximity: +1.0 if very close (0.3m)

---

## 🌿 Plants & Decorative Items

### Type: plant_stand

```
Rule: MUST be in corner OR near window
Corner threshold: 1.0m from corner
Window threshold: 1.5m from window

✓ Valid:   [Corner]--[Plant]
✓ Valid:   [Window]---[Plant]
✗ Invalid: [Center of room]--[Plant]
```

**Reward Components:**
- corner_bonus: +1.0 if in corner
- window_bonus: +1.0 if near window
- semantic_correctness: +1.0 for either

---

## 💡 Lighting

### Type: floor_lamp

```
Rule: MUST be within 2.0m of seating
Seating includes: sofa, armchair, ottoman, bench

✓ Valid:   [Sofa]----[Lamp]  (< 2m)
✗ Invalid: [Sofa]----------[Lamp]  (> 2m)

Often combined with corner placement for best results.
```

### Type: table_lamp

```
Rule: MUST be on table surface
Threshold: 0.5m from table center
Tables: coffee_table, side_table, console_table, nesting_tables

✓ Valid:   Lamp placed on/very near table
✗ Invalid: Lamp on floor or away from tables
```

**Reward Components:**
- semantic_correctness: +1.0 if near seating/tables

---

## 📚 Storage Furniture

### Types: bookshelf, console_table, media_cabinet

```
Rule: SHOULD be near walls (not in center)
Threshold: 0.8m from walls

✓ Valid:   [Wall]---[Bookshelf]
✗ Invalid: [Center]--[Bookshelf]

Purpose: Keeps center of room open for traffic flow
```

**Reward Components:**
- semantic_correctness: +0.8 if near wall
- wall_proximity: +0.8 bonus

---

## 🎨 Area Rugs

### Type: rug

```
Rule: SHOULD be near/under seating
Threshold: 3.0m from seating furniture

✓ Valid:   [Sofa]--[Rug]--[Chair]
✗ Invalid: [Rug in empty corner]

Purpose: Defines conversation/seating area
```

---

## 📏 Grid Alignment (All Items)

```
Rule: Snap to 30cm grid
Grid points: 0.0, 0.3, 0.6, 0.9, 1.2, ...

✓ Valid:   Position = 1.2m (on grid)
✗ Penalty: Position = 1.17m (off grid, gets snapped)

All items automatically snap to nearest grid point.
```

**Reward Component:**
- grid_alignment: +1.0 if on grid, +0.5 otherwise

---

## 🚧 Collision & Clearance

### Collision Buffer

```
Rule: Minimum 20cm gap between all furniture
Buffer: 0.20m around each item

✓ Valid:   [Item1]----0.20m----[Item2]
✗ Invalid: [Item1]-0.10m-[Item2]
```

### Door Clearance

```
Rule: Respect door swing radius
Clearance: 1.2m arc from door

✓ Valid:   Item outside clearance arc
✗ Invalid: Item blocks door swing
```

### Pathway Width

```
Rule: Maintain walkable paths
Minimum: 0.9m pathway width
Clearance: 0.6m minimum from walls

✓ Valid:   0.9m+ open path
✗ Invalid: Narrow (<0.6m) passages
```

---

## 🎯 Rotation Constraints

```
Rule: Only 90° rotations allowed
Valid angles: 0°, 90°, 180°, 270°

✓ Valid:   0°, 90°, 180°, 270°
✗ Invalid: 45°, 135°, any other angle

Purpose: Keeps furniture parallel to walls
```

**Reward Component:**
- parallel_placement: +1.0 if parallel to existing items
- alignment: +1.0 (always, since enforced)

---

## 📊 Reward Weight Summary

| Component | Max Value | When Awarded |
|-----------|-----------|--------------|
| semantic_correctness | 2.0 | Type-specific rules followed |
| wall_proximity | 1.0 | Appropriate wall distance |
| corner_bonus | 1.0 | Corner placement (specific types) |
| window_bonus | 1.0 | Near windows (plants) |
| grid_alignment | 1.0 | On grid points |
| parallel_placement | 1.0 | Aligned with room/furniture |
| functional_pairing | 1.0 | Good furniture combinations |
| accessibility | 1.0 | Adequate clearance |
| clearance | 1.0 | Pathway maintenance |
| visual_balance | 1.0 | Spatial distribution |
| color_harmony | 1.0 | Color scheme match |
| diversity | 1.0 | Variety of types |
| budget_efficiency | 1.0 | Good budget use |
| completeness | 1.0 | All items placed |
| size_appropriateness | 1.0 | Proper item sizes |

**Total Possible**: ~16.0 points per placement

---

## 🎓 Rule Priority

When rules conflict, priority order:

1. **Safety**: Door clearance (CRITICAL)
2. **Physics**: Collision avoidance (CRITICAL)
3. **Boundaries**: Room bounds (CRITICAL)
4. **Semantic**: Type-specific rules (HIGH)
5. **Grid**: Alignment (MEDIUM)
6. **Aesthetic**: Balance, harmony (LOW)

---

## 💡 Design Philosophy

### Why These Rules?

1. **Wall Items on Walls** - Mimics real installation
2. **Plants Near Windows** - Natural light requirement
3. **Lamps Near Seating** - Functional lighting
4. **Storage on Perimeter** - Maximizes open space
5. **Grid Alignment** - Professional appearance
6. **Collision Buffer** - Livable spaces

### Real-World Validation

All rules based on:
- ✓ Interior design best practices
- ✓ Ergonomics and accessibility standards
- ✓ Common furniture arrangement patterns
- ✓ User experience considerations

---

## 🔧 Customization Guide

### Adjust Thresholds

In `furniture_env_semantic.py`:

```python
# Make rules stricter
wall_proximity = 0.25  # Was 0.35 (closer to wall)
corner_threshold = 0.7  # Was 1.0 (tighter corners)

# Make rules looser  
wall_proximity = 0.50  # Was 0.35 (more flexible)
seating_distance = 2.5  # Was 2.0 (lamps farther)
```

### Add New Rules

```python
def _check_semantic_constraints(self, furniture, catalog_item):
    # Example: Add rule for coffee tables
    if item_type == 'coffee_table':
        # Must be within 1.5m of sofa
        if not self._is_near_furniture_type(x, y, 'sofa', 1.5):
            return False
    # ... existing rules
```

### Disable Rules

```python
# Comment out unwanted rules
# if item_type == 'plant_stand':
#     if not (near_corner or near_window):
#         return False
```

---

## 📈 Expected Compliance Rates

After 1000 training episodes:

| Rule Type | Compliance Rate |
|-----------|----------------|
| Wall items on walls | 95-100% |
| Plants in corners/windows | 85-95% |
| Lamps near seating | 90-100% |
| Storage on perimeter | 85-95% |
| Grid alignment | 95-100% |
| No collisions | 95-100% |

---

## ✅ Validation Checklist

Use this to verify placements:

```
□ Wall items within 0.35m of walls?
□ Mirrors/shelves avoid windows/doors?
□ Plants in corners or near windows?
□ Lamps near seating areas?
□ Storage on perimeter (not center)?
□ Grid alignment (30cm)?
□ Collision buffer (20cm+)?
□ Door clearance maintained?
□ Pathways ≥ 0.9m wide?
□ Rotations at 90° increments?
```

All should be ✓ for valid placement.

---

## 🎯 Quick Decision Tree

```
Is item wall-mounted?
├─ YES → Place within 0.35m of wall
│         Avoid windows/doors
│         
└─ NO → Is it a plant?
    ├─ YES → Corner OR window
    │         
    └─ NO → Is it a lamp?
        ├─ YES → Near seating/tables
        │         
        └─ NO → Is it storage?
            ├─ YES → Near walls
            │         
            └─ NO → Standard placement
                     (grid, collision, clearance)
```

---

## 🏆 Pro Tips

1. **Always check semantic rules first** - Before other constraints
2. **Corner placement often optimal** - For plants, lamps, etc.
3. **Wall perimeter for storage** - Keeps room open
4. **Grid alignment is free** - Automatically enforced
5. **Trust the rewards** - System learns optimal patterns

---

*For implementation details, see `furniture_env_semantic.py`*
*For full documentation, see `README_SEMANTIC.md`*
