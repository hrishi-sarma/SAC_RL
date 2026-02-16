"""
Placement Rules Graph Definition
Defines hierarchical spatial-functional relationships for furniture placement
"""

import numpy as np
from typing import Dict, List, Tuple, Optional


class PlacementRulesGraph:
    """
    Defines explicit graph structure for furniture placement rules
    
    Structure:
    - Root: Furniture Type
    - Level 1: MUST_BE_IN, MUST_AVOID
    - Level 2: Specific zones/constraints
    - Level 3: Attributes (distances, penalties, weights)
    """
    
    def __init__(self):
        self.rules = self._build_placement_rules()
        self.zone_hierarchy = self._build_zone_hierarchy()
        self.tag_compatibility = self._build_tag_compatibility()
        
    def _build_placement_rules(self) -> Dict:
        """
        Build hierarchical placement rules for each furniture type
        """
        return {
            'coffee_table': {
                'MUST_BE_IN': {
                    'seating_zone': {
                        'priority': 1.0,
                        'weight': 5.0,
                        'max_distance': 1.5,
                        'min_distance': 0.3
                    },
                    'conversation_zone': {
                        'priority': 0.8,
                        'weight': 3.0,
                        'max_distance': 2.0
                    },
                    'near_sofa': {
                        'priority': 0.9,
                        'weight': 4.0,
                        'min_distance': 0.4,
                        'max_distance': 1.2,
                        'target_type': 'sofa'
                    }
                },
                'MUST_AVOID': {
                    'door_clearance': {
                        'priority': 1.0,
                        'penalty': -20.0,
                        'radius': 1.5,
                        'type': 'critical'
                    },
                    'traffic_paths': {
                        'priority': 0.9,
                        'penalty': -15.0,
                        'width': 1.0,
                        'type': 'high'
                    },
                    'window_blocking': {
                        'priority': 0.7,
                        'penalty': -8.0,
                        'radius': 0.6,
                        'type': 'medium'
                    },
                    'room_entrance': {
                        'priority': 1.0,
                        'penalty': -25.0,
                        'radius': 1.8,
                        'type': 'critical'
                    }
                },
                'TAGS': ['seating_companion', 'focal_point', 'surface'],
                'FUNCTIONAL_ZONE': 'seating_area'
            },
            
            'side_table': {
                'MUST_BE_IN': {
                    'near_sofa': {
                        'priority': 1.0,
                        'weight': 5.0,
                        'min_distance': 0.2,
                        'max_distance': 0.6,
                        'target_type': 'sofa',
                        'preferred_side': 'arm'
                    },
                    'near_armchair': {
                        'priority': 0.9,
                        'weight': 4.0,
                        'min_distance': 0.2,
                        'max_distance': 0.5,
                        'target_type': 'armchair'
                    },
                    'seating_zone': {
                        'priority': 0.8,
                        'weight': 3.0,
                        'max_distance': 1.0
                    }
                },
                'MUST_AVOID': {
                    'door_clearance': {
                        'priority': 0.9,
                        'penalty': -12.0,
                        'radius': 1.2,
                        'type': 'high'
                    },
                    'traffic_paths': {
                        'priority': 0.8,
                        'penalty': -10.0,
                        'width': 0.8,
                        'type': 'medium'
                    },
                    'window_blocking': {
                        'priority': 0.5,
                        'penalty': -5.0,
                        'radius': 0.4,
                        'type': 'low'
                    }
                },
                'TAGS': ['seating_accessory', 'task_support', 'surface'],
                'FUNCTIONAL_ZONE': 'seating_area'
            },
            
            'bookshelf': {
                'MUST_BE_IN': {
                    'wall_aligned': {
                        'priority': 1.0,
                        'weight': 6.0,
                        'max_distance_from_wall': 0.3,
                        'prefer_solid_wall': True
                    },
                    'reading_zone': {
                        'priority': 0.8,
                        'weight': 3.0,
                        'max_distance': 2.5
                    },
                    'corner_placement': {
                        'priority': 0.6,
                        'weight': 2.0,
                        'corner_preference': True
                    }
                },
                'MUST_AVOID': {
                    'door_clearance': {
                        'priority': 0.9,
                        'penalty': -15.0,
                        'radius': 1.3,
                        'type': 'high'
                    },
                    'window_blocking': {
                        'priority': 0.9,
                        'penalty': -12.0,
                        'radius': 0.8,
                        'type': 'high'
                    },
                    'traffic_paths': {
                        'priority': 0.7,
                        'penalty': -8.0,
                        'width': 0.9,
                        'type': 'medium'
                    },
                    'room_center': {
                        'priority': 0.8,
                        'penalty': -10.0,
                        'radius': 1.5,
                        'type': 'medium'
                    }
                },
                'TAGS': ['storage', 'wall_unit', 'display_surface'],
                'FUNCTIONAL_ZONE': 'storage_area'
            },
            
            'console_table': {
                'MUST_BE_IN': {
                    'wall_aligned': {
                        'priority': 1.0,
                        'weight': 6.0,
                        'max_distance_from_wall': 0.2,
                        'prefer_solid_wall': True
                    },
                    'entryway_zone': {
                        'priority': 0.7,
                        'weight': 3.0,
                        'max_distance': 1.5
                    },
                    'behind_sofa': {
                        'priority': 0.6,
                        'weight': 2.5,
                        'target_type': 'sofa',
                        'position': 'behind',
                        'distance': 0.3
                    }
                },
                'MUST_AVOID': {
                    'door_swing': {
                        'priority': 1.0,
                        'penalty': -18.0,
                        'radius': 1.4,
                        'type': 'critical'
                    },
                    'traffic_paths': {
                        'priority': 0.8,
                        'penalty': -10.0,
                        'width': 0.9,
                        'type': 'medium'
                    },
                    'window_blocking': {
                        'priority': 0.6,
                        'penalty': -6.0,
                        'radius': 0.5,
                        'type': 'low'
                    },
                    'room_center': {
                        'priority': 0.9,
                        'penalty': -12.0,
                        'radius': 2.0,
                        'type': 'high'
                    }
                },
                'TAGS': ['wall_unit', 'entryway', 'display_surface'],
                'FUNCTIONAL_ZONE': 'entryway'
            },
            
            'armchair': {
                'MUST_BE_IN': {
                    'seating_zone': {
                        'priority': 1.0,
                        'weight': 5.0,
                        'max_distance': 2.0
                    },
                    'reading_corner': {
                        'priority': 0.8,
                        'weight': 4.0,
                        'near_window': True,
                        'near_wall': True
                    },
                    'conversation_arrangement': {
                        'priority': 0.7,
                        'weight': 3.0,
                        'near_sofa': True,
                        'facing_angle': [60, 120]  # degrees
                    }
                },
                'MUST_AVOID': {
                    'door_clearance': {
                        'priority': 1.0,
                        'penalty': -18.0,
                        'radius': 1.4,
                        'type': 'critical'
                    },
                    'traffic_paths': {
                        'priority': 0.9,
                        'penalty': -14.0,
                        'width': 1.0,
                        'type': 'high'
                    },
                    'window_blocking': {
                        'priority': 0.4,
                        'penalty': -4.0,
                        'radius': 0.5,
                        'type': 'low'
                    }
                },
                'TAGS': ['primary_seating', 'flexible_seating', 'focal_point'],
                'FUNCTIONAL_ZONE': 'seating_area'
            },
            
            'floor_lamp': {
                'MUST_BE_IN': {
                    'near_seating': {
                        'priority': 1.0,
                        'weight': 5.0,
                        'target_types': ['sofa', 'armchair'],
                        'min_distance': 0.3,
                        'max_distance': 0.8,
                        'preferred_side': 'beside'
                    },
                    'reading_zone': {
                        'priority': 0.8,
                        'weight': 3.0,
                        'max_distance': 1.5
                    },
                    'corner_placement': {
                        'priority': 0.6,
                        'weight': 2.0,
                        'corner_preference': True
                    }
                },
                'MUST_AVOID': {
                    'door_clearance': {
                        'priority': 0.9,
                        'penalty': -10.0,
                        'radius': 1.0,
                        'type': 'high'
                    },
                    'traffic_paths': {
                        'priority': 1.0,
                        'penalty': -15.0,
                        'width': 0.8,
                        'type': 'critical'
                    },
                    'room_center': {
                        'priority': 0.7,
                        'penalty': -8.0,
                        'radius': 1.5,
                        'type': 'medium'
                    }
                },
                'TAGS': ['task_lighting', 'ambient_lighting', 'corner_decor'],
                'FUNCTIONAL_ZONE': 'seating_area'
            },
            
            'storage_basket': {
                'MUST_BE_IN': {
                    'corner_placement': {
                        'priority': 0.9,
                        'weight': 4.0,
                        'corner_preference': True
                    },
                    'near_seating': {
                        'priority': 0.7,
                        'weight': 2.5,
                        'max_distance': 2.0
                    },
                    'under_console': {
                        'priority': 0.8,
                        'weight': 3.0,
                        'target_type': 'console_table',
                        'position': 'under'
                    }
                },
                'MUST_AVOID': {
                    'door_clearance': {
                        'priority': 0.8,
                        'penalty': -8.0,
                        'radius': 1.0,
                        'type': 'medium'
                    },
                    'traffic_paths': {
                        'priority': 0.9,
                        'penalty': -12.0,
                        'width': 0.7,
                        'type': 'high'
                    },
                    'room_center': {
                        'priority': 0.6,
                        'penalty': -6.0,
                        'radius': 1.5,
                        'type': 'low'
                    }
                },
                'TAGS': ['corner_storage', 'flexible_tucked_away'],
                'FUNCTIONAL_ZONE': 'storage_area'
            },
            
            'plant_stand': {
                'MUST_BE_IN': {
                    'corner_placement': {
                        'priority': 0.8,
                        'weight': 3.5,
                        'corner_preference': True
                    },
                    'near_window': {
                        'priority': 1.0,
                        'weight': 5.0,
                        'max_distance': 1.5,
                        'min_distance': 0.4
                    },
                    'accent_zone': {
                        'priority': 0.6,
                        'weight': 2.0,
                        'max_distance': 2.0
                    }
                },
                'MUST_AVOID': {
                    'door_clearance': {
                        'priority': 0.7,
                        'penalty': -7.0,
                        'radius': 1.0,
                        'type': 'medium'
                    },
                    'traffic_paths': {
                        'priority': 0.8,
                        'penalty': -10.0,
                        'width': 0.7,
                        'type': 'medium'
                    },
                    'room_center': {
                        'priority': 0.7,
                        'penalty': -8.0,
                        'radius': 1.8,
                        'type': 'medium'
                    }
                },
                'TAGS': ['corner_decor', 'ambient_lighting', 'natural_element'],
                'FUNCTIONAL_ZONE': 'accent_area'
            }
        }
    
    def _build_zone_hierarchy(self) -> Dict:
        """
        Define zone hierarchies and their relationships
        """
        return {
            'seating_area': {
                'parent': 'living_space',
                'children': ['conversation_zone', 'reading_zone'],
                'compatible_tags': [
                    'primary_seating', 'seating_companion', 
                    'seating_accessory', 'task_lighting'
                ],
                'priority': 1.0
            },
            'conversation_zone': {
                'parent': 'seating_area',
                'children': [],
                'compatible_tags': [
                    'seating_companion', 'focal_point', 'surface'
                ],
                'priority': 0.9
            },
            'reading_zone': {
                'parent': 'seating_area',
                'children': [],
                'compatible_tags': [
                    'task_lighting', 'seating_accessory', 
                    'flexible_seating'
                ],
                'priority': 0.8
            },
            'storage_area': {
                'parent': 'living_space',
                'children': [],
                'compatible_tags': [
                    'storage', 'wall_unit', 'corner_storage'
                ],
                'priority': 0.7
            },
            'entryway': {
                'parent': 'living_space',
                'children': [],
                'compatible_tags': [
                    'entryway', 'wall_unit', 'display_surface'
                ],
                'priority': 0.8
            },
            'accent_area': {
                'parent': 'living_space',
                'children': [],
                'compatible_tags': [
                    'corner_decor', 'ambient_lighting', 
                    'natural_element'
                ],
                'priority': 0.6
            }
        }
    
    def _build_tag_compatibility(self) -> Dict:
        """
        Define tag compatibility matrix
        Returns compatibility scores between tags
        """
        return {
            'seating_companion': {
                'primary_seating': 0.95,
                'seating_accessory': 0.85,
                'task_lighting': 0.80,
                'focal_point': 0.75,
                'surface': 0.90
            },
            'task_lighting': {
                'primary_seating': 0.90,
                'seating_accessory': 0.85,
                'reading_zone': 0.95,
                'corner_decor': 0.70
            },
            'wall_unit': {
                'display_surface': 0.85,
                'storage': 0.90,
                'entryway': 0.80
            },
            'corner_storage': {
                'storage': 0.85,
                'flexible_tucked_away': 0.90,
                'corner_decor': 0.75
            },
            'corner_decor': {
                'ambient_lighting': 0.85,
                'natural_element': 0.90,
                'accent_area': 0.95
            }
        }
    
    def get_furniture_rules(self, furniture_type: str) -> Optional[Dict]:
        """Get placement rules for a specific furniture type"""
        return self.rules.get(furniture_type)
    
    def get_must_avoid_zones(self, furniture_type: str) -> Dict:
        """Get all zones this furniture must avoid"""
        rules = self.get_furniture_rules(furniture_type)
        if rules:
            return rules.get('MUST_AVOID', {})
        return {}
    
    def get_must_be_in_zones(self, furniture_type: str) -> Dict:
        """Get all zones this furniture should be in"""
        rules = self.get_furniture_rules(furniture_type)
        if rules:
            return rules.get('MUST_BE_IN', {})
        return {}
    
    def get_furniture_tags(self, furniture_type: str) -> List[str]:
        """Get tags for a furniture type"""
        rules = self.get_furniture_rules(furniture_type)
        if rules:
            return rules.get('TAGS', [])
        return []
    
    def get_tag_compatibility_score(self, tag1: str, tag2: str) -> float:
        """Get compatibility score between two tags"""
        if tag1 in self.tag_compatibility:
            return self.tag_compatibility[tag1].get(tag2, 0.5)
        if tag2 in self.tag_compatibility:
            return self.tag_compatibility[tag2].get(tag1, 0.5)
        return 0.5  # neutral compatibility
