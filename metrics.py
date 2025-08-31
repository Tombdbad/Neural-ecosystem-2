"""
Emergence Tracker for the Neural Ecosystem.

This module tracks emergence patterns, complexity metrics, and system evolution
for the symbiotic intelligence architecture. Works with both Knowledge Keeper
beings to understand how community wisdom and individual development emerge naturally.

Mesa 3.2.0 compatible implementation with compassionate language.
"""

import time
import numpy as np
from typing import Dict, List, Any, Optional
from collections import defaultdict, deque

class EmergenceTracker:
    """
    Tracks emergence patterns and complexity metrics in the Neural Ecosystem.

    Monitors how natural intelligence, wisdom, and community patterns emerge
    from the symbiotic interactions between beings and Knowledge Keepers.
    """

    def __init__(self, model):
        """
        Initialize emergence tracking system.

        Args:
            model: The Neural Ecosystem model instance
        """
        self.model = model

        # Emergence detection thresholds
        self.emergence_thresholds = {
            'complexity': 0.6,
            'coherence': 0.7,
            'diversity': 0.5,
            'stability': 0.6,
            'wisdom_depth': 0.8
        }

        # Tracking systems
        self.emergence_history = []
        self.complexity_metrics = {}
        self.coherence_patterns = {}
        self.diversity_measurements = {}
        self.stability_indicators = {}

        # Pattern recognition
        self.emergent_behaviors = []
        self.collective_intelligence_indicators = []
        self.wisdom_emergence_patterns = []

        # Temporal emergence tracking
        self.emergence_cycles = {}
        self.pattern_evolution = {}
        self.community_development_stages = []

        print("EmergenceTracker initialized - monitoring natural intelligence emergence")
        print("Focus: Community wisdom, authentic development, symbiotic learning patterns")

    def step(self):
        """
        Execute a single step of emergence pattern detection and analysis.
        Monitors community development and wisdom emergence patterns.
        """
        current_step = getattr(self.model, 'steps', 0)

        # Calculate current emergence metrics
        emergence_metrics = self._calculate_emergence_metrics()

        # Detect emergent behaviors
        new_behaviors = self._detect_emergent_behaviors(emergence_metrics)
        self.emergent_behaviors.extend(new_behaviors)

        # Track collective intelligence indicators
        intelligence_indicators = self._assess_collective_intelligence()
        self.collective_intelligence_indicators.append(intelligence_indicators)

        # Monitor wisdom emergence patterns
        wisdom_patterns = self._track_wisdom_emergence(emergence_metrics)
        self.wisdom_emergence_patterns.append(wisdom_patterns)

        # Store emergence history
        emergence_entry = {
            'step': current_step,
            'timestamp': time.time(),
            'metrics': emergence_metrics,
            'behaviors': new_behaviors,
            'intelligence': intelligence_indicators,
            'wisdom_patterns': wisdom_patterns
        }

        self.emergence_history.append(emergence_entry)

        # Clean up old history
        if len(self.emergence_history) > 1000:
            self.emergence_history = self.emergence_history[-1000:]

    def _calculate_emergence_metrics(self) -> Dict[str, float]:
        """Calculate current emergence metrics."""
        if not hasattr(self.model, 'agents') or len(self.model.agents) == 0:
            return {
                'complexity': 0.0,
                'coherence': 0.0,
                'diversity': 0.0,
                'stability': 0.5,
                'wisdom_depth': 0.0
            }

        beings = list(self.model.agents)

        # Calculate complexity (variety in behaviors and states)
        complexity = self._calculate_complexity(beings)

        # Calculate coherence (alignment in community behavior)
        coherence = self._calculate_coherence(beings)

        # Calculate diversity (different types of development)
        diversity = self._calculate_diversity(beings)

        # Calculate stability (consistency over time)
        stability = self._calculate_stability(beings)

        # Calculate wisdom depth (collective wisdom accumulation)
        wisdom_depth = self._calculate_wisdom_depth(beings)

        return {
            'complexity': complexity,
            'coherence': coherence,
            'diversity': diversity,
            'stability': stability,
            'wisdom_depth': wisdom_depth
        }

    def _calculate_complexity(self, beings: List) -> float:
        """Calculate system complexity based on being diversity."""
        if not beings:
            return 0.0

        # Analyze energy distribution
        energies = [getattr(being, 'energy', 50) for being in beings]
        energy_variance = np.var(energies) / 2500  # Normalize by max variance (50^2)

        # Analyze wisdom distribution
        wisdom_levels = [getattr(being, 'accumulated_wisdom', 0) for being in beings]
        wisdom_variance = np.var(wisdom_levels) if wisdom_levels else 0

        # Analyze neurochemical diversity
        neurochemical_diversity = 0.0
        for being in beings:
            if hasattr(being, 'neurochemical_system'):
                neuro = being.neurochemical_system
                chemicals = [
                    getattr(neuro, 'contentment', 0.5),
                    getattr(neuro, 'curiosity', 0.5),
                    getattr(neuro, 'empathy', 0.5),
                    getattr(neuro, 'courage', 0.5)
                ]
                neurochemical_diversity += np.var(chemicals)

        neurochemical_diversity /= len(beings)

        complexity = (energy_variance + min(1.0, wisdom_variance) + neurochemical_diversity) / 3
        return min(1.0, complexity)

    def _calculate_coherence(self, beings: List) -> float:
        """Calculate coherence in community behavior."""
        if len(beings) < 2:
            return 0.5

        # Calculate social connection coherence
        connections = [getattr(being, 'social_connections', 0) for being in beings]
        avg_connections = np.mean(connections)
        connection_coherence = min(1.0, avg_connections / 3.0)  # Normalize to expected max

        # Calculate energy coherence (how well energies align)
        energies = [getattr(being, 'energy', 50) for being in beings]
        energy_std = np.std(energies)
        energy_coherence = max(0.0, 1.0 - energy_std / 50.0)  # Lower std = higher coherence

        # Calculate wisdom development coherence
        wisdom_levels = [getattr(being, 'accumulated_wisdom', 0) for being in beings]
        if max(wisdom_levels) > 0:
            wisdom_coherence = 1.0 - (np.std(wisdom_levels) / max(wisdom_levels))
        else:
            wisdom_coherence = 0.5

        coherence = (connection_coherence + energy_coherence + wisdom_coherence) / 3
        return max(0.0, min(1.0, coherence))

    def _calculate_diversity(self, beings: List) -> float:
        """Calculate diversity in development patterns."""
        if not beings:
            return 0.0

        # Count different growth stages
        growth_stages = set()
        for being in beings:
            stage = getattr(being, 'current_growth_stage', 'unknown')
            growth_stages.add(stage)

        stage_diversity = len(growth_stages) / 5.0  # Normalize by expected max stages

        # Analyze neurochemical profile diversity
        profiles = []
        for being in beings:
            if hasattr(being, 'neurochemical_system'):
                neuro = being.neurochemical_system
                profile = (
                    getattr(neuro, 'empathy', 0.5),
                    getattr(neuro, 'curiosity', 0.5),
                    getattr(neuro, 'courage', 0.5),
                    getattr(neuro, 'contentment', 0.5)
                )
                profiles.append(profile)

        profile_diversity = 0.0
        if len(profiles) > 1:
            # Calculate pairwise differences
            total_differences = 0
            pairs = 0
            for i in range(len(profiles)):
                for j in range(i + 1, len(profiles)):
                    difference = sum(abs(profiles[i][k] - profiles[j][k]) for k in range(4))
                    total_differences += difference
                    pairs += 1

            if pairs > 0:
                profile_diversity = (total_differences / pairs) / 4.0  # Normalize

        diversity = (stage_diversity + profile_diversity) / 2
        return min(1.0, diversity)

    def _calculate_stability(self, beings: List) -> float:
        """Calculate system stability over time."""
        if len(self.emergence_history) < 5:
            return 0.5  # Default stability for new systems

        # Analyze stability of key metrics over recent history
        recent_history = self.emergence_history[-5:]

        complexity_values = [entry['metrics'].get('complexity', 0) for entry in recent_history]
        coherence_values = [entry['metrics'].get('coherence', 0) for entry in recent_history]

        # Lower variance = higher stability
        complexity_stability = max(0.0, 1.0 - np.var(complexity_values))
        coherence_stability = max(0.0, 1.0 - np.var(coherence_values))

        # Current energy stability
        current_energies = [getattr(being, 'energy', 50) for being in beings]
        energy_stability = 1.0 if np.mean(current_energies) > 60 else 0.5

        stability = (complexity_stability + coherence_stability + energy_stability) / 3
        return min(1.0, stability)

    def _calculate_wisdom_depth(self, beings: List) -> float:
        """Calculate collective wisdom depth."""
        if not beings:
            return 0.0

        total_wisdom = sum(getattr(being, 'accumulated_wisdom', 0) for being in beings)
        avg_wisdom = total_wisdom / len(beings)

        # Normalize wisdom depth (scale by expected maximum)
        wisdom_depth = min(1.0, avg_wisdom / 10.0)

        # Bonus for wisdom distribution (better if multiple beings have wisdom)
        wisdom_holders = sum(1 for being in beings if getattr(being, 'accumulated_wisdom', 0) > 1.0)
        distribution_bonus = min(0.3, wisdom_holders / len(beings))

        return min(1.0, wisdom_depth + distribution_bonus)

    def _detect_emergent_behaviors(self, metrics: Dict[str, float]) -> List[str]:
        """Detect emergent behaviors based on metrics."""
        behaviors = []

        # High complexity + high coherence = organized complexity
        if metrics['complexity'] > 0.7 and metrics['coherence'] > 0.7:
            behaviors.append('organized_complexity_emergence')

        # High diversity + high stability = stable diversity
        if metrics['diversity'] > 0.6 and metrics['stability'] > 0.7:
            behaviors.append('stable_diverse_community')

        # High wisdom depth = collective intelligence
        if metrics['wisdom_depth'] > 0.8:
            behaviors.append('collective_intelligence_emergence')

        # All metrics above threshold = higher-order emergence
        if all(metrics[key] > self.emergence_thresholds.get(key, 0.6) for key in metrics):
            behaviors.append('higher_order_emergence')

        return behaviors

    def _assess_collective_intelligence(self) -> Dict[str, float]:
        """Assess indicators of collective intelligence."""
        if not hasattr(self.model, 'agents'):
            return {'collective_problem_solving': 0.0, 'distributed_wisdom': 0.0, 'adaptive_coordination': 0.0}

        beings = list(self.model.agents)

        # Collective problem solving (based on mutual help)
        helping_behaviors = 0
        total_interactions = 0
        for being in beings:
            if hasattr(being, 'recent_interactions'):
                interactions = getattr(being, 'recent_interactions', [])
                total_interactions += len(interactions)
                helping_behaviors += sum(1 for interaction in interactions 
                                       if interaction.get('type') == 'helping')

        problem_solving_score = helping_behaviors / max(total_interactions, 1)

        # Distributed wisdom (wisdom spread across beings)
        wisdom_scores = [getattr(being, 'accumulated_wisdom', 0) for being in beings]
        if max(wisdom_scores) > 0:
            wisdom_distribution = 1.0 - (np.std(wisdom_scores) / max(wisdom_scores))
        else:
            wisdom_distribution = 0.0

        # Adaptive coordination (social connections + empathy)
        total_connections = sum(getattr(being, 'social_connections', 0) for being in beings)
        avg_empathy = 0.0
        empathy_count = 0

        for being in beings:
            if hasattr(being, 'neurochemical_system'):
                empathy = getattr(being.neurochemical_system, 'empathy', 0.5)
                avg_empathy += empathy
                empathy_count += 1

        if empathy_count > 0:
            avg_empathy /= empathy_count

        coordination_score = (total_connections / len(beings) + avg_empathy) / 2

        return {
            'collective_problem_solving': min(1.0, problem_solving_score),
            'distributed_wisdom': min(1.0, wisdom_distribution),
            'adaptive_coordination': min(1.0, coordination_score)
        }

    def _track_wisdom_emergence(self, metrics: Dict[str, float]) -> Dict[str, Any]:
        """Track patterns of wisdom emergence."""
        wisdom_patterns = {
            'current_wisdom_depth': metrics['wisdom_depth'],
            'wisdom_acceleration': 0.0,
            'integration_patterns': [],
            'sharing_behaviors': 0
        }

        # Calculate wisdom acceleration
        if len(self.emergence_history) >= 3:
            recent_wisdom = [entry['metrics'].get('wisdom_depth', 0) 
                           for entry in self.emergence_history[-3:]]
            if len(recent_wisdom) > 1:
                wisdom_slope = np.polyfit(range(len(recent_wisdom)), recent_wisdom, 1)[0]
                wisdom_patterns['wisdom_acceleration'] = max(0.0, wisdom_slope)

        # Detect integration patterns
        if hasattr(self.model, 'agents'):
            beings = list(self.model.agents)

            # High wisdom integrator count
            high_integrators = sum(1 for being in beings
                                 if hasattr(being, 'neurochemical_system') and
                                 getattr(being.neurochemical_system, 'wisdom_integrator', 1.0) > 1.2)

            if high_integrators > 0:
                wisdom_patterns['integration_patterns'].append('active_wisdom_integration')

            # Wisdom sharing behaviors
            sharing_count = 0
            for being in beings:
                if hasattr(being, 'recent_interactions'):
                    interactions = getattr(being, 'recent_interactions', [])
                    sharing_count += sum(1 for interaction in interactions
                                       if interaction.get('type') == 'wisdom_sharing')

            wisdom_patterns['sharing_behaviors'] = sharing_count

        return wisdom_patterns

    def get_emergence_score(self) -> float:
        """Get current emergence score for the system."""
        if not self.emergence_history:
            return 0.0

        latest_metrics = self.emergence_history[-1]['metrics']

        # Calculate weighted emergence score
        weights = {
            'complexity': 0.2,
            'coherence': 0.25,
            'diversity': 0.15,
            'stability': 0.2,
            'wisdom_depth': 0.2
        }

        emergence_score = sum(latest_metrics.get(metric, 0) * weight 
                            for metric, weight in weights.items())

        return min(1.0, emergence_score)

    def get_emergence_report(self) -> Dict:
        """Get comprehensive emergence report for Knowledge Keepers."""
        if not self.emergence_history:
            return {'status': 'no_data_available'}

        latest_metrics = self.emergence_history[-1]['metrics']

        report = {
            'current_emergence_score': self.get_emergence_score(),
            'emergence_metrics': latest_metrics,
            'emergent_behaviors_detected': len(self.emergent_behaviors),
            'collective_intelligence_indicators': len(self.collective_intelligence_indicators),
            'wisdom_emergence_events': len(self.wisdom_emergence_patterns),
            'community_development_stage': self._assess_community_development_stage(latest_metrics),
            'notable_patterns': self._identify_notable_patterns(),
            'emergence_trends': self._analyze_emergence_trends()
        }

        return report

    def _assess_community_development_stage(self, metrics: Dict) -> str:
        """Assess the current community development stage."""
        emergence_score = self.get_emergence_score()
        wisdom_level = metrics.get('collective_wisdom', 0)
        social_emergence = metrics.get('social_emergence', 0)

        if emergence_score < 0.3:
            return 'foundation_building'
        elif emergence_score < 0.5:
            return 'pattern_emergence'
        elif emergence_score < 0.7:
            return 'community_integration'
        elif wisdom_level > 0.7 and social_emergence > 0.7:
            return 'wisdom_culture_formation'
        else:
            return 'mature_symbiotic_community'

    def _identify_notable_patterns(self) -> List[str]:
        """Identify notable patterns that have emerged."""
        notable_patterns = []

        # Check for sustained high emergence
        if len(self.emergence_history) >= 20:
            recent_scores = [self.get_emergence_score() for _ in range(10)]
            if np.mean(recent_scores) > 0.7:
                notable_patterns.append('sustained_high_emergence')

        # Check for wisdom acceleration
        if len(self.pattern_evolution.get('collective_wisdom', [])) >= 10:
            recent_wisdom = [entry['value'] for entry in self.pattern_evolution['collective_wisdom'][-10:]]
            wisdom_trend = np.polyfit(range(len(recent_wisdom)), recent_wisdom, 1)[0]
            if wisdom_trend > 0.05:
                notable_patterns.append('accelerating_wisdom_development')

        # Check for stable community emergence
        if 'stable_diverse_community' in self.emergent_behaviors:
            notable_patterns.append('stable_diverse_community_achieved')

        return notable_patterns

    def _analyze_emergence_trends(self) -> Dict:
        """Analyze trends in emergence patterns."""
        trends = {
            'complexity_trend': 'stable',
            'wisdom_trend': 'stable',
            'social_trend': 'stable',
            'overall_direction': 'stable'
        }

        if len(self.emergence_history) < 10:
            return trends

        # Analyze complexity trend
        complexity_values = [entry['metrics'].get('complexity', 0) for entry in self.emergence_history[-10:]]
        complexity_slope = np.polyfit(range(len(complexity_values)), complexity_values, 1)[0]

        if complexity_slope > 0.02:
            trends['complexity_trend'] = 'increasing'
        elif complexity_slope < -0.02:
            trends['complexity_trend'] = 'decreasing'

        # Analyze wisdom trend
        wisdom_values = [entry['metrics'].get('collective_wisdom', 0) for entry in self.emergence_history[-10:]]
        wisdom_slope = np.polyfit(range(len(wisdom_values)), wisdom_values, 1)[0]

        if wisdom_slope > 0.02:
            trends['wisdom_trend'] = 'increasing'
        elif wisdom_slope < -0.02:
            trends['wisdom_trend'] = 'decreasing'

        # Analyze social emergence trend
        social_values = [entry['metrics'].get('social_emergence', 0) for entry in self.emergence_history[-10:]]
        social_slope = np.polyfit(range(len(social_values)), social_values, 1)[0]

        if social_slope > 0.02:
            trends['social_trend'] = 'increasing'
        elif social_slope < -0.02:
            trends['social_trend'] = 'decreasing'

        # Overall direction
        positive_trends = sum(1 for trend in [complexity_slope, wisdom_slope, social_slope] if trend > 0.01)

        if positive_trends >= 2:
            trends['overall_direction'] = 'flourishing'
        elif positive_trends == 0:
            trends['overall_direction'] = 'consolidating'
        else:
            trends['overall_direction'] = 'evolving'

        return trends

    def serialize(self) -> Dict[str, Any]:
        """Serialize emergence tracker state."""
        return {
            'emergence_thresholds': self.emergence_thresholds,
            'emergent_behaviors_count': len(self.emergent_behaviors),
            'collective_intelligence_indicators_count': len(self.collective_intelligence_indicators),
            'wisdom_emergence_events': len(self.wisdom_emergence_patterns),
            'emergence_history_length': len(self.emergence_history),
            'current_emergence_score': self.get_emergence_score()
        }

    def deserialize(self, data: Dict[str, Any]) -> None:
        """Deserialize emergence tracker state."""
        self.emergence_thresholds = data.get('emergence_thresholds', self.emergence_thresholds)
        # Other state restoration would happen here
"""
Emergence Tracker for the Neural Ecosystem.
Mesa 3.2.0 compatible implementation with compassionate metrics.
"""

import time
import numpy as np
from typing import Dict, List, Any, Optional

class EmergenceTracker:
    """
    Tracks emergence patterns in the Neural Ecosystem.
    Monitors natural intelligence emergence, community development, and symbiotic learning.
    """
    
    def __init__(self, model):
        """
        Initialize emergence tracker.
        
        Args:
            model: The Neural Ecosystem model instance
        """
        self.model = model
        
        # Emergence tracking
        self.emergence_history = []
        self.community_patterns = {}
        self.wisdom_emergence_events = []
        
        # Metrics
        self.current_emergence_score = 0.0
        self.community_coherence = 0.0
        self.collective_wisdom_level = 0.0
        
        print("EmergenceTracker initialized - monitoring natural intelligence emergence")
        print("Focus: Community wisdom, authentic development, symbiotic learning patterns")
    
    def step(self):
        """Process emergence tracking for current step."""
        if not self.model.agents:
            return
        
        # Calculate emergence metrics
        self._calculate_emergence_score()
        self._track_community_patterns()
        self._monitor_wisdom_emergence()
        
        # Store emergence data
        self._record_emergence_event()
    
    def _calculate_emergence_score(self):
        """Calculate overall emergence score."""
        beings = list(self.model.agents)
        if not beings:
            self.current_emergence_score = 0.0
            return
        
        # Factors for emergence
        factors = []
        
        # Average wisdom level
        total_wisdom = sum(getattr(being, 'accumulated_wisdom', 0) for being in beings)
        avg_wisdom = total_wisdom / len(beings)
        factors.append(min(1.0, avg_wisdom / 5.0))
        
        # Social connections
        total_connections = sum(getattr(being, 'social_connections', 0) for being in beings)
        connection_factor = min(1.0, total_connections / (len(beings) * 2))
        factors.append(connection_factor)
        
        # Energy distribution
        energies = [getattr(being, 'energy', 50) for being in beings]
        energy_factor = np.mean(energies) / 100.0
        factors.append(energy_factor)
        
        self.current_emergence_score = np.mean(factors)
    
    def _track_community_patterns(self):
        """Track patterns in community development."""
        beings = list(self.model.agents)
        
        # Community coherence
        if len(beings) > 1:
            # Calculate position clustering
            positions = [being.pos for being in beings if being.pos]
            if len(positions) > 1:
                distances = []
                for i, pos1 in enumerate(positions):
                    for pos2 in positions[i+1:]:
                        dist = ((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)**0.5
                        distances.append(dist)
                
                avg_distance = np.mean(distances)
                max_distance = (self.model.width**2 + self.model.height**2)**0.5
                self.community_coherence = 1.0 - (avg_distance / max_distance)
    
    def _monitor_wisdom_emergence(self):
        """Monitor emergence of collective wisdom."""
        beings = list(self.model.agents)
        
        # Count beings in different wisdom stages
        wisdom_stages = {}
        for being in beings:
            stage = getattr(being, 'current_growth_stage', 'unknown')
            wisdom_stages[stage] = wisdom_stages.get(stage, 0) + 1
        
        # Collective wisdom based on diversity and depth
        stage_diversity = len(wisdom_stages)
        total_wisdom = sum(getattr(being, 'accumulated_wisdom', 0) for being in beings)
        
        self.collective_wisdom_level = (stage_diversity * 0.3 + min(total_wisdom / 10.0, 1.0) * 0.7)
    
    def _record_emergence_event(self):
        """Record current emergence state."""
        event = {
            'timestamp': time.time(),
            'step': self.model.steps,
            'emergence_score': self.current_emergence_score,
            'community_coherence': self.community_coherence,
            'collective_wisdom_level': self.collective_wisdom_level,
            'being_count': len(self.model.agents)
        }
        
        self.emergence_history.append(event)
        
        # Keep history manageable
        if len(self.emergence_history) > 1000:
            self.emergence_history = self.emergence_history[-500:]
    
    def get_emergence_score(self) -> float:
        """Get current emergence score."""
        return self.current_emergence_score
    
    def get_emergence_report(self) -> Dict[str, Any]:
        """Get comprehensive emergence report."""
        return {
            'current_emergence_score': self.current_emergence_score,
            'community_coherence': self.community_coherence,
            'collective_wisdom_level': self.collective_wisdom_level,
            'emergence_history_length': len(self.emergence_history),
            'recent_trends': self._calculate_recent_trends()
        }
    
    def _calculate_recent_trends(self) -> Dict[str, str]:
        """Calculate recent emergence trends."""
        if len(self.emergence_history) < 5:
            return {'trend': 'insufficient_data'}
        
        recent_scores = [event['emergence_score'] for event in self.emergence_history[-5:]]
        
        if len(recent_scores) < 2:
            return {'trend': 'stable'}
        
        # Simple trend calculation
        trend_slope = (recent_scores[-1] - recent_scores[0]) / len(recent_scores)
        
        if trend_slope > 0.02:
            return {'trend': 'emerging'}
        elif trend_slope < -0.02:
            return {'trend': 'declining'}
        else:
            return {'trend': 'stable'}
