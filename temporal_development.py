"""
Temporal Development Tracker for the Neural Ecosystem.

This module tracks development patterns across different time scales,
monitoring natural growth rhythms and community evolution phases.

Mesa 3.2.0 compatible implementation with compassionate temporal tracking.
"""

import time
import numpy as np
from typing import Dict, List, Any, Optional
from collections import defaultdict

class TemporalDevelopmentTracker:
    """
    Tracks development patterns across multiple time scales.

    Monitors daily, weekly, monthly, seasonal, and yearly development cycles
    to understand natural rhythms of growth and community evolution.
    """

    def __init__(self, model):
        """
        Initialize temporal development tracker.

        Args:
            model: The Neural Ecosystem model instance
        """
        self.model = model

        # Time scale tracking
        self.daily_patterns = []          # Every 24 steps
        self.weekly_patterns = []         # Every 168 steps
        self.monthly_patterns = []        # Every 672 steps
        self.seasonal_patterns = []       # Every 2184 steps
        self.yearly_patterns = []         # Every 8736 steps

        # Development phase tracking
        self.community_development_patterns = {
            'formation_phase': {'start': 0, 'characteristics': []},
            'exploration_phase': {'start': None, 'characteristics': []},
            'integration_phase': {'start': None, 'characteristics': []},
            'wisdom_sharing_phase': {'start': None, 'characteristics': []},
            'milestones': []
        }

        # Current state
        self.current_development_stage = 'forming_community'
        self.stage_transition_history = []

        # Natural rhythm detection
        self.energy_cycles = []
        self.wisdom_cycles = []
        self.social_cycles = []

        print("TemporalDevelopmentTracker initialized - tracking authentic development across time")
        print("Time scales: daily, weekly, monthly, seasonal, yearly patterns")

    def step(self, beings: List) -> None:
        """Process temporal development tracking for current step."""
        current_step = self.model.steps

        # Track development at different time scales
        self._track_daily_patterns(beings, current_step)
        self._track_weekly_patterns(beings, current_step)
        self._track_monthly_patterns(beings, current_step)
        self._track_seasonal_patterns(beings, current_step)
        self._track_yearly_patterns(beings, current_step)

        # Update development stage
        self._update_development_stage(beings, current_step)

        # Track natural rhythms
        self._track_natural_rhythms(beings, current_step)

        # Identify development milestones
        self._identify_milestones(beings, current_step)

    def _track_daily_patterns(self, beings: List, current_step: int) -> None:
        """Track daily development patterns (every 24 steps)."""
        if current_step % 24 == 0 and current_step > 0:
            daily_snapshot = self._create_development_snapshot(beings, current_step, 'daily')
            self.daily_patterns.append(daily_snapshot)

            # Keep only recent daily patterns
            if len(self.daily_patterns) > 30:  # Keep 30 days
                self.daily_patterns = self.daily_patterns[-30:]

    def _track_weekly_patterns(self, beings: List, current_step: int) -> None:
        """Track weekly development patterns (every 168 steps)."""
        if current_step % 168 == 0 and current_step > 0:
            weekly_snapshot = self._create_development_snapshot(beings, current_step, 'weekly')
            self.weekly_patterns.append(weekly_snapshot)

            # Keep only recent weekly patterns
            if len(self.weekly_patterns) > 12:  # Keep 12 weeks
                self.weekly_patterns = self.weekly_patterns[-12:]

    def _track_monthly_patterns(self, beings: List, current_step: int) -> None:
        """Track monthly development patterns (every 672 steps)."""
        if current_step % 672 == 0 and current_step > 0:
            monthly_snapshot = self._create_development_snapshot(beings, current_step, 'monthly')
            self.monthly_patterns.append(monthly_snapshot)

            # Keep only recent monthly patterns
            if len(self.monthly_patterns) > 12:  # Keep 12 months
                self.monthly_patterns = self.monthly_patterns[-12:]

    def _track_seasonal_patterns(self, beings: List, current_step: int) -> None:
        """Track seasonal development patterns (every 2184 steps)."""
        if current_step % 2184 == 0 and current_step > 0:
            seasonal_snapshot = self._create_development_snapshot(beings, current_step, 'seasonal')
            self.seasonal_patterns.append(seasonal_snapshot)

    def _track_yearly_patterns(self, beings: List, current_step: int) -> None:
        """Track yearly development patterns (every 8736 steps)."""
        if current_step % 8736 == 0 and current_step > 0:
            yearly_snapshot = self._create_development_snapshot(beings, current_step, 'yearly')
            self.yearly_patterns.append(yearly_snapshot)

    def _create_development_snapshot(self, beings: List, current_step: int, time_scale: str) -> Dict:
        """Create a development snapshot for the given time scale."""
        if not beings:
            return {
                'step': current_step,
                'time_scale': time_scale,
                'timestamp': time.time(),
                'being_count': 0,
                'metrics': {}
            }

        # Calculate aggregate metrics
        total_energy = sum(getattr(being, 'energy', 0) for being in beings)
        total_wisdom = sum(getattr(being, 'accumulated_wisdom', 0) for being in beings)
        total_connections = sum(getattr(being, 'social_connections', 0) for being in beings)

        # Growth stage distribution
        stage_distribution = defaultdict(int)
        for being in beings:
            stage = getattr(being, 'current_growth_stage', 'unknown')
            stage_distribution[stage] += 1

        # Neurochemical analysis
        avg_empathy = 0
        avg_curiosity = 0
        avg_contentment = 0

        for being in beings:
            if hasattr(being, 'neurochemical_system'):
                avg_empathy += getattr(being.neurochemical_system, 'empathy', 0.5)
                avg_curiosity += getattr(being.neurochemical_system, 'curiosity', 0.5)
                avg_contentment += getattr(being.neurochemical_system, 'contentment', 0.5)

        if beings:
            avg_empathy /= len(beings)
            avg_curiosity /= len(beings)
            avg_contentment /= len(beings)

        snapshot = {
            'step': current_step,
            'time_scale': time_scale,
            'timestamp': time.time(),
            'being_count': len(beings),
            'metrics': {
                'average_energy': total_energy / len(beings),
                'average_wisdom': total_wisdom / len(beings),
                'total_social_connections': total_connections,
                'stage_distribution': dict(stage_distribution),
                'neurochemical_averages': {
                    'empathy': avg_empathy,
                    'curiosity': avg_curiosity,
                    'contentment': avg_contentment
                }
            }
        }

        return snapshot

    def _update_development_stage(self, beings: List, current_step: int) -> None:
        """Update the current development stage of the community."""
        if not beings:
            return

        # Calculate community metrics for stage determination
        avg_wisdom = sum(getattr(being, 'accumulated_wisdom', 0) for being in beings) / len(beings)
        total_connections = sum(getattr(being, 'social_connections', 0) for being in beings)
        connection_density = total_connections / len(beings) if beings else 0

        # Determine appropriate stage
        new_stage = self.current_development_stage

        if current_step < 50:
            new_stage = 'forming_community'
        elif avg_wisdom < 1.0 and connection_density < 1.5:
            new_stage = 'early_exploration'
        elif avg_wisdom < 3.0 and connection_density < 3.0:
            new_stage = 'active_development'
        elif avg_wisdom < 6.0:
            new_stage = 'wisdom_integration'
        else:
            new_stage = 'mature_community'

        # Record stage transition if changed
        if new_stage != self.current_development_stage:
            transition = {
                'from_stage': self.current_development_stage,
                'to_stage': new_stage,
                'step': current_step,
                'timestamp': time.time(),
                'triggers': {
                    'average_wisdom': avg_wisdom,
                    'connection_density': connection_density,
                    'being_count': len(beings)
                }
            }

            self.stage_transition_history.append(transition)
            self.current_development_stage = new_stage

            print(f"Community development stage transition: {transition['from_stage']} → {new_stage}")

    def _track_natural_rhythms(self, beings: List, current_step: int) -> None:
        """Track natural rhythms in energy, wisdom, and social patterns."""
        if not beings:
            return

        # Energy rhythm tracking
        avg_energy = sum(getattr(being, 'energy', 50) for being in beings) / len(beings)
        self.energy_cycles.append({
            'step': current_step,
            'average_energy': avg_energy,
            'timestamp': time.time()
        })

        # Wisdom rhythm tracking
        avg_wisdom = sum(getattr(being, 'accumulated_wisdom', 0) for being in beings) / len(beings)
        self.wisdom_cycles.append({
            'step': current_step,
            'average_wisdom': avg_wisdom,
            'timestamp': time.time()
        })

        # Social rhythm tracking
        total_connections = sum(getattr(being, 'social_connections', 0) for being in beings)
        self.social_cycles.append({
            'step': current_step,
            'total_connections': total_connections,
            'connection_density': total_connections / len(beings),
            'timestamp': time.time()
        })

        # Keep rhythm data manageable
        max_rhythm_data = 500
        if len(self.energy_cycles) > max_rhythm_data:
            self.energy_cycles = self.energy_cycles[-max_rhythm_data:]
        if len(self.wisdom_cycles) > max_rhythm_data:
            self.wisdom_cycles = self.wisdom_cycles[-max_rhythm_data:]
        if len(self.social_cycles) > max_rhythm_data:
            self.social_cycles = self.social_cycles[-max_rhythm_data:]

    def _identify_milestones(self, beings: List, current_step: int) -> None:
        """Identify significant development milestones."""
        if not beings:
            return

        milestones = []

        # Wisdom milestones
        total_wisdom = sum(getattr(being, 'accumulated_wisdom', 0) for being in beings)
        if total_wisdom >= 10.0:  # Community wisdom milestone
            milestone_key = 'collective_wisdom_emergence'
            if milestone_key not in [m.get('type') for m in self.community_development_patterns.get('milestones', [])]:
                milestones.append({
                    'type': milestone_key,
                    'step': current_step,
                    'description': 'Community reached collective wisdom emergence',
                    'significance': 0.8,
                    'total_wisdom': total_wisdom
                })

        # Social connection milestones
        total_connections = sum(getattr(being, 'social_connections', 0) for being in beings)
        if total_connections >= len(beings) * 2:  # Average of 2 connections per being
            milestone_key = 'high_social_connectivity'
            if milestone_key not in [m.get('type') for m in self.community_development_patterns.get('milestones', [])]:
                milestones.append({
                    'type': milestone_key,
                    'step': current_step,
                    'description': 'Community achieved high social connectivity',
                    'significance': 0.7,
                    'total_connections': total_connections
                })

        # Diverse development stages milestone
        stage_distribution = defaultdict(int)
        for being in beings:
            stage = getattr(being, 'current_growth_stage', 'unknown')
            stage_distribution[stage] += 1

        if len(stage_distribution) >= 3:  # At least 3 different stages represented
            milestone_key = 'diverse_development_stages'
            if milestone_key not in [m.get('type') for m in self.community_development_patterns.get('milestones', [])]:
                milestones.append({
                    'type': milestone_key,
                    'step': current_step,
                    'description': 'Community shows diverse development stages',
                    'significance': 0.6,
                    'stage_diversity': len(stage_distribution)
                })

        # Add milestones to community patterns
        if milestones:
            if 'milestones' not in self.community_development_patterns:
                self.community_development_patterns['milestones'] = []
            self.community_development_patterns['milestones'].extend(milestones)

            for milestone in milestones:
                print(f"Development milestone achieved: {milestone['description']}")

    def get_current_stage(self) -> str:
        """Get current development stage."""
        return self.current_development_stage

    def get_development_report(self) -> Dict[str, Any]:
        """Get comprehensive development report."""
        return {
            'current_stage': self.current_development_stage,
            'stage_transitions': len(self.stage_transition_history),
            'patterns_tracked': {
                'daily': len(self.daily_patterns),
                'weekly': len(self.weekly_patterns),
                'monthly': len(self.monthly_patterns),
                'seasonal': len(self.seasonal_patterns),
                'yearly': len(self.yearly_patterns)
            },
            'natural_rhythms': {
                'energy_cycles': len(self.energy_cycles),
                'wisdom_cycles': len(self.wisdom_cycles),
                'social_cycles': len(self.social_cycles)
            },
            'milestones_achieved': len(self.community_development_patterns.get('milestones', [])),
            'recent_trends': self._analyze_recent_trends()
        }

    def _analyze_recent_trends(self) -> Dict[str, str]:
        """Analyze recent development trends."""
        trends = {}

        # Energy trend
        if len(self.energy_cycles) >= 5:
            recent_energy = [cycle['average_energy'] for cycle in self.energy_cycles[-5:]]
            energy_trend = (recent_energy[-1] - recent_energy[0]) / len(recent_energy)

            if energy_trend > 2.0:
                trends['energy'] = 'increasing'
            elif energy_trend < -2.0:
                trends['energy'] = 'decreasing'
            else:
                trends['energy'] = 'stable'

        # Wisdom trend
        if len(self.wisdom_cycles) >= 5:
            recent_wisdom = [cycle['average_wisdom'] for cycle in self.wisdom_cycles[-5:]]
            wisdom_trend = (recent_wisdom[-1] - recent_wisdom[0]) / len(recent_wisdom)

            if wisdom_trend > 0.1:
                trends['wisdom'] = 'growing'
            elif wisdom_trend < -0.1:
                trends['wisdom'] = 'declining'
            else:
                trends['wisdom'] = 'stable'

        # Social trend
        if len(self.social_cycles) >= 5:
            recent_connections = [cycle['connection_density'] for cycle in self.social_cycles[-5:]]
            social_trend = (recent_connections[-1] - recent_connections[0]) / len(recent_connections)

            if social_trend > 0.2:
                trends['social'] = 'connecting'
            elif social_trend < -0.2:
                trends['social'] = 'isolating'
            else:
                trends['social'] = 'stable'

        return trends

    def get_rhythm_analysis(self) -> Dict[str, Any]:
        """Get analysis of natural rhythms and cycles."""
        analysis = {
            'energy_rhythm': self._analyze_rhythm_pattern(self.energy_cycles, 'average_energy'),
            'wisdom_rhythm': self._analyze_rhythm_pattern(self.wisdom_cycles, 'average_wisdom'),
            'social_rhythm': self._analyze_rhythm_pattern(self.social_cycles, 'connection_density')
        }

        return analysis

    def _analyze_rhythm_pattern(self, cycle_data: List[Dict], metric_key: str) -> Dict[str, Any]:
        """Analyze rhythm patterns in cycle data."""
        if len(cycle_data) < 10:
            return {'status': 'insufficient_data'}

        values = [cycle[metric_key] for cycle in cycle_data[-20:]]  # Last 20 data points

        # Calculate basic statistics
        mean_value = np.mean(values)
        std_value = np.std(values)

        # Simple periodicity detection
        # Look for cyclical patterns in the data
        has_cycle = False
        cycle_length = None

        if len(values) >= 12:  # Need at least 12 points for cycle detection
            # Simple autocorrelation-like analysis
            correlations = []
            for lag in range(1, min(6, len(values)//2)):  # Check lags up to 6
                lagged_values = values[lag:]
                original_values = values[:-lag]

                if len(lagged_values) > 3:
                    correlation = np.corrcoef(original_values, lagged_values)[0, 1]
                    if not np.isnan(correlation):
                        correlations.append((lag, correlation))

            # Find strongest positive correlation
            if correlations:
                best_lag, best_correlation = max(correlations, key=lambda x: x[1])
                if best_correlation > 0.5:  # Threshold for detecting cycle
                    has_cycle = True
                    cycle_length = best_lag

        return {
            'mean': mean_value,
            'variability': std_value,
            'has_cycle': has_cycle,
            'cycle_length': cycle_length,
            'recent_trend': self._calculate_trend(values[-5:]) if len(values) >= 5 else 'stable'
        }

    def _calculate_trend(self, values: List[float]) -> str:
        """Calculate trend direction from recent values."""
        if len(values) < 2:
            return 'stable'

        # Simple linear trend
        x = np.arange(len(values))
        slope = np.polyfit(x, values, 1)[0]

        if slope > 0.1:
            return 'increasing'
        elif slope < -0.1:
            return 'decreasing'
        else:
            return 'stable'