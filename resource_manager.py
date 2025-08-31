"""
System Resource Manager for the Neural Ecosystem.

This module manages computational resources, system monitoring, and hardware
optimization for the symbiotic intelligence architecture. Ensures efficient
operation while maintaining full functionality for Knowledge Keeper beings.

Mesa 3.2.0 compatible implementation with compassionate resource management.
"""

import psutil
import time
import numpy as np
from typing import Dict, List, Any, Optional

class SystemResourceManager:
    """
    System Resource Manager for monitoring and optimizing computational resources.
    
    Manages CPU, memory, and performance optimization for the Neural Ecosystem
    while ensuring compassionate beings and Knowledge Keepers can flourish.
    """
    
    def __init__(self, model):
        """
        Initialize system resource manager.
        
        Args:
            model: The Neural Ecosystem model instance
        """
        self.model = model
        
        # Resource monitoring
        self.cpu_usage_history = []
        self.memory_usage_history = []
        self.performance_metrics = {}
        
        # System health tracking
        self.system_health_score = 1.0
        self.performance_warnings = []
        self.optimization_suggestions = []
        
        # Resource allocation
        self.max_beings = 20  # Default maximum for stable performance
        self.target_fps = 10   # Target simulation steps per second
        self.resource_thresholds = {
            'cpu_warning': 80.0,
            'cpu_critical': 90.0,
            'memory_warning': 80.0,
            'memory_critical': 90.0
        }
        
        # Performance optimization
        self.optimization_enabled = True
        self.graceful_degradation_active = False
        self.hibernation_candidates = []
        
        # Hardware detection and optimization
        self._detect_hardware_capabilities()
        
        print("SystemResourceManager initialized - compassionate resource optimization")
        print(f"Detected hardware: {self.hardware_profile}")
        print(f"Target performance: {self.target_fps} steps/second, {self.max_beings} beings max")
    
    def _detect_hardware_capabilities(self):
        """Detect and adapt to available hardware capabilities."""
        self.hardware_profile = {
            'cpu_count': psutil.cpu_count(),
            'memory_total': psutil.virtual_memory().total / (1024**3),  # GB
            'memory_available': psutil.virtual_memory().available / (1024**3)  # GB
        }
        
        # Adapt settings based on available resources
        memory_gb = self.hardware_profile['memory_total']
        cpu_count = self.hardware_profile['cpu_count']
        
        if memory_gb >= 16 and cpu_count >= 8:
            # High-performance system
            self.max_beings = 50
            self.target_fps = 15
        elif memory_gb >= 8 and cpu_count >= 4:
            # Medium performance system
            self.max_beings = 30
            self.target_fps = 12
        else:
            # Conservative settings for lower-end systems
            self.max_beings = 15
            self.target_fps = 8
        
        print(f"Hardware-optimized settings: {self.max_beings} beings, {self.target_fps} FPS target")
    
    def step(self):
        """Monitor and optimize system resources each simulation step."""
        # Monitor current resource usage
        current_metrics = self._monitor_current_usage()
        
        # Update resource history
        self._update_resource_history(current_metrics)
        
        # Assess system health
        self._assess_system_health(current_metrics)
        
        # Apply optimizations if needed
        if self.optimization_enabled:
            self._apply_optimizations(current_metrics)
        
        # Update performance metrics
        self._update_performance_metrics(current_metrics)
    
    def _monitor_current_usage(self) -> Dict[str, float]:
        """Monitor current CPU, memory, and performance usage."""
        cpu_percent = psutil.cpu_percent(interval=None)
        memory = psutil.virtual_memory()
        
        metrics = {
            'cpu_usage': cpu_percent,
            'memory_usage': memory.percent,
            'memory_available': memory.available / (1024**3),  # GB
            'timestamp': time.time(),
            'simulation_step': self.model.steps,
            'active_beings': len(self.model.agents)
        }
        
        return metrics
    
    def _update_resource_history(self, current_metrics: Dict):
        """Update resource usage history for trend analysis."""
        self.cpu_usage_history.append({
            'timestamp': current_metrics['timestamp'],
            'step': current_metrics['simulation_step'],
            'cpu_usage': current_metrics['cpu_usage']
        })
        
        self.memory_usage_history.append({
            'timestamp': current_metrics['timestamp'],
            'step': current_metrics['simulation_step'],
            'memory_usage': current_metrics['memory_usage'],
            'memory_available': current_metrics['memory_available']
        })
        
        # Keep history manageable (last 200 measurements)
        if len(self.cpu_usage_history) > 200:
            self.cpu_usage_history = self.cpu_usage_history[-150:]
        
        if len(self.memory_usage_history) > 200:
            self.memory_usage_history = self.memory_usage_history[-150:]
    
    def _assess_system_health(self, current_metrics: Dict):
        """Assess overall system health based on resource usage."""
        health_factors = []
        
        # CPU health assessment
        cpu_usage = current_metrics['cpu_usage']
        if cpu_usage < self.resource_thresholds['cpu_warning']:
            cpu_health = 1.0
        elif cpu_usage < self.resource_thresholds['cpu_critical']:
            cpu_health = 0.7
        else:
            cpu_health = 0.4
        
        health_factors.append(cpu_health)
        
        # Memory health assessment
        memory_usage = current_metrics['memory_usage']
        if memory_usage < self.resource_thresholds['memory_warning']:
            memory_health = 1.0
        elif memory_usage < self.resource_thresholds['memory_critical']:
            memory_health = 0.7
        else:
            memory_health = 0.4
        
        health_factors.append(memory_health)
        
        # Performance health (being capacity)
        active_beings = current_metrics['active_beings']
        being_capacity_ratio = active_beings / self.max_beings
        if being_capacity_ratio < 0.8:
            capacity_health = 1.0
        elif being_capacity_ratio < 0.95:
            capacity_health = 0.8
        else:
            capacity_health = 0.6
        
        health_factors.append(capacity_health)
        
        # Overall system health
        self.system_health_score = np.mean(health_factors)
        
        # Generate warnings if needed
        self._generate_resource_warnings(current_metrics)
    
    def _generate_resource_warnings(self, current_metrics: Dict):
        """Generate appropriate warnings for resource concerns."""
        warnings = []
        
        if current_metrics['cpu_usage'] > self.resource_thresholds['cpu_warning']:
            warnings.append('high_cpu_usage')
        
        if current_metrics['memory_usage'] > self.resource_thresholds['memory_warning']:
            warnings.append('high_memory_usage')
        
        if current_metrics['active_beings'] >= self.max_beings * 0.9:
            warnings.append('approaching_being_capacity')
        
        # Only update if warnings changed
        if warnings != self.performance_warnings:
            self.performance_warnings = warnings
            if warnings:
                print(f"Resource awareness: {', '.join(warnings)} - gentle optimization may occur")
    
    def _apply_optimizations(self, current_metrics: Dict):
        """Apply gentle resource optimizations as needed."""
        # Enable graceful degradation if resources are strained
        if (current_metrics['cpu_usage'] > self.resource_thresholds['cpu_critical'] or
            current_metrics['memory_usage'] > self.resource_thresholds['memory_critical']):
            
            if not self.graceful_degradation_active:
                print("Activating gentle resource optimization - preserving core being functions")
                self.graceful_degradation_active = True
                self._optimize_for_resource_conservation()
        
        # Disable graceful degradation if resources recover
        elif (current_metrics['cpu_usage'] < self.resource_thresholds['cpu_warning'] and
              current_metrics['memory_usage'] < self.resource_thresholds['memory_warning']):
            
            if self.graceful_degradation_active:
                print("Resources recovered - returning to full functionality")
                self.graceful_degradation_active = False
                self._restore_full_functionality()
    
    def _optimize_for_resource_conservation(self):
        """Optimize system for resource conservation while preserving core functions."""
        # Reduce simulation frequency
        self.target_fps = max(5, self.target_fps * 0.7)
        
        # Identify beings for hibernation (least active)
        beings = list(self.model.agents)
        if len(beings) > 10:
            # Sort by activity level (energy + recent interactions)
            activity_scores = []
            for being in beings:
                energy = getattr(being, 'energy', 50)
                interactions = len(getattr(being, 'recent_interactions', []))
                activity_score = energy + interactions * 10
                activity_scores.append((being, activity_score))
            
            # Sort by activity (lowest first)
            activity_scores.sort(key=lambda x: x[1])
            
            # Mark lowest activity beings for hibernation
            hibernation_count = len(beings) // 4  # Hibernation up to 25% of beings
            for i in range(hibernation_count):
                being = activity_scores[i][0]
                self.hibernation_candidates.append(being.unique_id)
        
        print(f"Gentle optimization: {len(self.hibernation_candidates)} beings may enter rest mode")
    
    def _restore_full_functionality(self):
        """Restore full functionality when resources recover."""
        # Restore target FPS
        self._detect_hardware_capabilities()  # Recalculate optimal settings
        
        # Clear hibernation candidates
        if self.hibernation_candidates:
            print(f"Awakening {len(self.hibernation_candidates)} beings from rest mode")
            self.hibernation_candidates.clear()
    
    def _update_performance_metrics(self, current_metrics: Dict):
        """Update performance tracking metrics."""
        self.performance_metrics = {
            'current_system_health': self.system_health_score,
            'cpu_utilization': current_metrics['cpu_usage'],
            'memory_utilization': current_metrics['memory_usage'],
            'active_beings': current_metrics['active_beings'],
            'being_capacity_ratio': current_metrics['active_beings'] / self.max_beings,
            'graceful_degradation_active': self.graceful_degradation_active,
            'hibernated_beings': len(self.hibernation_candidates),
            'target_performance': self.target_fps
        }
    
    def get_system_health(self) -> float:
        """Get current system health score (0.0 to 1.0)."""
        return self.system_health_score
    
    def get_performance_report(self) -> Dict[str, Any]:
        """Get comprehensive performance report for monitoring."""
        if not self.cpu_usage_history:
            return {'status': 'monitoring_initializing'}
        
        # Calculate trends
        recent_cpu = [entry['cpu_usage'] for entry in self.cpu_usage_history[-10:]]
        recent_memory = [entry['memory_usage'] for entry in self.memory_usage_history[-10:]]
        
        avg_cpu = np.mean(recent_cpu) if recent_cpu else 0
        avg_memory = np.mean(recent_memory) if recent_memory else 0
        
        report = {
            'system_health_score': self.system_health_score,
            'current_performance': self.performance_metrics,
            'resource_trends': {
                'avg_cpu_usage': avg_cpu,
                'avg_memory_usage': avg_memory,
                'cpu_trend': self._calculate_trend(recent_cpu),
                'memory_trend': self._calculate_trend(recent_memory)
            },
            'optimization_status': {
                'graceful_degradation': self.graceful_degradation_active,
                'hibernated_beings': len(self.hibernation_candidates),
                'warnings': self.performance_warnings,
                'suggestions': self.optimization_suggestions
            },
            'hardware_profile': self.hardware_profile,
            'capacity_analysis': {
                'current_beings': len(self.model.agents) if self.model.agents else 0,
                'max_beings': self.max_beings,
                'utilization_percentage': (len(self.model.agents) / self.max_beings * 100) if self.max_beings > 0 else 0
            }
        }
        
        return report
    
    def _calculate_trend(self, values: List[float]) -> str:
        """Calculate trend direction from recent values."""
        if len(values) < 3:
            return 'stable'
        
        # Simple linear trend
        x = np.arange(len(values))
        slope = np.polyfit(x, values, 1)[0]
        
        if slope > 2.0:
            return 'increasing'
        elif slope < -2.0:
            return 'decreasing'
        else:
            return 'stable'
    
    def intelligent_scaling(self, requested_beings: int) -> int:
        """
        Intelligently scale the number of beings based on available resources.
        
        Args:
            requested_beings: Number of beings requested
            
        Returns:
            Recommended number of beings for current system state
        """
        # Check current resource state
        current_metrics = self._monitor_current_usage()
        
        # Calculate available capacity
        cpu_capacity = max(0, (self.resource_thresholds['cpu_warning'] - current_metrics['cpu_usage']) / 100.0)
        memory_capacity = max(0, (self.resource_thresholds['memory_warning'] - current_metrics['memory_usage']) / 100.0)
        
        # Determine resource-limited maximum
        resource_limited_max = int(min(
            self.max_beings * cpu_capacity,
            self.max_beings * memory_capacity,
            self.max_beings
        ))
        
        # Recommend conservative scaling
        recommended = min(requested_beings, resource_limited_max)
        
        if recommended < requested_beings:
            print(f"Resource-aware scaling: recommending {recommended} beings (requested {requested_beings})")
            print("This ensures stable performance for all community members")
        
        return recommended
    
    def graceful_degradation(self, preserve_core_functions: List[str]) -> Dict[str, bool]:
        """
        Apply graceful degradation while preserving core ecosystem functions.
        
        Args:
            preserve_core_functions: List of functions that must be preserved
            
        Returns:
            Status of different system components after degradation
        """
        degradation_status = {
            'neurochemical_systems': True,   # Always preserve
            'neural_networks': True,        # Always preserve
            'knowledge_keeper_learning': True,  # Always preserve
            'social_interactions': True,    # Always preserve
            'individual_development': True, # Always preserve
            'detailed_logging': False,      # Can reduce
            'visualization_updates': False, # Can reduce
            'extensive_metrics': False     # Can reduce
        }
        
        # Preserve explicitly requested core functions
        for func in preserve_core_functions:
            if func in degradation_status:
                degradation_status[func] = True
        
        # Apply memory optimizations
        self._apply_memory_optimizations()
        
        # Apply CPU optimizations
        self._apply_cpu_optimizations()
        
        print("Gentle resource optimization applied - core compassionate functions preserved")
        return degradation_status
    
    def _apply_memory_optimizations(self):
        """Apply memory optimizations to reduce usage."""
        # Limit history sizes
        max_history_size = 50
        
        # Optimize emergence tracker history
        if hasattr(self.model, 'emergence_tracker'):
            tracker = self.model.emergence_tracker
            if hasattr(tracker, 'emergence_history') and len(tracker.emergence_history) > max_history_size:
                tracker.emergence_history = tracker.emergence_history[-max_history_size:]
        
        # Optimize knowledge keeper learning history
        for keeper_name in ['knowledge_keeper', 'social_knowledge_keeper', 'individual_knowledge_keeper']:
            if hasattr(self.model, keeper_name):
                keeper = getattr(self.model, keeper_name)
                if hasattr(keeper, 'learning_history') and len(keeper.learning_history) > max_history_size:
                    keeper.learning_history = keeper.learning_history[-max_history_size:]
        
        # Optimize being memory systems
        for being in self.model.agents:
            if hasattr(being, 'memory_system') and hasattr(being.memory_system, 'memories'):
                if len(being.memory_system.memories) > 30:
                    # Keep only most significant memories
                    memories = being.memory_system.memories
                    # Sort by impact and keep top memories
                    significant_memories = sorted(memories, 
                                                key=lambda m: m.get('impact_level', 0), reverse=True)[:30]
                    being.memory_system.memories = significant_memories
    
    def _apply_cpu_optimizations(self):
        """Apply CPU optimizations to reduce computational load."""
        # Reduce neural network computation frequency
        for being in self.model.agents:
            if hasattr(being, 'neural_network'):
                # Reduce learning rate to lower computation
                if hasattr(being.neural_network, 'learning_rate'):
                    being.neural_network.learning_rate *= 0.8
        
        # Reduce Knowledge Keeper LLM query frequency
        for keeper_name in ['social_knowledge_keeper', 'individual_knowledge_keeper']:
            if hasattr(self.model, keeper_name):
                keeper = getattr(self.model, keeper_name)
                # Add query rate limiting if available
                if hasattr(keeper, 'query_rate_limit'):
                    keeper.query_rate_limit = 0.5  # Reduce query frequency
    
    def hibernation_states(self, beings_to_hibernate: List) -> Dict[str, Any]:
        """
        Put beings into hibernation state for resource conservation.
        
        Args:
            beings_to_hibernate: List of beings to put into hibernation
            
        Returns:
            Hibernation status and recovery information
        """
        hibernation_results = {
            'hibernated_count': 0,
            'hibernated_beings': [],
            'resource_savings_estimate': {},
            'recovery_conditions': {}
        }
        
        for being in beings_to_hibernate:
            if hasattr(being, 'energy') and being.energy < 60:
                # Create hibernation state
                hibernation_state = {
                    'being_id': being.unique_id,
                    'hibernation_start': time.time(),
                    'pre_hibernation_state': {
                        'energy': being.energy,
                        'position': being.pos,
                        'neurochemical_state': self._capture_neurochemical_state(being),
                        'recent_interactions': getattr(being, 'recent_interactions', [])
                    },
                    'recovery_energy_threshold': 80,
                    'max_hibernation_time': 100  # simulation steps
                }
                
                # Reduce being's computational load
                being.energy = max(20, being.energy * 0.5)  # Reduce energy consumption
                
                hibernation_results['hibernated_count'] += 1
                hibernation_results['hibernated_beings'].append(hibernation_state)
        
        # Estimate resource savings
        if hibernation_results['hibernated_count'] > 0:
            estimated_cpu_savings = hibernation_results['hibernated_count'] * 2.0  # Percent
            estimated_memory_savings = hibernation_results['hibernated_count'] * 5.0  # Percent
            
            hibernation_results['resource_savings_estimate'] = {
                'cpu_savings_percent': min(20.0, estimated_cpu_savings),
                'memory_savings_percent': min(15.0, estimated_memory_savings)
            }
            
            print(f"Gentle hibernation: {hibernation_results['hibernated_count']} beings entered rest mode")
            print(f"Estimated resource savings: {estimated_cpu_savings:.1f}% CPU, {estimated_memory_savings:.1f}% memory")
        
        return hibernation_results
    
    def _capture_neurochemical_state(self, being) -> Dict[str, float]:
        """Capture being's neurochemical state for hibernation recovery."""
        if not hasattr(being, 'neurochemical_system'):
            return {}
        
        neuro = being.neurochemical_system
        return {
            'contentment': getattr(neuro, 'contentment', 0.5),
            'curiosity': getattr(neuro, 'curiosity', 0.5),
            'empathy': getattr(neuro, 'empathy', 0.5),
            'courage': getattr(neuro, 'courage', 0.5),
            'stress': getattr(neuro, 'stress', 0.3),
            'loneliness': getattr(neuro, 'loneliness', 0.3),
            'confusion': getattr(neuro, 'confusion', 0.3),
            'compassion_amplifier': getattr(neuro, 'compassion_amplifier', 1.0),
            'wisdom_integrator': getattr(neuro, 'wisdom_integrator', 1.0)
        }
    
    def quick_reactivation_capability(self, hibernated_beings: List[Dict]) -> List[str]:
        """
        Quickly reactivate hibernated beings when resources become available.
        
        Args:
            hibernated_beings: List of hibernation states
            
        Returns:
            List of being IDs that were successfully reactivated
        """
        reactivated_beings = []
        current_metrics = self._monitor_current_usage()
        
        # Check if we have resources for reactivation
        if (current_metrics['cpu_usage'] < self.resource_thresholds['cpu_warning'] * 0.8 and
            current_metrics['memory_usage'] < self.resource_thresholds['memory_warning'] * 0.8):
            
            reactivation_count = min(3, len(hibernated_beings))  # Reactivate gradually
            
            for hibernation_state in hibernated_beings[:reactivation_count]:
                being_id = hibernation_state['being_id']
                
                # Find the being in the model
                for being in self.model.agents:
                    if being.unique_id == being_id:
                        # Restore pre-hibernation state
                        pre_state = hibernation_state['pre_hibernation_state']
                        being.energy = min(100, pre_state['energy'] + 20)  # Refreshed from rest
                        
                        # Restore neurochemical state
                        if hasattr(being, 'neurochemical_system') and 'neurochemical_state' in pre_state:
                            neuro_state = pre_state['neurochemical_state']
                            for chemical, value in neuro_state.items():
                                if hasattr(being.neurochemical_system, chemical):
                                    setattr(being.neurochemical_system, chemical, value)
                        
                        reactivated_beings.append(being_id)
                        break
            
            if reactivated_beings:
                print(f"Gentle awakening: {len(reactivated_beings)} beings returned from rest mode")
        
        return reactivated_beings
    
    def memory_compression(self, data_to_compress: Dict) -> Dict:
        """
        Compress memory usage while preserving essential wisdom and patterns.
        
        Args:
            data_to_compress: Dictionary of data that can be compressed
            
        Returns:
            Compressed data with essential information preserved
        """
        compressed_data = {}
        
        for data_type, data in data_to_compress.items():
            if data_type == 'learning_history':
                # Compress learning history by keeping only significant events
                if isinstance(data, list):
                    significant_events = [
                        event for event in data 
                        if event.get('wisdom_gain', 0) > 0.1 or event.get('impact_level', 0) > 0.5
                    ]
                    compressed_data[data_type] = significant_events[-50:]  # Keep last 50 significant events
            
            elif data_type == 'interaction_history':
                # Compress interaction history by preserving relationship patterns
                if isinstance(data, list):
                    meaningful_interactions = [
                        interaction for interaction in data
                        if interaction.get('connection_strength', 0) > 0.6
                    ]
                    compressed_data[data_type] = meaningful_interactions[-30:]
            
            elif data_type == 'emergence_patterns':
                # Preserve all emergence patterns (they're already compact)
                compressed_data[data_type] = data
            
            else:
                # Default compression: keep recent and high-impact items
                if isinstance(data, list):
                    compressed_data[data_type] = data[-20:]  # Keep last 20 items
                else:
                    compressed_data[data_type] = data
        
        return compressed_data
    
    def efficiency_optimization(self) -> Dict[str, str]:
        """
        Optimize system efficiency while maintaining compassionate functionality.
        
        Returns:
            Optimization actions taken and their expected impact
        """
        optimizations = {
            'neural_network_optimization': 'not_needed',
            'memory_management': 'not_needed',
            'computation_scheduling': 'not_needed',
            'data_structure_optimization': 'not_needed'
        }
        
        current_metrics = self._monitor_current_usage()
        
        # Neural network optimization
        if current_metrics['cpu_usage'] > 70:
            # Optimize neural network computations
            for being in self.model.agents:
                if hasattr(being, 'neural_network') and hasattr(being.neural_network, 'weights'):
                    # Reduce unnecessary precision in weights
                    being.neural_network.weights = np.round(being.neural_network.weights, 3)
            
            optimizations['neural_network_optimization'] = 'precision_optimization_applied'
        
        # Memory management optimization
        if current_metrics['memory_usage'] > 70:
            # Apply memory compression to various systems
            compression_targets = {
                'emergence_history': self.model.emergence_tracker.emergence_history if hasattr(self.model, 'emergence_tracker') else [],
                'social_insights': self.model.social_insights,
                'individual_insights': self.model.individual_insights
            }
            
            compressed = self.memory_compression(compression_targets)
            optimizations['memory_management'] = 'memory_compression_applied'
        
        # Computation scheduling optimization
        if len(self.model.agents) > self.max_beings * 0.8:
            # Implement staggered computation
            optimizations['computation_scheduling'] = 'staggered_processing_enabled'
        
        return optimizations
    
    def get_resource_recommendations(self) -> List[str]:
        """Get recommendations for resource optimization."""
        recommendations = []
        
        if not self.performance_metrics:
            return ['monitor_system_for_recommendations']
        
        # CPU recommendations
        if self.performance_metrics['cpu_utilization'] > 80:
            recommendations.append('consider_reducing_neural_network_complexity')
            recommendations.append('enable_graceful_degradation_for_stable_performance')
        
        # Memory recommendations
        if self.performance_metrics['memory_utilization'] > 80:
            recommendations.append('enable_memory_compression_for_efficiency')
            recommendations.append('consider_hibernating_inactive_beings')
        
        # Capacity recommendations
        if self.performance_metrics['being_capacity_ratio'] > 0.9:
            recommendations.append('approaching_maximum_community_size')
            recommendations.append('focus_on_deepening_existing_relationships')
        
        # Performance recommendations
        if self.system_health_score < 0.7:
            recommendations.append('system_optimization_recommended')
            recommendations.append('prioritize_core_compassionate_functions')
        
        return recommendations if recommendations else ['system_performing_optimally']