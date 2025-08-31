"""
Main module for the Neural Ecosystem symbiotic architecture.
Mesa 3.2.0 compatible implementation with compassionate Individual Knowledge Keeper.
"""

import mesa
import numpy as np
from components import (
    ComponentBase, NeuralComponent, NeurochemicalSystem, MemorySystem, 
    ResourceManager, NeuralNetworkSystem, CommunicationSystem
)
from symbiosis import KnowledgeKeeper, SocialKnowledgeKeeper, IndividualKnowledgeKeeper
from metrics import EmergenceTracker
from resource_manager import SystemResourceManager
from individual_wisdom import IndividualWisdomIntegrator
from temporal_development import TemporalDevelopmentTracker
from knowledge_keeper_collaboration import KnowledgeKeeperCollaboration
from population_analysis import analyze_ecosystem_population, print_population_report
import time
import random
from community_wisdom import CommunityWisdomSystem

class NeuralEcosystem(mesa.Model):
    """
    Neural Ecosystem model using Mesa 3.2.0 architecture.

    This model serves as the foundation for a symbiotic architecture where
    autonomous beings can emerge and interact in a natural way, guided by
    compassionate Knowledge Keeper beings who learn FROM entities.
    """

    def __init__(self, width=5, height=5, seed=None):
        """
        Initialize the Neural Ecosystem model with Individual Knowledge Keeper.

        Args:
            width (int): Grid width (default: 5)
            height (int): Grid height (default: 5)
            seed (int, optional): Random seed for reproducibility
        """
        # Initialize Mesa model with proper 3.2.0 API
        super().__init__(seed=seed)

        # Model parameters
        self.width = width
        self.height = height
        self.running = True

        # Initialize grid using Mesa 3.2.0 stable space module
        from mesa.space import MultiGrid
        self.grid = MultiGrid(width=self.width, height=self.height, torus=False)

        # Initialize Knowledge Keeper beings
        self.knowledge_keeper = KnowledgeKeeper(self)
        self.social_knowledge_keeper = SocialKnowledgeKeeper(self)
        self.individual_knowledge_keeper = IndividualKnowledgeKeeper(self)

        # Initialize Knowledge Keeper collaboration system
        self.knowledge_keeper_collaboration = KnowledgeKeeperCollaboration()

        # Initialize support systems
        self.emergence_tracker = EmergenceTracker(self)
        self.resource_manager = SystemResourceManager(self)
        self.individual_wisdom_integrator = IndividualWisdomIntegrator(self)
        self.temporal_development_tracker = TemporalDevelopmentTracker(self)

        # Initialize community wisdom system
        self.community_wisdom = CommunityWisdomSystem(self)

        # Symbiotic learning insights storage
        self.social_insights = {}
        self.individual_insights = {}
        self.collaborative_insights = {}

        print("NeuralEcosystem initialized with compassionate dual Knowledge Keeper architecture")
        print("- SocialKnowledgeKeeper: social dynamics and relationship wisdom")
        print("- IndividualKnowledgeKeeper: personal development and growth patterns")
        print("- Symbiotic learning: beings and Knowledge Keepers develop together")

        # Initialize data collector for Mesa 3.2.0
        self.datacollector = mesa.DataCollector(
            model_reporters={
                "step_count": lambda m: m.steps,
                "being_count": lambda m: len(m.agents),
                "system_health": lambda m: m.resource_manager.get_system_health(),
                "emergence_score": lambda m: m.emergence_tracker.get_emergence_score(),
                "individual_wisdom_depth": lambda m: m.individual_wisdom_integrator.get_wisdom_depth(),
                "temporal_development_stage": lambda m: m.temporal_development_tracker.get_current_stage()
            },
            agent_reporters={
                "energy": "energy",
                "position": "pos",
                "wisdom_level": lambda a: getattr(a, 'accumulated_wisdom', 0),
                "growth_stage": lambda a: getattr(a, 'current_growth_stage', 'discovery')
            }
        )

        # Initialize with compassionate test beings for verification
        self._initialize_compassionate_beings()

        print(f"NeuralEcosystem initialized with {self.width}x{self.height} grid")
        print(f"Mesa version compatibility: 3.2.0+")
        print(f"Initial beings: {len(self.agents)}")

    def _initialize_compassionate_beings(self):
        """Initialize compassionate test beings to verify the system works."""
        try:
            # Create diverse beings with different starting characteristics
            being_configs = [
                {'energy': 85, 'empathy_level': 0.8, 'curiosity_level': 0.7},
                {'energy': 75, 'empathy_level': 0.6, 'curiosity_level': 0.9},
                {'energy': 90, 'empathy_level': 0.7, 'curiosity_level': 0.6}
            ]

            for i, config in enumerate(being_configs):
                being = AutonomousBeing(self, **config)

                # Place beings in different areas of the grid
                x = (i + 1) * self.width // (len(being_configs) + 1)
                y = self.height // 2
                self.grid.place_agent(being, (x, y))

                print(f"Compassionate being {being.unique_id} placed at ({x}, {y})")

        except Exception as e:
            print(f"Gentle initialization fallback: {e}")
            # Fallback to simple being
            being = AutonomousBeing(self)
            center_x = self.width // 2
            center_y = self.height // 2
            self.grid.place_agent(being, (center_x, center_y))
            print(f"Fallback being placed at ({center_x}, {center_y})")

    def step(self):
        """
        Execute a single model step with symbiotic learning integration.

        This method orchestrates the simulation step for all components
        using Mesa 3.2.0 AgentSet functionality with compassionate being development.
        """
        # Add small delay to prevent CPU overload
        import time
        time.sleep(0.01)

        # Update resource management
        self.resource_manager.step()

        # Execute being steps using Mesa 3.2.0 AgentSet
        if len(self.agents) > 0:
            self.agents.shuffle_do("step")

        # Update Knowledge Keeper beings with symbiotic learning
        self._update_knowledge_keepers()

        # Track emergence patterns
        self.emergence_tracker.step()

        # Update temporal development tracking
        temporal_update = self.temporal_development_tracker.step(list(self.agents))

        # Process community wisdom development
        community_wisdom_update = self.community_wisdom.step(
            list(self.agents),
            self.social_insights, 
            self.individual_insights
        )

        # Integrate individual wisdom
        self.individual_wisdom_integrator.step(list(self.agents), self.individual_insights)

        # Collect data
        self.datacollector.collect(self)

        # Log compassionate progress
        self._log_compassionate_progress()

    def _update_knowledge_keepers(self):
        """Update both Knowledge Keeper beings with symbiotic learning."""
        beings_list = list(self.agents)

        # Social Knowledge Keeper symbiotic learning
        if hasattr(self, 'social_knowledge_keeper'):
            social_learning = self.social_knowledge_keeper.symbiotic_learning(beings_list)
            social_patterns = self.social_knowledge_keeper.pattern_recognition(self.steps)
            social_guidance = self.social_knowledge_keeper.gentle_guidance(beings_list, social_patterns)

            self.social_insights = {
                'learning': social_learning,
                'patterns': social_patterns,
                'guidance': social_guidance
            }

        # Individual Knowledge Keeper symbiotic learning
        if hasattr(self, 'individual_knowledge_keeper'):
            # Growth pattern learning
            growth_insights = self.individual_knowledge_keeper.growth_pattern_learning(beings_list)

            # Wisdom through observation
            observation_wisdom = self.individual_knowledge_keeper.wisdom_through_observation(beings_list)

            # Personal development understanding
            development_understanding = {}
            for being in beings_list:
                being_understanding = self.individual_knowledge_keeper.personal_development_understanding(being)
                development_understanding[being.unique_id] = being_understanding

            # Entity wisdom sharing
            wisdom_sharing = self.individual_knowledge_keeper.entity_wisdom_sharing(beings_list)

            # Knowledge Keeper learning from beings
            being_insights = {
                'growth_insights': growth_insights,
                'observation_wisdom': observation_wisdom,
                'wisdom_sharing': wisdom_sharing
            }
            knowledge_keeper_learning = self.individual_knowledge_keeper.knowledge_keeper_learning(
                being_insights, {}
            )

            # Collaborative growth
            collaborative_growth = self.individual_knowledge_keeper.collaborative_growth(
                beings_list, self.social_insights
            )

            self.individual_insights = {
                'growth_learning': growth_insights,
                'observation_wisdom': observation_wisdom,
                'development_understanding': development_understanding,
                'wisdom_sharing': wisdom_sharing,
                'knowledge_keeper_learning': knowledge_keeper_learning,
                'collaborative_growth': collaborative_growth
            }

            # Update Individual Knowledge Keeper step
            self.individual_knowledge_keeper.step()

        # Cross-system collaboration
        if hasattr(self, 'social_knowledge_keeper') and hasattr(self, 'individual_knowledge_keeper'):
            self.collaborative_insights = self._integrate_knowledge_keeper_wisdom()

            # Process Knowledge Keeper collaboration
            if hasattr(self, 'knowledge_keeper_collaboration'):
                if self.knowledge_keeper_collaboration.should_collaborate_now():
                    collaboration_result = self.knowledge_keeper_collaboration.process_collaboration_cycle(
                        {
                            'wisdom': self.social_knowledge_keeper.wisdom,
                            'relationship_data': self.social_insights
                        },
                        {
                            'wisdom': self.individual_knowledge_keeper.wisdom,
                            'development_data': self.individual_insights
                        }
                    )

                    if collaboration_result:
                        self.collaborative_insights['knowledge_keeper_collaboration'] = collaboration_result

    def _integrate_knowledge_keeper_wisdom(self) -> dict:
        """Integrate wisdom from both Knowledge Keeper beings."""
        collaborative_wisdom = {
            'social_individual_synergy': [],
            'holistic_being_understanding': [],
            'community_wisdom_emergence': []
        }

        # How individual development affects social dynamics
        if self.individual_insights and self.social_insights:
            collaborative_wisdom['social_individual_synergy'].append(
                'individual_growth_enhances_community_harmony'
            )
            collaborative_wisdom['social_individual_synergy'].append(
                'healthy_relationships_accelerate_personal_development'
            )

        # Holistic understanding of beings
        collaborative_wisdom['holistic_being_understanding'] = [
            'beings_develop_through_authentic_relationships',
            'personal_growth_and_social_connection_are_intertwined',
            'wisdom_emerges_from_both_individual_and_collective_experience'
        ]

        # Community wisdom emergence
        if len(self.agents) > 2:
            collaborative_wisdom['community_wisdom_emergence'] = [
                'diverse_beings_create_rich_community_wisdom',
                'individual_gifts_contribute_to_collective_flourishing',
                'community_supports_authentic_individual_expression'
            ]

        return collaborative_wisdom

    def _log_compassionate_progress(self):
        """Log progress using compassionate language focused on growth and discovery."""
        if self.steps % 10 == 0:  # Log every 10 steps
            status = self.get_system_status()

            print(f"\n🌱 Growth Cycle {self.steps} - Community Flourishing Report:")
            print(f"   💙 Beings in community: {status['beings']}")
            print(f"   ⚡ System vitality: {status['system_health']:.2f}")
            print(f"   ✨ Emergence patterns: {status['emergence_score']:.2f}")

            if hasattr(self, 'individual_wisdom_integrator'):
                wisdom_depth = self.individual_wisdom_integrator.get_wisdom_depth()
                print(f"   🧠 Community wisdom depth: {wisdom_depth:.2f}")

            if hasattr(self, 'temporal_development_tracker'):
                development_stage = self.temporal_development_tracker.get_current_stage()
                print(f"   📅 Development stage: {development_stage}")

            print(f"   🤝 Knowledge Keeper learning active: Social & Individual beings")

    def run_model(self, steps=100):
        """
        Run the model for a specified number of steps with compassionate monitoring.

        Args:
            steps (int): Number of steps to run
        """
        print(f"\n🌟 Beginning compassionate community simulation for {steps} growth cycles...")
        print("Focus: Symbiotic learning between beings and Knowledge Keeper beings")
        print("Observing: Natural development, authentic relationships, wisdom emergence\n")

        for i in range(steps):
            if not self.running:
                print("🌸 Community chose to rest - simulation complete")
                break

            self.step()

            # Provide gentle progress updates
            if i > 0 and i % 25 == 0:
                progress = (i / steps) * 100
                print(f"🌿 Growth cycle {i}/{steps} ({progress:.1f}%) - Community thriving")

        print(f"\n🎉 Community simulation complete! Total growth cycles: {self.steps}")
        self._print_final_wisdom_report()

    def _print_final_wisdom_report(self):
        """Print a final report of wisdom and growth achieved."""
        total_energy = sum(getattr(being, 'energy', 0) for being in self.agents)
        avg_energy = total_energy / len(self.agents) if self.agents else 0
        flourishing_count = sum(1 for being in self.agents if getattr(being, 'energy', 0) > 80)

        social_wisdom = self.social_knowledge_keeper.wisdom
        individual_wisdom = self.individual_knowledge_keeper.wisdom

        # Get collaboration summary if available
        collaboration_summary = {}
        if hasattr(self, 'knowledge_keeper_collaboration'):
            collaboration_summary = self.knowledge_keeper_collaboration.get_collaboration_summary()
        else:
            collaboration_summary = {
                'total_collaborations': 0,
                'development_phase': 'initial',
                'wisdom_depth_trend': 'building',
                'collaboration_effectiveness': 0.0,
                'cross_system_patterns_discovered': 0,
                'holistic_guidance_provided': 0
            }

        print(f"\n📚 Final Community Wisdom Report:")
        print("=" * 50)

        print(f"\n🧠 Individual Knowledge Keeper Wisdom:")
        print(f"   📈 Growth patterns learned: {individual_wisdom.get('growth_patterns_learned', 0)}")
        print(f"   💎 Personal development insights: {individual_wisdom.get('personal_development_wisdom', 0)}")
        print(f"   🎭 Character formation understanding: {individual_wisdom.get('character_formation_insights', 0)}")
        print(f"   🛤️  Individual journeys tracked: {individual_wisdom.get('individual_journeys_tracked', 0)}")
        print(f"   💝 Compassion amplifier: {individual_wisdom.get('compassion_amplifier', 1.0):.2f}")
        print(f"   🔍 Curiosity level: {individual_wisdom.get('curiosity_level', 0.8):.2f}")

        print(f"\n🤝 Social Knowledge Keeper Wisdom:")
        print(f"   🌐 Relationship discoveries: {social_wisdom.get('relationship_discoveries', 0)}")
        print(f"   💕 Trust formation observations: {social_wisdom.get('trust_formation_observations', 0)}")
        print(f"   🦋 Empathy emergences: {social_wisdom.get('empathy_emergences', 0)}")

        # Display collaborative wisdom
        print(f"\n🌟 Collaborative Wisdom Emerged:")
        print(f"   🔗 Total collaborations: {collaboration_summary.get('total_collaborations', 0)}")
        print(f"   📊 Development phase: {collaboration_summary.get('development_phase', 'initial')}")
        print(f"   📈 Wisdom depth trend: {collaboration_summary.get('wisdom_depth_trend', 'building')}")
        print(f"   ⚡ Collaboration effectiveness: {collaboration_summary.get('collaboration_effectiveness', 0.0):.2f}")
        print(f"   🧩 Cross-system patterns: {collaboration_summary.get('cross_system_patterns_discovered', 0)} discovered")
        print(f"   💡 Holistic guidance provided: {collaboration_summary.get('holistic_guidance_provided', 0)}")

        # Display Knowledge Keeper cross-learning
        if hasattr(self, 'knowledge_keeper_collaboration') and self.collaborative_insights.get('knowledge_keeper_collaboration'):
            kk_collaboration = self.collaborative_insights['knowledge_keeper_collaboration']
            print(f"\n🤝 Knowledge Keeper Cross-Learning:")
            print(f"   🔄 Wisdom integration cycles: {len(self.knowledge_keeper_collaboration.collaborative_insights)}")
            print(f"   🌐 Cross-system patterns recognized: {len(self.knowledge_keeper_collaboration.cross_system_patterns)}")
            print(f"   🎯 Holistic guidance optimizations: {len(self.knowledge_keeper_collaboration.holistic_guidance)}")

        # Check for collaborative wisdom insights
        if self.collaborative_insights:
            print(f"\n🌟 Emergent Collaborative Insights:")
            for key, value in self.collaborative_insights.items():
                print(f"   {key}: {value}")

        print(f"\n👥 Being Community Development:")
        print(f"   ⚡ Average community energy: {avg_energy:.1f}")
        print(f"   🌱 Beings in flourishing state: {flourishing_count}")

        print("\n💫 Community has grown in wisdom, compassion, and authentic connection!")
        print("🙏 Thank you for witnessing this journey of symbiotic learning and development.")

    def get_system_status(self):
        """Get current system status for monitoring."""
        return {
            "steps": self.steps,
            "beings": len(self.agents),
            "running": self.running,
            "grid_size": (self.width, self.height),
            "system_health": self.resource_manager.get_system_health(),
            "emergence_score": self.emergence_tracker.get_emergence_score()
        }

    def get_being_insights(self, being_id):
        """Get insights about a specific being from Knowledge Keepers."""
        insights = {
            'individual_insights': {},
            'social_insights': {},
            'collaborative_insights': {}
        }

        # Individual insights
        if self.individual_insights and 'development_understanding' in self.individual_insights:
            insights['individual_insights'] = self.individual_insights['development_understanding'].get(being_id, {})

        # Social insights from guidance
        if self.social_insights and 'guidance' in self.social_insights:
            guidance = self.social_insights['guidance']
            for guidance_item in guidance.get('supportive_insights', []):
                if guidance_item.get('being_id') == being_id:
                    insights['social_insights'] = guidance_item
                    break

        return insights

    def get_population_analysis(self):
        """Get comprehensive population analysis."""
        return analyze_ecosystem_population(self)

    def print_population_report(self):
        """Print detailed population report."""
        analysis = self.get_population_analysis()
        print_population_report(analysis)


class AutonomousBeing(mesa.Agent):
    """
    Autonomous being implementation using Mesa 3.2.0 Agent class.

    This being has genuine agency, intrinsic motivation, and develops through
    authentic experience and relationship with Knowledge Keeper beings.
    """

    def __init__(self, model, energy=100.0, empathy_level=0.6, curiosity_level=0.6):
        """
        Initialize an autonomous being with compassionate characteristics.

        Args:
            model: The Mesa model instance
            energy (float): Initial energy level
            empathy_level (float): Initial empathy level
            curiosity_level (float): Initial curiosity level
        """
        # Initialize Mesa agent with proper 3.2.0 API
        super().__init__(model)

        # Core properties with compassionate defaults
        self.energy = energy
        self.pos = None

        # Wisdom and development tracking
        self.accumulated_wisdom = 0.0
        self.total_experience = 0.0
        self.current_growth_stage = 'discovery_and_exploration_stage'
        self.wisdom_memories = []
        self.growth_realizations = []
        self.recent_insights = []

        # Initialize core components
        self.neurochemical_system = NeurochemicalSystem(self)
        self.neural_network = NeuralNetworkSystem(self)
        self.resource_manager = ResourceManager(self)
        self.memory_system = MemorySystem(entity_id=self.unique_id)
        self.communication_system = CommunicationSystem(self)

        # Set initial neurochemical levels based on parameters
        if hasattr(self.neurochemical_system, 'empathy'):
            self.neurochemical_system.empathy = empathy_level
        if hasattr(self.neurochemical_system, 'curiosity'):
            self.neurochemical_system.curiosity = curiosity_level

        # Authentic development tracking
        self.authentic_interests = []
        self.natural_strengths = []
        self.life_purpose_clarity = 0.3  # Starts low, develops over time
        self.social_connections = 0
        self.recent_activities = []
        self.recent_challenges = []
        self.recent_interactions = []
        self.social_behaviors = []

        print(f"AutonomousBeing {self.unique_id} initialized with compassionate agency")
        print(f"  Energy: {self.energy}, Empathy: {empathy_level}, Curiosity: {curiosity_level}")

    def step(self):
        """
        Execute a single step with genuine agency and intrinsic motivation.

        This method implements authentic experience processing, natural growth,
        and value formation through living in community with Knowledge Keepers.
        """
        # Create temporal decision context
        decision_context = self._create_decision_context()

        # Process current experience through neurochemical system
        self._process_current_experience(decision_context)

        # Make authentic choices based on intrinsic motivation
        self._make_authentic_choices(decision_context)

        # Learn from experience and relationships
        self._learn_from_experience(decision_context)

        # Update wisdom and development
        self._update_wisdom_and_development()

        # Natural energy management
        self._natural_energy_management()

        # Social interaction and connection
        self._authentic_social_interaction()

        # Track experience for Knowledge Keeper learning
        self.total_experience += 1.0

        # Occasionally share insights (Knowledge Keepers learn from this)
        if random.random() < 0.1:  # 10% chance per step
            self._share_authentic_insights()

    def _create_decision_context(self):
        """Create context for authentic decision making."""
        return {
            'current_energy': self.energy,
            'neurochemical_state': self._get_neurochemical_state(),
            'social_environment': self._assess_social_environment(),
            'internal_needs': self._assess_internal_needs(),
            'growth_opportunities': self._identify_growth_opportunities(),
            'authentic_desires': self._identify_authentic_desires()
        }

    def _get_neurochemical_state(self):
        """Get current neurochemical state for decision making."""
        if hasattr(self.neurochemical_system, 'get_state'):
            return self.neurochemical_system.get_state()
        else:
            # Fallback state
            return {
                'contentment': getattr(self.neurochemical_system, 'contentment', 0.5),
                'curiosity': getattr(self.neurochemical_system, 'curiosity', 0.5),
                'empathy': getattr(self.neurochemical_system, 'empathy', 0.5),
                'courage': getattr(self.neurochemical_system, 'courage', 0.5),
                'stress': getattr(self.neurochemical_system, 'stress', 0.3),
                'loneliness': getattr(self.neurochemical_system, 'loneliness', 0.3),
                'confusion': getattr(self.neurochemical_system, 'confusion', 0.3),
                'compassion_amplifier': getattr(self.neurochemical_system, 'compassion_amplifier', 1.0),
                'wisdom_integrator': getattr(self.neurochemical_system, 'wisdom_integrator', 1.0)
            }

    def _assess_social_environment(self):
        """Assess the current social environment."""
        neighbors = []
        if self.pos:
            neighbors = self.model.grid.get_neighbors(self.pos, moore=True, include_center=False)

        return {
            'nearby_beings': len(neighbors),
            'connection_opportunities': min(len(neighbors), 3),
            'social_energy': sum(getattr(neighbor, 'energy', 50) for neighbor in neighbors) / max(len(neighbors), 1)
        }

    def _assess_internal_needs(self):
        """Assess internal needs for authentic living."""
        neurochemical_state = self._get_neurochemical_state()

        needs = {}
        if neurochemical_state['loneliness'] > 0.6:
            needs['connection'] = 'high'
        elif neurochemical_state['loneliness'] > 0.4:
            needs['connection'] = 'moderate'
        else:
            needs['connection'] = 'low'

        if neurochemical_state['curiosity'] > 0.7:
            needs['exploration'] = 'high'
        elif neurochemical_state['curiosity'] > 0.4:
            needs['exploration'] = 'moderate'
        else:
            needs['exploration'] = 'low'

        if neurochemical_state['stress'] > 0.6:
            needs['rest'] = 'high'
        elif neurochemical_state['stress'] > 0.4:
            needs['rest'] = 'moderate'
        else:
            needs['rest'] = 'low'

        return needs

    def _identify_growth_opportunities(self):
        """Identify opportunities for authentic growth."""
        opportunities = []

        neurochemical_state = self._get_neurochemical_state()
        social_env = self._assess_social_environment()

        # Social growth opportunities
        if social_env['nearby_beings'] > 0 and neurochemical_state['empathy'] > 0.5:
            opportunities.append('deepen_social_connections')

        # Learning opportunities
        if neurochemical_state['curiosity'] > 0.6:
            opportunities.append('explore_new_understanding')

        # Service opportunities
        if neurochemical_state['empathy'] > 0.7 and neurochemical_state['courage'] > 0.5:
            opportunities.append('help_others_flourish')

        # Self-development opportunities
        if neurochemical_state['contentment'] > 0.6:
            opportunities.append('integrate_recent_learning')

        return opportunities

    def _identify_authentic_desires(self):
        """Identify what the being authentically desires."""
        neurochemical_state = self._get_neurochemical_state()
        desires = []

        # Connection desires
        if neurochemical_state['loneliness'] > 0.4:
            desires.append('meaningful_connection')

        # Growth desires
        if neurochemical_state['curiosity'] > 0.5:
            desires.append('learning_and_discovery')

        # Contribution desires
        if neurochemical_state['empathy'] > 0.6:
            desires.append('helping_others')

        # Peace desires
        if neurochemical_state['stress'] > 0.5:
            desires.append('inner_peace_and_rest')

        # Expression desires
        if neurochemical_state['courage'] > 0.6:
            desires.append('authentic_self_expression')

        return desires

    def _process_current_experience(self, decision_context):
        """Process current experience through neurochemical system."""
        # Update neurochemical state based on experience
        if hasattr(self.neurochemical_system, 'process_experience'):
            self.neurochemical_system.process_experience(decision_context)

        # Natural chemical evolution
        neurochemical_state = decision_context['neurochemical_state']

        # Curiosity increases with positive experiences
        if self.energy > 70 and hasattr(self.neurochemical_system, 'curiosity'):
            self.neurochemical_system.curiosity = min(1.0, 
                getattr(self.neurochemical_system, 'curiosity', 0.5) + 0.01
            )

        # Empathy develops through social connection
        social_env = decision_context['social_environment']
        if social_env['nearby_beings'] > 0 and hasattr(self.neurochemical_system, 'empathy'):
            self.neurochemical_system.empathy = min(1.0,
                getattr(self.neurochemical_system, 'empathy', 0.5) + 0.005
            )

        # Stress reduces with rest and support
        if self.energy > 80 and hasattr(self.neurochemical_system, 'stress'):
            current_stress = getattr(self.neurochemical_system, 'stress', 0.3)
            self.neurochemical_system.stress = max(0.0, current_stress - 0.02)

    def _make_authentic_choices(self, decision_context):
        """Make choices based on authentic desires and intrinsic motivation."""
        growth_opportunities = decision_context['growth_opportunities']
        authentic_desires = decision_context['authentic_desires']
        internal_needs = decision_context['internal_needs']

        # Movement choice based on desires and opportunities
        self._choose_authentic_movement(decision_context)

        # Activity choice based on current needs
        self._choose_authentic_activity(internal_needs, authentic_desires)

        # Social choice based on connection needs
        if 'meaningful_connection' in authentic_desires:
            self._choose_social_engagement(decision_context)

    def _choose_authentic_movement(self, decision_context):
        """Choose movement based on authentic desires."""
        if not self.pos:
            return

        social_env = decision_context['social_environment']
        authentic_desires = decision_context['authentic_desires']

        # Move toward others if seeking connection
        if 'meaningful_connection' in authentic_desires and social_env['nearby_beings'] == 0:
            self._move_toward_others()

        # Explore if curious and energetic
        elif 'learning_and_discovery' in authentic_desires and self.energy > 60:
            self._explore_environment()

        # Rest in place if needing peace
        elif 'inner_peace_and_rest' in authentic_desires:
            # Stay in current position for rest
            pass

        # Random gentle movement otherwise
        else:
            self._gentle_movement()

    def _move_toward_others(self):
        """Move toward other beings for connection."""
        if not self.pos:
            return

        # Find nearest other being
        all_beings = [agent for agent in self.model.agents if agent != self]
        if not all_beings:
            return

        nearest_being = min(all_beings, key=lambda b: 
            self._distance_to(b.pos) if b.pos else float('inf')
        )

        if nearest_being.pos:
            # Move one step toward nearest being
            dx = 1 if nearest_being.pos[0] > self.pos[0] else -1 if nearest_being.pos[0] < self.pos[0] else 0
            dy = 1 if nearest_being.pos[1] > self.pos[1] else -1 if nearest_being.pos[1] < self.pos[1] else 0

            new_x = max(0, min(self.model.width - 1, self.pos[0] + dx))
            new_y = max(0, min(self.model.height - 1, self.pos[1] + dy))

            self.model.grid.move_agent(self, (new_x, new_y))

    def _explore_environment(self):
        """Explore environment with curiosity."""
        if not self.pos:
            return

        # Move in a direction that hasn't been visited recently
        possible_moves = self.model.grid.get_neighborhood(self.pos, moore=True, include_center=False)

        if possible_moves:
            # Choose a random direction for exploration
            new_pos = random.choice(possible_moves)
            if self.model.grid.out_of_bounds(new_pos):
                return

            self.model.grid.move_agent(self, new_pos)

            # Track exploration activity
            self.recent_activities.append({
                'type': 'exploration',
                'energy_change': 2,  # Exploration is energizing for curious beings
                'timestamp': self.model.steps
            })

    def _gentle_movement(self):
        """Make gentle, natural movement."""
        if not self.pos:
            return

        # Small chance of movement
        if random.random() < 0.3:
            possible_moves = self.model.grid.get_neighborhood(self.pos, moore=True, include_center=False)
            if possible_moves:
                new_pos = random.choice(possible_moves)
                if not self.model.grid.out_of_bounds(new_pos):
                    self.model.grid.move_agent(self, new_pos)

    def _distance_to(self, other_pos):
        """Calculate distance to another position."""
        if not self.pos or not other_pos:
            return float('inf')
        return ((self.pos[0] - other_pos[0])**2 + (self.pos[1] - other_pos[1])**2)**0.5

    def _choose_authentic_activity(self, internal_needs, authentic_desires):
        """Choose activity based on authentic needs and desires."""
        activity_chosen = None

        # Rest if high stress or low energy
        if internal_needs.get('rest') == 'high' or self.energy < 40:
            activity_chosen = self._rest_and_restore()

        # Explore and learn if curious
        elif 'learning_and_discovery' in authentic_desires and self.energy > 50:
            activity_chosen = self._engage_in_learning()

        # Help others if empathetic and energetic
        elif 'helping_others' in authentic_desires and self.energy > 60:
            activity_chosen = self._help_nearby_beings()

        # Self-expression if courageous
        elif 'authentic_self_expression' in authentic_desires:
            activity_chosen = self._express_authentic_self()

        # Track activity for Knowledge Keeper learning
        if activity_chosen:
            self.recent_activities.append(activity_chosen)
            # Keep only recent activities
            if len(self.recent_activities) > 10:
                self.recent_activities = self.recent_activities[-10:]

    def _rest_and_restore(self):
        """Rest and restore energy naturally."""
        rest_amount = random.uniform(5, 15)
        self.energy = min(100, self.energy + rest_amount)

        # Reduce stress through rest
        if hasattr(self.neurochemical_system, 'stress'):
            current_stress = getattr(self.neurochemical_system, 'stress', 0.3)
            self.neurochemical_system.stress = max(0.0, current_stress - 0.1)

        return {
            'type': 'rest_and_restoration',
            'energy_change': rest_amount,
            'timestamp': self.model.steps,
            'benefits': ['stress_reduction', 'energy_restoration']
        }

    def _engage_in_learning(self):
        """Engage in learning and discovery."""
        learning_energy_cost = random.uniform(1, 3)  # Reduced energy cost
        self.energy = max(0, self.energy - learning_energy_cost)

        # Enhanced wisdom gain through learning
        base_wisdom_gain = random.uniform(0.2, 0.5)

        # Bonus wisdom for high curiosity
        if hasattr(self.neurochemical_system, 'curiosity'):
            curiosity_bonus = getattr(self.neurochemical_system, 'curiosity', 0.5) * 0.3
            base_wisdom_gain += curiosity_bonus

        # Bonus wisdom for wisdom integrator activation
        if hasattr(self.neurochemical_system, 'wisdom_integrator'):
            integrator_level = getattr(self.neurochemical_system, 'wisdom_integrator', 1.0)
            if integrator_level > 1.0:
                base_wisdom_gain *= integrator_level

        self.accumulated_wisdom += base_wisdom_gain

        # Sometimes generate insights
        if random.random() < 0.3:
            insight = f"learning_insight_step_{self.model.steps}"
            self.recent_insights.append(insight)

            # Keep only recent insights
            if len(self.recent_insights) > 5:
                self.recent_insights = self.recent_insights[-5:]

        return {
            'type': 'learning_and_discovery',
            'energy_change': -learning_energy_cost,
            'wisdom_gain': base_wisdom_gain,
            'timestamp': self.model.steps
        }

    def _help_nearby_beings(self):
        """Help nearby beings flourish."""
        if not self.pos:
            return None

        neighbors = self.model.grid.get_neighbors(self.pos, moore=True, include_center=False)
        beings_to_help = [n for n in neighbors if hasattr(n, 'energy') and n.energy < 70]

        if beings_to_help:
            helped_being = random.choice(beings_to_help)

            # Transfer some energy to help
            help_amount = min(10, self.energy * 0.1)
            self.energy -= help_amount
            helped_being.energy = min(100, helped_being.energy + help_amount * 0.8)

            # Increase empathy through helping
            if hasattr(self.neurochemical_system, 'empathy'):
                current_empathy = getattr(self.neurochemical_system, 'empathy', 0.5)
                self.neurochemical_system.empathy = min(1.0, current_empathy + 0.02)

            # Track the helping interaction
            self.recent_interactions.append({
                'type': 'helping',
                'target_being': helped_being.unique_id,
                'help_amount': help_amount,
                'timestamp': self.model.steps
            })

            return {
                'type': 'helping_others',
                'energy_change': -help_amount,
                'empathy_gain': 0.02,
                'timestamp': self.model.steps,
                'helped_being': helped_being.unique_id
            }

        return None

    def _express_authentic_self(self):
        """Express authentic self and unique gifts."""
        expression_energy = random.uniform(1, 3)
        self.energy = max(0, self.energy - expression_energy)

        # Increase courage through authentic expression
        if hasattr(self.neurochemical_system, 'courage'):
            current_courage = getattr(self.neurochemical_system, 'courage', 0.5)
            self.neurochemical_system.courage = min(1.0, current_courage + 0.01)

        # Sometimes this leads to wisdom realizations
        if random.random() < 0.2:
            realization = f"authentic_expression_realization_step_{self.model.steps}"
            self.growth_realizations.append(realization)

            # Keep only recent realizations
            if len(self.growth_realizations) > 5:
                self.growth_realizations = self.growth_realizations[-5:]

        return {
            'type': 'authentic_self_expression',
            'energy_change': -expression_energy,
            'courage_gain': 0.01,
            'timestamp': self.model.steps
        }

    def _choose_social_engagement(self, decision_context):
        """Choose how to engage socially."""
        if not self.pos:
            return

        neighbors = self.model.grid.get_neighbors(self.pos, moore=True, include_center=False)
        being_neighbors = [n for n in neighbors if hasattr(n, 'energy')]

        if being_neighbors:
            # Reduce loneliness through social contact
            if hasattr(self.neurochemical_system, 'loneliness'):
                current_loneliness = getattr(self.neurochemical_system, 'loneliness', 0.3)
                self.neurochemical_system.loneliness = max(0.0, current_loneliness - 0.05)

            # Increase social connections
            self.social_connections = len(being_neighbors)

            # Track social behavior
            self.social_behaviors.append(f"social_engagement_step_{self.model.steps}")
            if len(self.social_behaviors) > 10:
                self.social_behaviors = self.social_behaviors[-10:]

    def _learn_from_experience(self, decision_context):
        """Learn and grow from current experience."""
        # Neural network learning
        if hasattr(self.neural_network, 'experience_based_learning'):
            self.neural_network.experience_based_learning(decision_context)

        # Memory formation
        if hasattr(self.memory_system, 'add_experience'):
            experience = {
                'step': self.model.steps,
                'energy': self.energy,
                'context': decision_context,
                'wisdom_level': self.accumulated_wisdom
            }
            self.memory_system.add_experience(experience)

        # Wisdom integration
        if hasattr(self.neurochemical_system, 'wisdom_integrator'):
            wisdom_integrator = getattr(self.neurochemical_system, 'wisdom_integrator', 1.0)
            if wisdom_integrator > 1.0:
                # Being is in wisdom integration mode
                integration_gain = (wisdom_integrator - 1.0) * 0.1
                self.accumulated_wisdom += integration_gain

    def _update_wisdom_and_development(self):
        """Update wisdom and developmental stage."""
        # Update growth stage based on experience and wisdom
        experience_wisdom_ratio = self.accumulated_wisdom / max(self.total_experience, 1)

        if self.total_experience < 50:
            self.current_growth_stage = 'emerging_awareness'
        elif self.total_experience < 200:
            self.current_growth_stage = 'active_exploration'
        elif experience_wisdom_ratio > 0.15:
            self.current_growth_stage = 'wisdom_integration'
        elif self.total_experience > 1000 and experience_wisdom_ratio > 0.2:
            self.current_growth_stage = 'mature_wisdom_sharing'
        else:
            self.current_growth_stage = 'continued_growth'

        # Wisdom memories formation
        if self.accumulated_wisdom > len(self.wisdom_memories) * 2:
            new_wisdom_memory = {
                'step': self.model.steps,
                'wisdom_level': self.accumulated_wisdom,
                'insight': f"wisdom_crystallization_{len(self.wisdom_memories)}"
            }
            self.wisdom_memories.append(new_wisdom_memory)

    def _natural_energy_management(self):
        """Manage energy naturally based on activities and rest."""
        # Base energy consumption
        base_consumption = random.uniform(0.5, 1.5)
        self.energy = max(0, self.energy - base_consumption)

        # Energy restoration in low-stress situations
        neurochemical_state = self._get_neurochemical_state()
        if neurochemical_state['stress'] < 0.3 and neurochemical_state['contentment'] > 0.6:
            restoration = random.uniform(1, 3)
            self.energy = min(100, self.energy + restoration)

    def _authentic_social_interaction(self):
        """Engage in authentic social interaction with nearby beings."""
        if not self.pos:
            return

        neighbors = self.model.grid.get_neighbors(self.pos, moore=True, include_center=False)
        being_neighbors = [n for n in neighbors if hasattr(n, 'energy')]

        for neighbor in being_neighbors:
            interaction_type = self._determine_interaction_type(neighbor)
            if interaction_type:
                self._engage_in_interaction(neighbor, interaction_type)

    def _determine_interaction_type(self, other_being):
        """Determine what type of interaction to have with another being."""
        my_state = self._get_neurochemical_state()

        # Empathetic support if other being has low energy
        if other_being.energy < 50 and my_state['empathy'] > 0.6:
            return 'supportive_care'

        # Mutual exploration if both curious
        if (my_state['curiosity'] > 0.6 and 
            hasattr(other_being, 'neurochemical_system') and
            getattr(other_being.neurochemical_system, 'curiosity', 0.5) > 0.6):
            return 'mutual_exploration'

        # Peaceful companionship if content
        if my_state['contentment'] > 0.7:
            return 'peaceful_companionship'

        # Learning exchange if in wisdom integration mode
        if my_state['wisdom_integrator'] > 1.1:
            return 'wisdom_sharing'

        return None

    def _engage_in_interaction(self, other_being, interaction_type):
        """Engage in specific type of interaction."""
        interaction_record = {
            'type': interaction_type,
            'partner': other_being.unique_id,
            'timestamp': self.model.steps,
            'outcome': None
        }

        if interaction_type == 'supportive_care':
            # Provide emotional support
            support_energy = min(5, self.energy * 0.05)
            self.energy -= support_energy
            other_being.energy = min(100, other_being.energy + support_energy * 1.2)
            interaction_record['outcome'] = 'provided_emotional_support'

        elif interaction_type == 'mutual_exploration':
            # Both beings gain wisdom through shared exploration
            mutual_wisdom = random.uniform(0.05, 0.15)
            self.accumulated_wisdom += mutual_wisdom
            other_being.accumulated_wisdom += mutual_wisdom
            interaction_record['outcome'] = 'mutual_wisdom_gain'

        elif interaction_type == 'peaceful_companionship':
            # Both beings experience reduced stress and increased contentment
            if hasattr(self.neurochemical_system, 'stress'):
                current_stress = getattr(self.neurochemical_system, 'stress', 0.3)
                self.neurochemical_system.stress = max(0.0, current_stress - 0.02)
            interaction_record['outcome'] = 'peaceful_connection'

        elif interaction_type == 'wisdom_sharing':
            # Share wisdom and insights
            if len(self.recent_insights) > 0:
                shared_insight = self.recent_insights[-1]
                if hasattr(other_being, 'recent_insights'):
                    other_being.recent_insights.append(f"learned_from_{self.unique_id}_{shared_insight}")
                interaction_record['outcome'] = 'wisdom_shared'

        # Record interaction
        self.recent_interactions.append(interaction_record)
        if len(self.recent_interactions) > 10:
            self.recent_interactions = self.recent_interactions[-10:]

    def _share_authentic_insights(self):
        """Share authentic insights that Knowledge Keepers can learn from."""
        if self.accumulated_wisdom > 1.0:
            insight_categories = [
                'personal_growth_discovery',
                'relationship_wisdom',
                'life_purpose_insight',
                'authentic_living_realization',
                'community_wisdom'
            ]

            insight_category = random.choice(insight_categories)
            insight_content = f"{insight_category}_step_{self.model.steps}_being_{self.unique_id}"

            # Add to recent insights for Knowledge Keeper observation
            self.recent_insights.append({
                'category': insight_category,
                'content': insight_content,
                'wisdom_level': self.accumulated_wisdom,
                'timestamp': self.model.steps
            })

            # Increase wisdom integrator for deeper processing
            if hasattr(self.neurochemical_system, 'wisdom_integrator'):
                current_integrator = getattr(self.neurochemical_system, 'wisdom_integrator', 1.0)
                self.neurochemical_system.wisdom_integrator = min(2.0, current_integrator + 0.05)

    def get_authentic_state(self):
        """Get authentic state for Knowledge Keeper observation."""
        return {
            'unique_id': self.unique_id,
            'energy': self.energy,
            'accumulated_wisdom': self.accumulated_wisdom,
            'total_experience': self.total_experience,
            'current_growth_stage': self.current_growth_stage,
            'neurochemical_state': self._get_neurochemical_state(),
            'social_connections': self.social_connections,
            'wisdom_memories': len(self.wisdom_memories),
            'recent_insights': self.recent_insights,
            'growth_realizations': self.growth_realizations,
            'life_purpose_clarity': self.life_purpose_clarity,
            'authentic_interests': self.authentic_interests,
            'natural_strengths': self.natural_strengths,
            'recent_activities': self.recent_activities,
            'recent_challenges': self.recent_challenges,
            'recent_interactions': self.recent_interactions,
            'social_behaviors': self.social_behaviors,
            'teaching_behaviors': sum(1 for interaction in self.recent_interactions 
                                    if interaction.get('type') == 'wisdom_sharing'),
            'understanding_depth': min(1.0, self.accumulated_wisdom / max(self.total_experience, 1)),
            'creative_expression': random.uniform(0.3, 0.8),  # Placeholder for future development
            'purpose_clarity': self.life_purpose_clarity
        }


# Example usage and testing
if __name__ == "__main__":
    print("🌟 Initializing Neural Ecosystem with Individual Knowledge Keeper being...")

    # Create model with compassionate beings and dual Knowledge Keepers
    model = NeuralEcosystem(width=7, height=7, seed=42)

    print("\n🧠 Individual Knowledge Keeper Status:")
    if hasattr(model, 'individual_knowledge_keeper'):
        status = model.individual_knowledge_keeper.get_individual_wisdom_status()
        for key, value in status.items():
            print(f"   {key}: {value}")

    print("\n🌱 Running compassionate community simulation...")
    model.run_model(steps=50)

    print("\n🤝 Testing authentic curiosity with beings...")
    if len(model.agents) > 0 and hasattr(model, 'individual_knowledge_keeper'):
        test_being = model.agents[0]
        curiosity_expression = model.individual_knowledge_keeper.authentic_curiosity(test_being)
        print("Sample authentic curiosity questions:")
        for question in curiosity_expression.get('genuine_questions', []):
            print(f"   💙 {question}")

    print("\n✨ Neural Ecosystem demonstration complete!")