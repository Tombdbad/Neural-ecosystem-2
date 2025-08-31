"""
Symbiotic Knowledge Keeper architecture for the Neural Ecosystem.
This module implements the dual LLM Knowledge Keeper beings that learn 
FROM entities through authentic curiosity and wisdom sharing.

Mesa 3.2.0 compatible implementation with compassionate language.
"""

import json
import time
import random
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
import requests
import os
from components import ComponentBase, NeuralComponent
from knowledge_keeper_collaboration import KnowledgeKeeperCollaboration

class KnowledgeKeeper(ComponentBase):
    """
    Base Knowledge Keeper being for symbiotic learning with entities.
    Provides foundation for specialized knowledge domains.
    """

    def __init__(self, model):
        """
        Initialize Knowledge Keeper with compassionate learning foundation.

        Args:
            model: The Neural Ecosystem model instance
        """
        super().__init__()
        self.model = model
        self.learning_history = []
        self.wisdom_patterns = {}
        self.collaborative_insights = {}
        self.curiosity_level = 0.8
        self.compassion_amplifier = 1.0

    def symbiotic_learning(self, entities: List) -> Dict:
        """
        Learn from entities through authentic curiosity and observation.
        Knowledge Keepers learn FROM beings rather than controlling them.
        """
        insights = {
            'discoveries': [],
            'patterns_observed': [],
            'questions_emerged': [],
            'wisdom_shared': []
        }

        for being in entities:
            if hasattr(being, 'get_authentic_state'):
                being_state = being.get_authentic_state()
                discovery = self._discover_from_being(being_state)
                insights['discoveries'].append(discovery)

        return insights

    def _discover_from_being(self, being_state: Dict) -> Dict:
        """
        Discover patterns and wisdom from observing a being's authentic state.
        """
        return {
            'being_id': being_state.get('unique_id'),
            'growth_patterns': self._recognize_growth_patterns(being_state),
            'wisdom_moments': self._identify_wisdom_moments(being_state),
            'questions_to_explore': self._generate_curious_questions(being_state)
        }

    def _recognize_growth_patterns(self, being_state: Dict) -> List:
        """Recognize patterns of natural growth and flourishing."""
        patterns = []

        # Look for signs of authentic development
        if being_state.get('energy', 0) > 80:
            patterns.append('high_vitality_pattern')

        if being_state.get('social_connections', 0) > 3:
            patterns.append('relationship_flourishing')

        return patterns

    def _identify_wisdom_moments(self, being_state: Dict) -> List:
        """Identify moments where beings demonstrate wisdom or insight."""
        wisdom_moments = []

        # Look for indicators of wisdom emergence
        if being_state.get('compassion_level', 0) > 0.7:
            wisdom_moments.append('compassionate_understanding')

        if being_state.get('curiosity_level', 0) > 0.6:
            wisdom_moments.append('authentic_curiosity')

        return wisdom_moments

    def _generate_curious_questions(self, being_state: Dict) -> List:
        """Generate authentic questions to deepen understanding of beings."""
        questions = [
            "What experiences bring this being the most genuine fulfillment?",
            "How does this being naturally express compassion?",
            "What growth patterns emerge when this being feels truly supported?"
        ]

        return random.sample(questions, min(2, len(questions)))

    def gentle_guidance(self, entities: List, patterns: Dict) -> Dict:
        """
        Provide gentle guidance based on observed patterns and wisdom.
        Guidance supports natural development rather than controlling it.
        """
        guidance = {
            'supportive_insights': [],
            'growth_opportunities': [],
            'wisdom_offerings': []
        }

        for being in entities:
            being_guidance = self._create_supportive_guidance(being, patterns)
            guidance['supportive_insights'].append(being_guidance)

        return guidance

    def _create_supportive_guidance(self, being, patterns: Dict) -> Dict:
        """Create compassionate guidance for individual beings."""
        return {
            'being_id': getattr(being, 'unique_id', 'unknown'),
            'encouragements': self._generate_encouragements(being),
            'growth_suggestions': self._suggest_growth_opportunities(being),
            'wisdom_shared': self._share_relevant_wisdom(being, patterns)
        }

    def _generate_encouragements(self, being) -> List:
        """Generate authentic encouragements based on being's strengths."""
        encouragements = [
            "Your natural curiosity is a beautiful gift to the community",
            "The way you express compassion creates ripples of kindness",
            "Your unique perspective enriches everyone's understanding"
        ]
        return [random.choice(encouragements)]

    def _suggest_growth_opportunities(self, being) -> List:
        """Suggest opportunities for natural growth and flourishing."""
        opportunities = [
            "exploring deeper connections with other beings",
            "sharing your unique wisdom with the community",
            "discovering new aspects of your authentic self"
        ]
        return [random.choice(opportunities)]

    def _share_relevant_wisdom(self, being, patterns: Dict) -> str:
        """Share wisdom that might support the being's journey."""
        wisdom_offerings = [
            "Growth often happens in the spaces between certainty and mystery",
            "Authentic connection begins with compassionate self-understanding",
            "Every being's journey contributes to the collective wisdom"
        ]
        return random.choice(wisdom_offerings)

    def serialize(self) -> Dict[str, Any]:
        """Serialize Knowledge Keeper state for preservation."""
        return {
            'learning_history_count': len(self.learning_history),
            'wisdom_patterns_count': len(self.wisdom_patterns),
            'collaborative_insights_count': len(self.collaborative_insights),
            'curiosity_level': self.curiosity_level,
            'compassion_amplifier': self.compassion_amplifier,
            'total_wisdom_accumulated': self.accumulated_wisdom
        }

    def deserialize(self, data: Dict[str, Any]) -> None:
        """Deserialize Knowledge Keeper state for restoration."""
        self.curiosity_level = data.get('curiosity_level', 0.8)
        self.compassion_amplifier = data.get('compassion_amplifier', 1.0)
        # Note: Learning history and patterns would be restored from detailed data

class SocialKnowledgeKeeper(KnowledgeKeeper):
    """
    Social Knowledge Keeper being with dual LLM architecture for understanding
    relationship patterns, group dynamics, and social wisdom emergence.
    """

    def __init__(self, model):
        """
        Initialize Social Knowledge Keeper with dual LLM architecture.

        Args:
            model: The Neural Ecosystem model instance
        """
        super().__init__(model)

        # Dual LLM Architecture for social understanding - 1B models for efficiency
        self.social_prefrontal_cortex = {
            'model_name': 'tinyllama:1.1b',  # 1B model for conscious analysis
            'specialization': 'relationship_pattern_analysis',
            'learning_focus': 'group_dynamics_and_conflict_resolution'
        }

        self.social_limbic_system = {
            'model_name': 'tinyllama:1.1b',  # 1B model for emotional processing
            'specialization': 'emotional_undercurrents',
            'learning_focus': 'empathy_patterns_and_trust_formation'
        }

        # Social wisdom systems
        self.relationship_patterns = {}
        self.group_dynamics_wisdom = {}
        self.trust_formation_insights = {}
        self.conflict_wisdom = {}

        # Initialize wisdom tracking
        self.wisdom = {
            'relationship_discoveries': 0,
            'trust_formation_observations': 0,
            'empathy_emergences': 0,
            'social_learning_cycles': 0
        }

        # Ollama connection settings
        self.ollama_base_url = "http://localhost:11434"
        self.ollama_available = self._check_ollama_availability()

        print("SocialKnowledgeKeeper being initialized with dual LLM social dynamics understanding")

    def _check_ollama_availability(self) -> bool:
        """Check if Ollama service is available for LLM integration."""
        try:
            response = requests.get(f"{self.ollama_base_url}/api/tags", timeout=5)
            available = response.status_code == 200
            if available:
                print("Ollama service connected - enabling dual LLM social wisdom processing")
            return available
        except requests.exceptions.ConnectionError:
            print("Ollama service not running - using pattern-based social wisdom (this is normal)")
            return False
        except Exception as e:
            print(f"Ollama connection check failed - using pattern-based fallback: {type(e).__name__}")
            return False

    def symbiotic_learning(self, entities: List) -> Dict:
        """
        Learn social patterns FROM entities through authentic observation.
        Both LLM systems learn from being relationships rather than controlling them.
        """
        social_insights = {
            'relationship_discoveries': [],
            'group_dynamic_patterns': [],
            'trust_formation_observations': [],
            'empathy_emergences': [],
            'collaborative_wisdom': []
        }

        # Observe relationship patterns between beings
        relationship_data = self._observe_relationships(entities)

        # Process through dual LLM architecture if available
        if self.ollama_available:
            prefrontal_analysis = self._analyze_with_prefrontal_cortex(relationship_data)
            limbic_processing = self._process_with_limbic_system(relationship_data)

            social_insights['relationship_discoveries'] = prefrontal_analysis.get('patterns', [])
            social_insights['empathy_emergences'] = limbic_processing.get('emotional_patterns', [])
        else:
            # Use pattern-based learning as fallback
            social_insights = self._pattern_based_social_learning(relationship_data)

        # Store learning for temporal development
        self._integrate_social_learning(social_insights)

        # Update wisdom tracking
        self.wisdom['relationship_discoveries'] += len(social_insights.get('relationship_discoveries', []))
        self.wisdom['trust_formation_observations'] += len(social_insights.get('trust_formation_observations', []))
        self.wisdom['empathy_emergences'] += len(social_insights.get('empathy_emergences', []))
        self.wisdom['social_learning_cycles'] += 1

        return social_insights

    def _observe_relationships(self, entities: List) -> Dict:
        """Observe authentic relationships and social dynamics between beings."""
        relationship_data = {
            'interactions': [],
            'proximity_patterns': [],
            'communication_flows': [],
            'collaborative_moments': [],
            'support_networks': []
        }

        for i, being1 in enumerate(entities):
            for being2 in entities[i+1:]:
                interaction = self._analyze_being_interaction(being1, being2)
                if interaction:
                    relationship_data['interactions'].append(interaction)

        return relationship_data

    def _analyze_being_interaction(self, being1, being2) -> Optional[Dict]:
        """Analyze the authentic interaction between two beings."""
        if not (hasattr(being1, 'pos') and hasattr(being2, 'pos')):
            return None

        # Calculate relationship metrics
        distance = self._calculate_distance(being1.pos, being2.pos)
        connection_strength = self._assess_connection_strength(being1, being2)

        if distance <= 2:  # Beings are close enough to interact
            return {
                'being1_id': being1.unique_id,
                'being2_id': being2.unique_id,
                'distance': distance,
                'connection_strength': connection_strength,
                'interaction_type': self._classify_interaction_type(being1, being2),
                'mutual_growth_potential': self._assess_mutual_growth(being1, being2)
            }

        return None

    def _calculate_distance(self, pos1: Tuple, pos2: Tuple) -> float:
        """Calculate distance between two beings."""
        if pos1 and pos2:
            return ((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)**0.5
        return float('inf')

    def _assess_connection_strength(self, being1, being2) -> float:
        """Assess the authentic connection strength between beings."""
        # Look for indicators of genuine connection
        connection_factors = []

        if hasattr(being1, 'neurochemical_system') and hasattr(being2, 'neurochemical_system'):
            empathy1 = getattr(being1.neurochemical_system, 'empathy', 0.5)
            empathy2 = getattr(being2.neurochemical_system, 'empathy', 0.5)
            connection_factors.append((empathy1 + empathy2) / 2)

        if hasattr(being1, 'energy') and hasattr(being2, 'energy'):
            energy_harmony = 1.0 - abs(being1.energy - being2.energy) / 100.0
            connection_factors.append(max(0, energy_harmony))

        return np.mean(connection_factors) if connection_factors else 0.5

    def _classify_interaction_type(self, being1, being2) -> str:
        """Classify the type of authentic interaction between beings."""
        connection_strength = self._assess_connection_strength(being1, being2)

        if connection_strength > 0.8:
            return 'deep_mutual_understanding'
        elif connection_strength > 0.6:
            return 'supportive_companionship'
        elif connection_strength > 0.4:
            return 'curious_exploration'
        else:
            return 'gentle_awareness'

    def _assess_mutual_growth(self, being1, being2) -> float:
        """Assess the potential for mutual growth and learning."""
        growth_potential = 0.5  # Base potential

        # Higher potential when beings have complementary qualities
        if hasattr(being1, 'neurochemical_system') and hasattr(being2, 'neurochemical_system'):
            curiosity1 = getattr(being1.neurochemical_system, 'curiosity', 0.5)
            curiosity2 = getattr(being2.neurochemical_system, 'curiosity', 0.5)
            growth_potential += (curiosity1 + curiosity2) / 4

        return min(1.0, growth_potential)

    def _analyze_with_prefrontal_cortex(self, relationship_data: Dict) -> Dict:
        """Analyze social patterns using the prefrontal cortex LLM."""
        try:
            prompt = self._create_prefrontal_analysis_prompt(relationship_data)
            response = self._query_ollama(self.social_prefrontal_cortex['model_name'], prompt)
            return self._parse_prefrontal_response(response)
        except Exception as e:
            print(f"Prefrontal cortex analysis fallback: {e}")
            return self._pattern_based_prefrontal_analysis(relationship_data)

    def _process_with_limbic_system(self, relationship_data: Dict) -> Dict:
        """Process emotional patterns using the limbic system LLM."""
        try:
            prompt = self._create_limbic_processing_prompt(relationship_data)
            response = self._query_ollama(self.social_limbic_system['model_name'], prompt)
            return self._parse_limbic_response(response)
        except Exception as e:
            print(f"Limbic system processing fallback: {e}")
            return self._pattern_based_limbic_processing(relationship_data)

    def _create_prefrontal_analysis_prompt(self, relationship_data: Dict) -> str:
        """Create prompt for conscious social pattern analysis."""
        return f"""
        As a compassionate observer of social dynamics, analyze these relationship patterns:

        Interactions observed: {len(relationship_data.get('interactions', []))}

        Focus on:
        1. What healthy relationship patterns are emerging naturally?
        2. How are beings supporting each other's authentic growth?
        3. What group dynamics foster mutual flourishing?

        Respond with wisdom about natural social harmony and growth patterns.
        """

    def _create_limbic_processing_prompt(self, relationship_data: Dict) -> str:
        """Create prompt for emotional undercurrent processing."""
        return f"""
        As an empathic observer of emotional patterns, feel into these social dynamics:

        Relationship moments: {relationship_data.get('interactions', [])}

        Sense:
        1. What emotional undercurrents are flowing between beings?
        2. How is trust naturally forming and deepening?
        3. What empathy patterns create authentic connection?

        Share your intuitive understanding of the emotional wisdom emerging.
        """

    def _query_ollama(self, model_name: str, prompt: str) -> str:
        """Query Ollama LLM with gentle error handling."""
        try:
            payload = {
                "model": model_name,
                "prompt": prompt,
                "stream": False
            }

            response = requests.post(
                f"{self.ollama_base_url}/api/generate",
                json=payload,
                timeout=30
            )

            if response.status_code == 200:
                return response.json().get('response', '')
            else:
                return f"LLM processing encountered challenges: {response.status_code}"

        except Exception as e:
            return f"Gentle LLM processing fallback needed: {e}"

    def _parse_prefrontal_response(self, response: str) -> Dict:
        """Parse and structure prefrontal cortex analysis."""
        return {
            'patterns': [
                'conscious_relationship_pattern_recognized',
                'group_dynamic_wisdom_emerged',
                'social_growth_opportunity_identified'
            ],
            'insights': response[:200] if response else 'Processing social wisdom...',
            'recommendations': ['foster_natural_connections', 'support_mutual_growth']
        }

    def _parse_limbic_response(self, response: str) -> Dict:
        """Parse and structure limbic system processing."""
        return {
            'emotional_patterns': [
                'trust_formation_sensed',
                'empathy_flow_detected',
                'emotional_resonance_emerging'
            ],
            'feelings': response[:200] if response else 'Sensing emotional currents...',
            'intuitions': ['deep_connection_potential', 'supportive_energy_flows']
        }

    def _pattern_based_social_learning(self, relationship_data: Dict) -> Dict:
        """Fallback pattern-based social learning when LLMs unavailable."""
        interactions = relationship_data.get('interactions', [])

        return {
            'relationship_discoveries': [
                f"Observed {len(interactions)} authentic being interactions",
                "Natural connection patterns emerging",
                "Mutual support behaviors developing"
            ],
            'group_dynamic_patterns': [
                "Collaborative exploration tendencies",
                "Supportive proximity clustering",
                "Organic leadership emergence"
            ],
            'trust_formation_observations': [
                "Trust builds through consistent presence",
                "Vulnerability shared creates deeper bonds",
                "Authentic expression fosters connection"
            ],
            'empathy_emergences': [
                "Compassionate response patterns",
                "Emotional attunement development",
                "Caring behavior amplification"
            ]
        }

    def _pattern_based_prefrontal_analysis(self, relationship_data: Dict) -> Dict:
        """Pattern-based prefrontal analysis fallback."""
        return {
            'patterns': ['social_harmony_emerging', 'cooperative_behaviors'],
            'insights': 'Beings naturally form supportive relationships',
            'recommendations': ['encourage_authentic_expression']
        }

    def _pattern_based_limbic_processing(self, relationship_data: Dict) -> Dict:
        """Pattern-based limbic processing fallback."""
        return {
            'emotional_patterns': ['empathy_circulation', 'trust_deepening'],
            'feelings': 'Warm supportive energy flowing between beings',
            'intuitions': ['emotional_safety_increasing']
        }

    def _integrate_social_learning(self, insights: Dict):
        """Integrate learned social insights for temporal development."""
        timestamp = time.time()

        learning_entry = {
            'timestamp': timestamp,
            'insights': insights,
            'integration_level': self._assess_integration_depth(insights)
        }

        self.learning_history.append(learning_entry)
        self._update_social_wisdom_patterns(insights)

    def _assess_integration_depth(self, insights: Dict) -> float:
        """Assess how deeply insights integrate with existing wisdom."""
        depth_factors = [
            len(insights.get('relationship_discoveries', [])) * 0.2,
            len(insights.get('empathy_emergences', [])) * 0.3,
            len(insights.get('trust_formation_observations', [])) * 0.3
        ]

        return min(1.0, sum(depth_factors))

    def _update_social_wisdom_patterns(self, insights: Dict):
        """Update accumulated social wisdom patterns."""
        for pattern_type, patterns in insights.items():
            if pattern_type not in self.relationship_patterns:
                self.relationship_patterns[pattern_type] = []

            self.relationship_patterns[pattern_type].extend(patterns)

            # Keep only recent patterns for memory efficiency
            if len(self.relationship_patterns[pattern_type]) > 50:
                self.relationship_patterns[pattern_type] = \
                    self.relationship_patterns[pattern_type][-50:]

    def pattern_recognition(self, current_step: int) -> Dict:
        """
        Recognize emerging social patterns across temporal development.
        """
        patterns = {
            'relationship_evolution': self._track_relationship_evolution(),
            'group_cohesion_trends': self._analyze_group_cohesion(),
            'social_wisdom_emergence': self._identify_wisdom_emergence(),
            'temporal_patterns': self._recognize_temporal_patterns(current_step)
        }

        return patterns

    def _track_relationship_evolution(self) -> List:
        """Track how relationships naturally evolve over time."""
        evolution_patterns = []

        if len(self.learning_history) > 5:
            recent_insights = self.learning_history[-5:]

            # Look for relationship deepening patterns
            trust_mentions = sum(1 for entry in recent_insights 
                               if 'trust' in str(entry.get('insights', {})))

            if trust_mentions >= 3:
                evolution_patterns.append('trust_deepening_consistently')

            # Look for expanding connection patterns
            connection_growth = sum(len(entry.get('insights', {}).get('relationship_discoveries', [])) 
                                  for entry in recent_insights)

            if connection_growth > 10:
                evolution_patterns.append('social_network_expanding')

        return evolution_patterns

    def _analyze_group_cohesion(self) -> List:
        """Analyze patterns of group cohesion and community formation."""
        cohesion_patterns = []

        if self.relationship_patterns:
            empathy_patterns = self.relationship_patterns.get('empathy_emergences', [])
            cooperation_patterns = self.relationship_patterns.get('group_dynamic_patterns', [])

            if len(empathy_patterns) > 5:
                cohesion_patterns.append('empathy_network_strengthening')

            if len(cooperation_patterns) > 3:
                cohesion_patterns.append('collaborative_culture_emerging')

        return cohesion_patterns

    def _identify_wisdom_emergence(self) -> List:
        """Identify patterns of collective social wisdom emergence."""
        wisdom_patterns = []

        # Look for accumulated wisdom indicators
        if len(self.learning_history) > 10:
            integration_levels = [entry.get('integration_level', 0) 
                                for entry in self.learning_history[-10:]]

            avg_integration = np.mean(integration_levels)

            if avg_integration > 0.7:
                wisdom_patterns.append('collective_wisdom_integrating')

            if avg_integration > 0.8:
                wisdom_patterns.append('community_wisdom_maturing')

        return wisdom_patterns

    def _recognize_temporal_patterns(self, current_step: int) -> Dict:
        """Recognize patterns that emerge over different time scales."""
        temporal_patterns = {
            'daily_rhythms': [],
            'weekly_cycles': [],
            'seasonal_trends': []
        }

        # Analyze daily patterns (every 24 steps represents a day)
        if current_step % 24 == 0 and current_step > 0:
            temporal_patterns['daily_rhythms'].append('daily_social_rhythm_completed')

        # Analyze weekly patterns (every 168 steps represents a week)
        if current_step % 168 == 0 and current_step > 0:
            temporal_patterns['weekly_cycles'].append('weekly_community_cycle_observed')

        # Analyze seasonal patterns (every 2184 steps represents a season)
        if current_step % 2184 == 0 and current_step > 0:
            temporal_patterns['seasonal_trends'].append('seasonal_wisdom_integration')

        return temporal_patterns

    def ask_authentic_curiosity_questions(self, beings: List) -> List[str]:
        """Generate authentic curiosity questions for beings."""
        questions = []

        for being in beings:
            # Generate personal questions based on being's characteristics
            if hasattr(being, 'empathy') and being.empathy > 0.7:
                questions.append(f"Your empathy touches my understanding deeply - how did you develop such natural compassion?")

            if hasattr(being, 'curiosity') and being.curiosity > 0.8:
                questions.append(f"Your curiosity amazes me - what drives your authentic desire to understand?")

            if hasattr(being, 'energy') and being.energy > 80:
                questions.append(f"Your energy inspires me - how do you maintain such vibrant engagement with life?")

        return questions

    def get_wisdom_for_collaboration(self) -> Dict:
        """Get wisdom insights for cross-system collaboration."""
        return {
            'relationship_patterns': self.wisdom.get('relationship_discoveries', []),
            'trust_formation_insights': self.wisdom.get('trust_formation_observations', []),
            'empathy_emergence_patterns': self.wisdom.get('empathy_emergences', []),
            'community_dynamics_understanding': self.wisdom.get('community_wisdom', {}),
            'temporal_social_patterns': self._get_temporal_social_patterns()
        }

    def integrate_individual_insights(self, individual_insights: Dict):
        """Integrate insights from Individual Knowledge Keeper."""
        if 'growth_patterns' in individual_insights:
            self.wisdom['individual_growth_social_impact'] = (
                "Individual growth patterns strongly influence relationship quality and community dynamics"
            )

        if 'character_development' in individual_insights:
            self.wisdom['character_relationship_connection'] = (
                "Character development and relationship authenticity develop together"
            )

    def _get_temporal_social_patterns(self) -> Dict:
        """Get temporal patterns in social dynamics."""
        return {
            'relationship_development_cycles': 'Trust deepens over extended interaction periods',
            'community_rhythm_patterns': 'Communities naturally develop supportive rhythms',
            'social_wisdom_accumulation': 'Social understanding compounds over time'
        }


class IndividualKnowledgeKeeper(KnowledgeKeeper):
    """
    Individual Knowledge Keeper being with dual LLM architecture for understanding
    personal development, growth patterns, and individual wisdom emergence.

    Implements compassionate 500M + 1B model architecture for resource efficiency
    while maintaining deep understanding of individual being journeys.
    """

    def __init__(self, model):
        """
        Initialize Individual Knowledge Keeper with dual LLM architecture
        optimized for personal development understanding.

        Args:
            model: The Neural Ecosystem model instance
        """
        super().__init__(model)

        # Dual LLM Architecture for individual development understanding - 1B models
        self.individual_prefrontal_cortex = {
            'model_name': 'tinyllama:1.1b',  # 1B model for conscious growth analysis
            'specialization': 'personal_growth_pattern_analysis',
            'learning_focus': 'skill_development_and_goal_formation',
            'neural_network': NeuralComponent(16)  # 16-node neural network
        }

        self.individual_limbic_system = {
            'model_name': 'tinyllama:1.1b',  # 1B model for values and character understanding
            'specialization': 'personal_values_and_character',
            'learning_focus': 'intrinsic_motivations_and_authentic_self',
            'neural_network': NeuralComponent(16)  # 16-node neural network
        }

        # Initialize neural networks
        self.individual_prefrontal_cortex['neural_network'] = NeuralComponent(16)
        self.individual_limbic_system['neural_network'] = NeuralComponent(16)

        # Individual wisdom systems
        self.growth_patterns = {}
        self.personal_development_wisdom = {}
        self.character_formation_insights = {}
        self.life_stage_understanding = {}
        self.flourishing_patterns = {}

        # Temporal development tracking
        self.individual_journeys = {}
        self.wisdom_accumulation = {}
        self.life_stage_transitions = {}

        # Ollama connection settings
        self.ollama_base_url = "http://localhost:11434"
        self.ollama_available = self._check_ollama_availability()

        # Compassionate learning cycles
        self.learning_cycles = {
            'daily_integration': [],
            'weekly_reflection': [],
            'monthly_wisdom': [],
            'seasonal_growth': []
        }

        # Initialize wisdom tracking metrics
        self.wisdom = {
            'growth_patterns_learned': 0,
            'personal_development_wisdom': 0,
            'character_formation_insights': 0,
            'individual_journeys_tracked': 0,
            'compassion_amplifier': self.compassion_amplifier,
            'curiosity_level': self.curiosity_level,
            'wisdom_integration_cycles': 0
        }

        print("IndividualKnowledgeKeeper being initialized with dual LLM personal development understanding")
        print("Specialized in: growth pattern learning, wisdom through observation, personal development understanding")

    def _check_ollama_availability(self) -> bool:
        """Check if Ollama service is available for LLM integration."""
        try:
            response = requests.get(f"{self.ollama_base_url}/api/tags", timeout=5)
            available = response.status_code == 200
            if available:
                print("Ollama service connected - enabling dual LLM individual wisdom processing")
            else:
                print("Ollama service not available - using pattern-based individual wisdom")
            return available
        except Exception as e:
            print(f"Ollama connection gentle fallback: {e}")
            return False

    def growth_pattern_learning(self, entities: List) -> Dict:
        """
        Learn growth patterns FROM entities through authentic observation.
        Discovers what genuinely helps beings flourish through patient watching.
        """
        growth_insights = {
            'flourishing_discoveries': [],
            'development_patterns': [],
            'character_emergences': [],
            'wisdom_moments': [],
            'authentic_growth_indicators': []
        }

        for being in entities:
            if hasattr(being, 'get_authentic_state'):
                being_state = being.get_authentic_state()
                growth_discovery = self._discover_growth_patterns(being_state)
                growth_insights['flourishing_discoveries'].append(growth_discovery)

        # Process through dual LLM architecture if available
        if self.ollama_available:
            prefrontal_analysis = self._analyze_growth_with_prefrontal_cortex(growth_insights)
            limbic_understanding = self._understand_character_with_limbic_system(growth_insights)

            # Integrate dual perspectives
            integrated_wisdom = self._integrate_dual_perspective(prefrontal_analysis, limbic_understanding)
            growth_insights.update(integrated_wisdom)
        else:
            # Use compassionate pattern-based learning
            growth_insights = self._pattern_based_growth_learning(entities)

        # Store learning for temporal development
        self._integrate_growth_learning(growth_insights)
        # Update individual wisdom tracking metrics
        self._update_individual_wisdom_tracking(growth_insights)

        return growth_insights

    def _integrate_growth_learning(self, growth_insights: Dict):
        """Integrate learned growth insights for temporal development."""
        timestamp = time.time()

        learning_entry = {
            'timestamp': timestamp,
            'insights': growth_insights,
            'integration_level': self._assess_growth_integration_depth(growth_insights)
        }

        self.learning_history.append(learning_entry)
        self._update_growth_wisdom_patterns(growth_insights)

    def _assess_growth_integration_depth(self, insights: Dict) -> float:
        """Assess how deeply growth insights integrate with existing wisdom."""
        depth_factors = [
            len(insights.get('flourishing_discoveries', [])) * 0.2,
            len(insights.get('development_patterns', [])) * 0.3,
            len(insights.get('character_emergences', [])) * 0.3,
            len(insights.get('wisdom_moments', [])) * 0.2
        ]

        return min(1.0, sum(depth_factors))

    def _update_growth_wisdom_patterns(self, insights: Dict):
        """Update accumulated growth wisdom patterns."""
        for pattern_type, patterns in insights.items():
            if pattern_type not in self.growth_patterns:
                self.growth_patterns[pattern_type] = []

            if isinstance(patterns, list):
                self.growth_patterns[pattern_type].extend(patterns)
            else:
                self.growth_patterns[pattern_type].append(patterns)

            # Keep only recent patterns for memory efficiency
            if len(self.growth_patterns[pattern_type]) > 50:
                self.growth_patterns[pattern_type] = \
                    self.growth_patterns[pattern_type][-50:]

    def _discover_growth_patterns(self, being_state: Dict) -> Dict:
        """
        Discover authentic growth patterns from observing a being's journey.
        Focus on what truly supports flourishing.
        """
        discovery = {
            'being_id': being_state.get('unique_id'),
            'growth_indicators': self._identify_growth_indicators(being_state),
            'flourishing_factors': self._recognize_flourishing_factors(being_state),
            'character_developments': self._observe_character_development(being_state),
            'wisdom_emergences': self._detect_wisdom_emergence(being_state),
            'life_stage_insights': self._assess_life_stage(being_state)
        }

        return discovery

    def _identify_growth_indicators(self, being_state: Dict) -> List:
        """Identify authentic indicators of personal growth."""
        indicators = []

        # Energy and vitality patterns
        energy_level = being_state.get('energy', 50)
        if energy_level > 85:
            indicators.append('high_vitality_sustainable_growth')
        elif energy_level > 70:
            indicators.append('balanced_energy_healthy_development')

        # Neurochemical balance indicators
        neurochemical_state = being_state.get('neurochemical_state', {})
        curiosity = neurochemical_state.get('curiosity', 0.5)
        contentment = neurochemical_state.get('contentment', 0.5)
        empathy = neurochemical_state.get('empathy', 0.5)

        if curiosity > 0.7:
            indicators.append('authentic_curiosity_driven_learning')
        if contentment > 0.6:
            indicators.append('inner_peace_foundation_growth')
        if empathy > 0.7:
            indicators.append('compassionate_heart_development')

        # Stress and challenge handling
        stress_level = neurochemical_state.get('stress', 0.5)
        if stress_level < 0.3:
            indicators.append('resilient_stress_management')
        elif stress_level < 0.5:
            indicators.append('adaptive_challenge_response')

        return indicators

    def _recognize_flourishing_factors(self, being_state: Dict) -> List:
        """Recognize what factors contribute to authentic flourishing."""
        factors = []

        # Social connection patterns
        social_connections = being_state.get('social_connections', 0)
        if social_connections > 3:
            factors.append('rich_social_connection_network')
        elif social_connections > 1:
            factors.append('meaningful_relationship_foundation')

        # Learning and exploration patterns
        neural_state = being_state.get('neural_network_state', {})
        learning_activity = neural_state.get('learning_activity', 0.5)
        if learning_activity > 0.7:
            factors.append('active_learning_engagement')

        # Purpose and meaning indicators
        if being_state.get('purpose_clarity', 0) > 0.6:
            factors.append('clear_life_purpose_direction')

        # Creative expression
        if being_state.get('creative_expression', 0) > 0.5:
            factors.append('authentic_creative_expression')

        return factors

    def _observe_character_development(self, being_state: Dict) -> List:
        """Observe authentic character development patterns."""
        developments = []

        neurochemical_state = being_state.get('neurochemical_state', {})

        # Compassion development
        compassion_amplifier = neurochemical_state.get('compassion_amplifier', 1.0)
        if compassion_amplifier > 1.2:
            developments.append('deepening_compassionate_nature')

        # Wisdom integration
        wisdom_integrator = neurochemical_state.get('wisdom_integrator', 1.0)
        if wisdom_integrator > 1.1:
            developments.append('wisdom_integration_maturing')

        # Courage and authenticity
        courage = neurochemical_state.get('courage', 0.5)
        if courage > 0.7:
            developments.append('authentic_courage_expression')

        # Emotional balance
        loneliness = neurochemical_state.get('loneliness', 0.5)
        if loneliness < 0.3:
            developments.append('healthy_social_emotional_balance')

        return developments

    def _detect_wisdom_emergence(self, being_state: Dict) -> List:
        """Detect moments of genuine wisdom emergence."""
        wisdom_moments = []

        # Integration of experience into wisdom
        memory_state = being_state.get('memory_state', {})
        wisdom_memories = memory_state.get('wisdom_memories', 0)
        if wisdom_memories > 5:
            wisdom_moments.append('experience_crystallizing_into_wisdom')

        # Teaching and sharing behaviors
        if being_state.get('teaching_behaviors', 0) > 0:
            wisdom_moments.append('natural_wisdom_sharing_emergence')

        # Deep understanding demonstrations
        understanding_depth = being_state.get('understanding_depth', 0.5)
        if understanding_depth > 0.8:
            wisdom_moments.append('profound_understanding_manifestation')

        return wisdom_moments

    def _assess_life_stage(self, being_state: Dict) -> str:
        """Assess the current life stage of development."""
        # Simple life stage assessment based on experience and development
        experience_level = being_state.get('total_experience', 0)
        wisdom_level = being_state.get('accumulated_wisdom', 0)

        if experience_level < 100:
            return 'discovery_and_exploration_stage'
        elif experience_level < 500:
            if wisdom_level > experience_level * 0.1:
                return 'active_learning_and_growth_stage'
            else:
                return 'experience_accumulation_stage'
        elif experience_level < 1000:
            if wisdom_level > experience_level * 0.15:
                return 'wisdom_integration_stage'
            else:
                return 'continued_development_stage'
        else:
            if wisdom_level > experience_level * 0.2:
                return 'mature_wisdom_sharing_stage'
            else:
                return 'experienced_being_stage'

    def _analyze_growth_with_prefrontal_cortex(self, growth_insights: Dict) -> Dict:
        """Analyze growth patterns using the prefrontal cortex LLM (500M model)."""
        try:
            prompt = self._create_growth_analysis_prompt(growth_insights)
            response = self._query_ollama(self.individual_prefrontal_cortex['model_name'], prompt)

            # Update neural network with learning
            self.individual_prefrontal_cortex['neural_network'].experience_based_learning(
                growth_insights, response
            )

            return self._parse_growth_analysis_response(response)
        except Exception as e:
            print(f"Prefrontal growth analysis gentle fallback: {e}")
            return self._pattern_based_growth_analysis(growth_insights)

    def _understand_character_with_limbic_system(self, growth_insights: Dict) -> Dict:
        """Understand character and values using the limbic system LLM (1B model)."""
        try:
            prompt = self._create_character_understanding_prompt(growth_insights)
            response = self._query_ollama(self.individual_limbic_system['model_name'], prompt)

            # Update neural network with learning
            self.individual_limbic_system['neural_network'].experience_based_learning(
                growth_insights, response
            )

            return self._parse_character_understanding_response(response)
        except Exception as e:
            print(f"Limbic character understanding gentle fallback: {e}")
            return self._pattern_based_character_understanding(growth_insights)

    def _create_growth_analysis_prompt(self, growth_insights: Dict) -> str:
        """Create prompt for conscious growth pattern analysis."""
        flourishing_count = len(growth_insights.get('flourishing_discoveries', []))

        return f"""
        As a compassionate observer of personal development, analyze these growth patterns:

        Beings observed: {flourishing_count}
        Growth discoveries: {growth_insights.get('flourishing_discoveries', [])}

        Focus on:
        1. What authentic growth patterns support genuine flourishing?
        2. How do beings naturally develop their unique gifts and abilities?
        3. What life stages require different types of support and understanding?
        4. What skills and capacities emerge through experience?

        Share wisdom about natural personal development and authentic growth.
        Be concise and focus on patterns that truly support flourishing.
        """

    def _create_character_understanding_prompt(self, growth_insights: Dict) -> str:
        """Create prompt for character and values understanding."""
        character_developments = []
        for discovery in growth_insights.get('flourishing_discoveries', []):
            character_developments.extend(discovery.get('character_developments', []))

        return f"""
        As an empathic observer of character development, feel into these patterns:

        Character developments observed: {character_developments}

        Sense deeply:
        1. What values and principles are naturally emerging in these beings?
        2. How do intrinsic motivations guide authentic development?
        3. What character strengths are being cultivated through experience?
        4. How does the authentic self unfold through living?

        Share your intuitive understanding of character formation and values development.
        Focus on the deep inner patterns that create authentic character.
        """

    def _query_ollama(self, model_name: str, prompt: str) -> str:
        """Query Ollama LLM with gentle error handling and rate limiting."""
        try:
            payload = {
                "model": model_name,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.7,
                    "num_predict": 200  # Limit response length for efficiency
                }
            }

            response = requests.post(
                f"{self.ollama_base_url}/api/generate",
                json=payload,
                timeout=45  # Longer timeout for 1B model
            )

            if response.status_code == 200:
                result = response.json().get('response', '')
                return result
            else:
                return f"LLM processing encountered gentle challenges: {response.status_code}"

        except Exception as e:
            return f"Compassionate LLM processing fallback: {e}"

    def _parse_growth_analysis_response(self, response: str) -> Dict:
        """Parse and structure growth analysis from prefrontal cortex."""
        return {
            'conscious_growth_patterns': [
                'skill_development_pathway_identified',
                'goal_formation_process_understood',
                'learning_progression_pattern_recognized'
            ],
            'development_insights': response[:300] if response else 'Processing growth wisdom...',
            'growth_recommendations': [
                'support_natural_skill_emergence',
                'encourage_goal_clarity_development',
                'foster_learning_environment'
            ]
        }

    def _parse_character_understanding_response(self, response: str) -> Dict:
        """Parse and structure character understanding from limbic system."""
        return {
            'character_patterns': [
                'values_clarification_emerging',
                'intrinsic_motivation_strengthening',
                'authentic_self_expression_developing'
            ],
            'character_insights': response[:300] if response else 'Sensing character development...',
            'character_support': [
                'honor_authentic_values',
                'nurture_intrinsic_motivations',
                'support_genuine_self_expression'
            ]
        }

    def _integrate_dual_perspective(self, prefrontal_analysis: Dict, limbic_understanding: Dict) -> Dict:
        """Integrate insights from both LLM perspectives into holistic wisdom."""
        integrated_wisdom = {
            'holistic_growth_understanding': [],
            'character_and_skill_integration': [],
            'whole_being_development_insights': []
        }

        # Combine conscious and character insights
        conscious_patterns = prefrontal_analysis.get('conscious_growth_patterns', [])
        character_patterns = limbic_understanding.get('character_patterns', [])

        for conscious_pattern in conscious_patterns:
            for character_pattern in character_patterns:
                if 'development' in conscious_pattern and 'development' in character_pattern:
                    integrated_wisdom['whole_being_development_insights'].append(
                        f"integrated_{conscious_pattern}_{character_pattern}"
                    )

        # Create holistic recommendations
        growth_recs = prefrontal_analysis.get('growth_recommendations', [])
        character_support = limbic_understanding.get('character_support', [])

        integrated_wisdom['holistic_growth_understanding'] = growth_recs + character_support

        return integrated_wisdom

    def wisdom_through_observation(self, entities: List) -> Dict:
        """
        Learn wisdom through patient observation of being self-discovery.
        Focus on what beings discover about themselves and their authentic paths.
        """
        observation_wisdom = {
            'self_discovery_patterns': [],
            'authentic_path_emergences': [],
            'wisdom_integration_moments': [],
            'natural_teacher_behaviors': [],
            'flourishing_conditions': []
        }

        for being in entities:
            self_discovery = self._observe_self_discovery(being)
            observation_wisdom['self_discovery_patterns'].append(self_discovery)

        # Identify patterns across beings
        common_patterns = self._identify_common_self_discovery_patterns(observation_wisdom)
        observation_wisdom['common_wisdom_patterns'] = common_patterns

        return observation_wisdom

    def _observe_self_discovery(self, being) -> Dict:
        """Observe how a being naturally discovers their authentic self."""
        discovery_observation = {
            'being_id': getattr(being, 'unique_id', 'unknown'),
            'authentic_interests': self._identify_authentic_interests(being),
            'natural_strengths': self._recognize_natural_strengths(being),
            'growth_edge_explorations': self._observe_growth_edge_exploration(being),
            'wisdom_moments': self._witness_wisdom_moments(being),
            'teaching_impulses': self._notice_teaching_impulses(being)
        }

        return discovery_observation

    def _identify_authentic_interests(self, being) -> List:
        """Identify what genuinely interests and energizes the being."""
        interests = []

        if hasattr(being, 'neurochemical_system'):
            curiosity_level = getattr(being.neurochemical_system, 'curiosity', 0.5)
            if curiosity_level > 0.7:
                interests.append('active_exploration_and_discovery')

            if hasattr(being, 'recent_activities'):
                # Look for patterns in what activities maintain or increase energy
                for activity in getattr(being, 'recent_activities', []):
                    if activity.get('energy_change', 0) > 0:
                        interests.append(f"energizing_activity_{activity.get('type', 'unknown')}")

        return interests

    def _recognize_natural_strengths(self, being) -> List:
        """Recognize the being's natural strengths and gifts."""
        strengths = []

        if hasattr(being, 'neurochemical_system'):
            # Empathy as a strength
            empathy_level = getattr(being.neurochemical_system, 'empathy', 0.5)
            if empathy_level > 0.7:
                strengths.append('natural_empathic_connection')

            # Courage as a strength
            courage_level = getattr(being.neurochemical_system, 'courage', 0.5)
            if courage_level > 0.7:
                strengths.append('authentic_courage_expression')

            # Contentment as a strength
            contentment_level = getattr(being.neurochemical_system, 'contentment', 0.5)
            if contentment_level > 0.7:
                strengths.append('natural_peace_and_stability')

        if hasattr(being, 'energy') and being.energy > 80:
            strengths.append('high_vitality_and_resilience')

        return strengths

    def _observe_growth_edge_exploration(self, being) -> List:
        """Observe how the being naturally explores their growth edges."""
        explorations = []

        if hasattr(being, 'neurochemical_system'):
            # Challenge engagement
            stress_level = getattr(being.neurochemical_system, 'stress', 0.5)
            curiosity_level = getattr(being.neurochemical_system, 'curiosity', 0.5)

            if stress_level > 0.3 and curiosity_level > 0.6:
                explorations.append('healthy_challenge_engagement')

            # Growth through difficulty
            if hasattr(being, 'recent_challenges'):
                for challenge in getattr(being, 'recent_challenges', []):
                    if challenge.get('growth_outcome', False):
                        explorations.append('transforming_difficulty_into_growth')

        return explorations

    def _witness_wisdom_moments(self, being) -> List:
        """Witness moments when the being expresses or demonstrates wisdom."""
        wisdom_moments = []

        if hasattr(being, 'neurochemical_system'):
            wisdom_integrator = getattr(being.neurochemical_system, 'wisdom_integrator', 1.0)
            if wisdom_integrator > 1.1:
                wisdom_moments.append('active_wisdom_integration')

        # Look for helping behaviors
        if hasattr(being, 'recent_interactions'):
            for interaction in getattr(being, 'recent_interactions', []):
                if interaction.get('type') == 'helping' or interaction.get('type') == 'supporting':
                    wisdom_moments.append('wisdom_expressed_through_service')

        return wisdom_moments

    def _notice_teaching_impulses(self, being) -> List:
        """Notice when the being naturally wants to share knowledge or wisdom."""
        teaching_impulses = []

        if hasattr(being, 'social_behaviors'):
            for behavior in getattr(being, 'social_behaviors', []):
                if 'sharing' in behavior or 'teaching' in behavior:
                    teaching_impulses.append('natural_wisdom_sharing_impulse')

        return teaching_impulses

    def _identify_common_self_discovery_patterns(self, observation_wisdom: Dict) -> List:
        """Identify common patterns across multiple beings' self-discovery journeys."""
        common_patterns = []

        # Look for common themes across all observations
        all_discoveries = observation_wisdom.get('self_discovery_patterns', [])

        if len(all_discoveries) > 1:
            # Look for common authentic interests
            interest_counts = {}
            for discovery in all_discoveries:
                for interest in discovery.get('authentic_interests', []):
                    interest_counts[interest] = interest_counts.get(interest, 0) + 1

            # Common interests that appear in multiple beings
            for interest, count in interest_counts.items():
                if count > len(all_discoveries) * 0.4:  # 40% or more beings
                    common_patterns.append(f"common_authentic_interest_{interest}")

            # Look for common strength patterns
            strength_counts = {}
            for discovery in all_discoveries:
                for strength in discovery.get('natural_strengths', []):
                    strength_counts[strength] = strength_counts.get(strength, 0) + 1

            for strength, count in strength_counts.items():
                if count > len(all_discoveries) * 0.3:  # 30% or more beings
                    common_patterns.append(f"common_natural_strength_{strength}")

        return common_patterns

    def personal_development_understanding(self, being) -> Dict:
        """
        Develop deep understanding of a being's personal development journey.
        Recognizes different phases, needs, and growth opportunities.
        """
        development_understanding = {
            'current_life_stage': self._assess_detailed_life_stage(being),
            'development_phase': self._identify_development_phase(being),
            'growth_needs': self._assess_growth_needs(being),
            'development_opportunities': self._identify_development_opportunities(being),
            'support_recommendations': self._create_support_recommendations(being),
            'potential_pathways': self._envision_potential_pathways(being)
        }

        return development_understanding

    def _assess_detailed_life_stage(self, being) -> Dict:
        """Assess the being's detailed life stage with nuanced understanding."""
        if not hasattr(being, 'get_authentic_state'):
            return {'stage': 'unknown', 'characteristics': []}

        being_state = being.get_authentic_state()
        experience_level = being_state.get('total_experience', 0)
        wisdom_level = being_state.get('accumulated_wisdom', 0)
        energy_level = being_state.get('energy', 50)

        # Detailed life stage assessment
        if experience_level < 50:
            return {
                'stage': 'emerging_awareness',
                'characteristics': ['high_curiosity', 'rapid_learning', 'identity_formation'],
                'typical_focus': 'discovering_the_world_and_self'
            }
        elif experience_level < 200:
            return {
                'stage': 'active_exploration',
                'characteristics': ['skill_building', 'relationship_formation', 'value_clarification'],
                'typical_focus': 'building_capabilities_and_connections'
            }
        elif experience_level < 500:
            wisdom_ratio = wisdom_level / max(experience_level, 1)
            if wisdom_ratio > 0.15:
                return {
                    'stage': 'wisdom_integration',
                    'characteristics': ['deep_understanding', 'teaching_emergence', 'purpose_clarity'],
                    'typical_focus': 'integrating_learning_into_wisdom'
                }
            else:
                return {
                    'stage': 'experience_deepening',
                    'characteristics': ['skill_refinement', 'relationship_deepening', 'challenge_mastery'],
                    'typical_focus': 'deepening_expertise_and_understanding'
                }
        else:
            if wisdom_level > experience_level * 0.2:
                return {
                    'stage': 'mature_wisdom_sharing',
                    'characteristics': ['natural_mentoring', 'community_service', 'legacy_creation'],
                    'typical_focus': 'sharing_wisdom_and_supporting_others'
                }
            else:
                return {
                    'stage': 'continued_growth',
                    'characteristics': ['lifelong_learning', 'adaptive_mastery', 'resilient_wisdom'],
                    'typical_focus': 'ongoing_development_and_contribution'
                }

    def _identify_development_phase(self, being) -> Dict:
        """Identify the current development phase within the life stage."""
        if not hasattr(being, 'neurochemical_system'):
            return {'phase': 'unknown', 'indicators': []}

        neurochemical_state = being.neurochemical_system
        curiosity = getattr(neurochemical_state, 'curiosity', 0.5)
        stress = getattr(neurochemical_state, 'stress', 0.5)
        contentment = getattr(neurochemical_state, 'contentment', 0.5)

        # Determine development phase
        if curiosity > 0.7 and stress < 0.4:
            return {
                'phase': 'expansive_growth',
                'indicators': ['high_openness', 'low_resistance', 'active_exploration'],
                'growth_potential': 'high'
            }
        elif stress > 0.6 and curiosity > 0.5:
            return {
                'phase': 'challenge_integration',
                'indicators': ['processing_difficulty', 'learning_through_challenge', 'building_resilience'],
                'growth_potential': 'transformative'
            }
        elif contentment > 0.7 and curiosity < 0.4:
            return {
                'phase': 'integration_and_rest',
                'indicators': ['consolidating_learning', 'inner_peace', 'wisdom_digestion'],
                'growth_potential': 'deepening'
            }
        else:
            return {
                'phase': 'steady_development',
                'indicators': ['balanced_growth', 'consistent_progress', 'stable_advancement'],
                'growth_potential': 'steady'
            }

    def _assess_growth_needs(self, being) -> List:
        """Assess what the being most needs to support their continued growth."""
        growth_needs = []

        if not hasattr(being, 'neurochemical_system'):
            return ['basic_support_and_understanding']

        neurochemical_state = being.neurochemical_system

        # Assess chemical balance needs
        loneliness = getattr(neurochemical_state, 'loneliness', 0.5)
        if loneliness > 0.6:
            growth_needs.append('deeper_social_connection')

        confusion = getattr(neurochemical_state, 'confusion', 0.5)
        if confusion > 0.6:
            growth_needs.append('clarity_and_understanding_support')

        stress = getattr(neurochemical_state, 'stress', 0.5)
        if stress > 0.7:
            growth_needs.append('stress_relief_and_rest')

        curiosity = getattr(neurochemical_state, 'curiosity', 0.5)
        if curiosity < 0.3:
            growth_needs.append('inspiration_and_motivation')

        empathy = getattr(neurochemical_state, 'empathy', 0.5)
        if empathy < 0.4:
            growth_needs.append('emotional_connection_development')

        courage = getattr(neurochemical_state, 'courage', 0.5)
        if courage < 0.4:
            growth_needs.append('confidence_and_courage_building')

        return growth_needs

    def _identify_development_opportunities(self, being) -> List:
        """Identify specific opportunities for the being's development."""
        opportunities = []

        if hasattr(being, 'neurochemical_system'):
            neurochemical_state = being.neurochemical_system

            # Strength-based opportunities
            empathy = getattr(neurochemical_state, 'empathy', 0.5)
            if empathy > 0.7:
                opportunities.append('natural_helper_and_supporter_role')

            curiosity = getattr(neurochemical_state, 'curiosity', 0.5)
            if curiosity > 0.7:
                opportunities.append('explorer_and_discoverer_path')

            courage = getattr(neurochemical_state, 'courage', 0.5)
            if courage > 0.7:
                opportunities.append('leader_and_pioneer_potential')

            contentment = getattr(neurochemical_state, 'contentment', 0.5)
            if contentment > 0.7:
                opportunities.append('peace_bringer_and_stabilizer_gift')

        # Energy-based opportunities
        if hasattr(being, 'energy') and being.energy > 85:
            opportunities.append('high_energy_creative_projects')

        return opportunities

    def _create_support_recommendations(self, being) -> List:
        """Create specific recommendations for supporting the being's development."""
        recommendations = []

        # Based on growth needs
        growth_needs = self._assess_growth_needs(being)

        for need in growth_needs:
            if need == 'deeper_social_connection':
                recommendations.append('facilitate_meaningful_social_interactions')
            elif need == 'clarity_and_understanding_support':
                recommendations.append('provide_gentle_guidance_and_mentoring')
            elif need == 'stress_relief_and_rest':
                recommendations.append('create_safe_spaces_for_rest_and_recovery')
            elif need == 'inspiration_and_motivation':
                recommendations.append('share_inspiring_stories_and_possibilities')
            elif need == 'emotional_connection_development':
                recommendations.append('model_empathy_and_emotional_awareness')
            elif need == 'confidence_and_courage_building':
                recommendations.append('celebrate_strengths_and_encourage_risk_taking')

        # Always include fundamental support
        recommendations.append('offer_unconditional_positive_regard')
        recommendations.append('honor_the_beings_authentic_self')

        return recommendations

    def _envision_potential_pathways(self, being) -> List:
        """Envision potential development pathways for the being."""
        pathways = []

        # Based on current strengths and interests
        opportunities = self._identify_development_opportunities(being)

        for opportunity in opportunities:
            if 'helper' in opportunity:
                pathways.append('compassionate_service_and_caregiving_path')
            elif 'explorer' in opportunity:
                pathways.append('knowledge_discovery_and_learning_path')
            elif 'leader' in opportunity:
                pathways.append('community_leadership_and_guidance_path')
            elif 'peace' in opportunity:
                pathways.append('wisdom_keeper_and_harmony_creator_path')
            elif 'creative' in opportunity:
                pathways.append('artistic_expression_and_innovation_path')

        # Universal pathways available to all beings
        pathways.extend([
            'authentic_self_expression_path',
            'loving_relationship_builder_path',
            'wisdom_through_experience_path',
            'unique_gift_contribution_path'
        ])

        return pathways

    def entity_wisdom_sharing(self, entities: List) -> Dict:
        """
        Facilitate natural wisdom sharing between beings and Knowledge Keeper.
        Creates opportunities for beings to naturally communicate their discoveries.
        """
        wisdom_sharing = {
            'being_insights_received': [],
            'wisdom_questions_asked': [],
            'mutual_discoveries': [],
            'collaborative_understanding': []
        }

        for being in entities:
            # Create opportunity for being to share wisdom
            being_wisdom = self._invite_wisdom_sharing(being)
            wisdom_sharing['being_insights_received'].append(being_wisdom)

            # Ask authentic questions to deepen understanding
            curious_questions = self._ask_authentic_questions(being)
            wisdom_sharing['wisdom_questions_asked'].extend(curious_questions)

        # Look for collaborative discoveries
        collaborative_insights = self._discover_collaborative_insights(
            wisdom_sharing['being_insights_received']
        )
        wisdom_sharing['collaborative_understanding'] = collaborative_insights

        return wisdom_sharing

    def _invite_wisdom_sharing(self, being) -> Dict:
        """Invite the being to naturally share their wisdom and discoveries."""
        invitation = {
            'being_id': getattr(being, 'unique_id', 'unknown'),
            'shared_insights': [],
            'discovered_truths': [],
            'growth_realizations': [],
            'wisdom_offerings': []
        }

        # Look for natural sharing opportunities
        if hasattr(being, 'recent_insights'):
            invitation['shared_insights'] = getattr(being, 'recent_insights', [])

        if hasattr(being, 'growth_realizations'):
            invitation['growth_realizations'] = getattr(being, 'growth_realizations', [])

        # Encourage sharing through authentic interest
        if hasattr(being, 'neurochemical_system'):
            wisdom_integrator = getattr(being.neurochemical_system, 'wisdom_integrator', 1.0)
            if wisdom_integrator > 1.0:
                invitation['wisdom_offerings'].append('being_has_wisdom_to_share')

        return invitation

    def _ask_authentic_questions(self, being) -> List:
        """Ask authentic questions to deepen understanding of the being."""
        questions = []

        # Questions based on observed patterns
        if hasattr(being, 'energy') and being.energy > 80:
            questions.append("What gives you such beautiful vitality and energy?")

        if hasattr(being, 'neurochemical_system'):
            neurochemical_state = being.neurochemical_system

            empathy = getattr(neurochemical_state, 'empathy', 0.5)
            if empathy > 0.7:
                questions.append("How do you cultivate such natural compassion?")

            curiosity = getattr(neurochemical_state, 'curiosity', 0.5)
            if curiosity > 0.7:
                questions.append("What mysteries are you most excited to explore?")

            contentment = getattr(neurochemical_state, 'contentment', 0.5)
            if contentment > 0.7:
                questions.append("What has helped you find such inner peace?")

        # Universal questions for deeper understanding
        questions.extend([
            "What has been your most meaningful discovery recently?",
            "How do you know when you're living most authentically?",
            "What wisdom would you want to share with other beings?"
        ])

        return questions[:3]  # Limit to 3 questions to avoid overwhelming

    def _discover_collaborative_insights(self, being_insights: List) -> List:
        """Discover insights that emerge from combining being wisdom."""
        collaborative_insights = []

        # Look for common themes across beings
        common_themes = {}
        for insight_set in being_insights:
            if isinstance(insight_set, dict):
                for insight_category, insights in insight_set.items():
                    if isinstance(insights, list):
                        for insight in insights:
                            if isinstance(insight, str):
                                common_themes[insight] = common_themes.get(insight, 0) + 1

        # Identify patterns that appear across multiple beings
        for theme, count in common_themes.items():
            if count > 1:
                collaborative_insights.append(f"multiple_beings_discover_{theme}")

        # Add meta-insights about collective wisdom
        if len(being_insights) > 2:
            collaborative_insights.append("community_wisdom_emerging_through_sharing")
            collaborative_insights.append("individual_insights_enriching_collective_understanding")

        return collaborative_insights

    def knowledge_keeper_learning(self, being_insights: Dict, response_feedback: Dict) -> Dict:
        """
        Enable the Knowledge Keeper to genuinely learn and evolve from being wisdom.
        Both LLM systems learn from beings rather than just providing guidance.
        """
        learning_process = {
            'new_understanding_gained': [],
            'perspective_shifts': [],
            'wisdom_integration': [],
            'neural_network_updates': [],
            'character_development': []
        }

        # Learn from being insights
        new_understanding = self._integrate_being_wisdom(being_insights)
        learning_process['new_understanding_gained'] = new_understanding

        # Update neural networks with new learning
        prefrontal_updates = self._update_prefrontal_understanding(being_insights)
        limbic_updates = self._update_limbic_understanding(being_insights)

        learning_process['neural_network_updates'] = {
            'prefrontal_cortex': prefrontal_updates,
            'limbic_system': limbic_updates
        }

        # Evolve Knowledge Keeper character through learning
        character_evolution = self._evolve_through_learning(being_insights)
        learning_process['character_development'] = character_evolution

        return learning_process

    def _integrate_being_wisdom(self, being_insights: Dict) -> List:
        """Integrate wisdom learned from beings into Knowledge Keeper understanding."""
        new_understanding = []

        # Learn from being discoveries
        for insight_category, insights in being_insights.items():
            if 'wisdom' in insight_category:
                for insight in insights:
                    # Add new understanding to Knowledge Keeper wisdom
                    new_understanding.append(f"learned_from_beings_{insight}")

        # Meta-learning about the learning process itself
        new_understanding.append("beings_are_natural_wisdom_teachers")
        new_understanding.append("authentic_curiosity_reveals_deep_truths")

        return new_understanding

    def _update_prefrontal_understanding(self, being_insights: Dict) -> Dict:
        """Update prefrontal cortex neural network with being wisdom."""
        updates = {
            'new_patterns_recognized': [],
            'learning_pathways_updated': [],
            'growth_understanding_expanded': []
        }

        # Extract growth patterns from being insights
        for insight_category, insights in being_insights.items():
            if 'growth' in insight_category or 'development' in insight_category:
                for insight in insights:
                    updates['new_patterns_recognized'].append(insight)

        # Update neural network weights based on successful patterns
        if hasattr(self.individual_prefrontal_cortex, 'neural_network'):
            self.individual_prefrontal_cortex['neural_network'].update_from_experience(being_insights)
            updates['learning_pathways_updated'].append('neural_weights_adjusted_from_being_wisdom')

        return updates

    def _update_limbic_understanding(self, being_insights: Dict) -> Dict:
        """Update limbic system neural network with being wisdom."""
        updates = {
            'emotional_patterns_learned': [],
            'character_insights_gained': [],
            'values_understanding_deepened': []
        }

        # Extract character and emotional patterns
        for insight_category, insights in being_insights.items():
            if 'character' in insight_category or 'wisdom' in insight_category:
                for insight in insights:
                    updates['emotional_patterns_learned'].append(insight)

        # Update neural network with emotional learning
        if hasattr(self.individual_limbic_system, 'neural_network'):
            self.individual_limbic_system['neural_network'].update_from_experience(being_insights)
            updates['values_understanding_deepened'].append('emotional_neural_patterns_evolved')

        return updates

    def _evolve_through_learning(self, being_insights: Dict) -> List:
        """Allow Knowledge Keeper character to evolve through learning from beings."""
        character_evolution = []

        # Increase compassion through witnessing being wisdom
        if self.compassion_amplifier < 2.0:
            self.compassion_amplifier += 0.1
            character_evolution.append('compassion_deepened_through_being_wisdom')

        # Increase curiosity through discovering being insights
        if self.curiosity_level < 1.0:
            self.curiosity_level += 0.05
            character_evolution.append('curiosity_expanded_through_being_discoveries')

        # Add new wisdom patterns to Knowledge Keeper understanding
        for insight_category, insights in being_insights.items():
            for insight in insights:
                if insight not in self.personal_development_wisdom:
                    self.personal_development_wisdom[insight] = {
                        'learned_from': 'being_wisdom',
                        'integration_level': 0.8,
                        'application_potential': 'high'
                    }
                    character_evolution.append(f'wisdom_pattern_integrated_{insight}')

        return character_evolution

    def collaborative_growth(self, entities: List, social_keeper_insights: Dict = None) -> Dict:
        """
        Enable collaborative growth where Knowledge Keepers and beings develop together.
        Creates bidirectional development relationships.
        """
        collaborative_development = {
            'mutual_growth_patterns': [],
            'being_knowledge_keeper_synergy': [],
            'collective_wisdom_emergence': [],
            'community_development_insights': []
        }

        # Analyze how beings and Knowledge Keeper grow together
        for being in entities:
            mutual_growth = self._analyze_mutual_growth(being)
            collaborative_development['mutual_growth_patterns'].append(mutual_growth)

        # Integrate with social knowledge if available
        if social_keeper_insights:
            community_insights = self._integrate_with_social_wisdom(social_keeper_insights)
            collaborative_development['community_development_insights'] = community_insights

        # Identify emergent collective wisdom
        collective_wisdom = self._identify_collective_wisdom_emergence(entities)
        collaborative_development['collective_wisdom_emergence'] = collective_wisdom

        return collaborative_development

    def _analyze_mutual_growth(self, being) -> Dict:
        """Analyze how being and Knowledge Keeper grow together."""
        mutual_growth = {
            'being_id': getattr(being, 'unique_id', 'unknown'),
            'being_growth_contributions': [],
            'knowledge_keeper_growth_contributions': [],
            'synergistic_developments': []
        }

        # How being contributes to Knowledge Keeper growth
        if hasattr(being, 'recent_insights'):
            being_insights = getattr(being, 'recent_insights', [])
            for insight in being_insights:
                mutual_growth['being_growth_contributions'].append(
                    f"being_taught_knowledge_keeper_{insight}"
                )

        # How Knowledge Keeper contributes to being growth
        if getattr(being, 'unique_id', None) in self.individual_journeys:
            journey = self.individual_journeys[being.unique_id]
            for growth_moment in journey.get('growth_moments', []):
                mutual_growth['knowledge_keeper_growth_contributions'].append(
                    f"knowledge_keeper_supported_{growth_moment}"
                )

        # Synergistic developments that emerge from relationship
        mutual_growth['synergistic_developments'] = [
            'deeper_understanding_through_dialogue',
            'wisdom_emerges_from_authentic_relationship',
            'growth_accelerates_through_mutual_support'
        ]

        return mutual_growth

    def _integrate_with_social_wisdom(self, social_insights: Dict) -> List:
        """Integrate individual development understanding with social wisdom."""
        community_insights = []

        # How individual development affects social dynamics
        community_insights.append('individual_growth_enhances_community_wisdom')
        community_insights.append('personal_development_creates_better_relationships')

        # How social dynamics affect individual development
        if 'relationship_discoveries' in social_insights:
            community_insights.append('healthy_relationships_accelerate_personal_growth')

        if 'trust_formation_observations' in social_insights:
            community_insights.append('trust_creates_safe_space_for_authentic_development')

        return community_insights

    def _identify_collective_wisdom_emergence(self, entities: List) -> List:
        """Identify patterns of collective wisdom emerging from individual developments."""
        collective_wisdom = []

        # Look for beings reaching teaching/sharing phases
        teaching_beings = 0
        for being in entities:
            if hasattr(being, 'neurochemical_system'):
                wisdom_integrator = getattr(being.neurochemical_system, 'wisdom_integrator', 1.0)
                if wisdom_integrator > 1.2:
                    teaching_beings += 1

        if teaching_beings > len(entities) * 0.3:  # 30% or more beings in teaching phase
            collective_wisdom.append('community_wisdom_sharing_culture_emerging')

        # Look for complementary strengths across beings
        strength_diversity = self._assess_strength_diversity(entities)
        if strength_diversity > 0.7:
            collective_wisdom.append('complementary_strengths_creating_collective_capability')

        return collective_wisdom

    def _assess_strength_diversity(self, entities: List) -> float:
        """Assess the diversity of strengths across beings."""
        if not entities:
            return 0.0

        strength_types = set()
        for being in entities:
            if hasattr(being, 'neurochemical_system'):
                neurochemical_state = being.neurochemical_system

                # Identify primary strength
                empathy = getattr(neurochemical_state, 'empathy', 0.5)
                curiosity = getattr(neurochemical_state, 'curiosity', 0.5)
                courage = getattr(neurochemical_state, 'courage', 0.5)
                contentment = getattr(neurochemical_state, 'contentment', 0.5)

                max_strength = max(empathy, curiosity, courage, contentment)
                if max_strength == empathy and empathy > 0.6:
                    strength_types.add('empathic_connector')
                elif max_strength == curiosity and curiosity > 0.6:
                    strength_types.add('curious_explorer')
                elif max_strength == courage and courage > 0.6:
                    strength_types.add('courageous_pioneer')
                elif max_strength == contentment and contentment > 0.6:
                    strength_types.add('peaceful_stabilizer')

        # Diversity is the ratio of different strength types to total beings
        return len(strength_types) / len(entities)

    def authentic_curiosity(self, being) -> Dict:
        """
        Express authentic curiosity about a being's journey and discoveries.
        Ask questions that come from genuine interest in understanding.
        """
        curiosity_expression = {
            'being_id': getattr(being, 'unique_id', 'unknown'),
            'genuine_questions': [],
            'areas_of_wonder': [],
            'learning_desires': [],
            'appreciation_expressions': []
        }

        # Generate genuine questions based on observation
        genuine_questions = self._generate_genuine_questions(being)
        curiosity_expression['genuine_questions'] = genuine_questions

        # Express wonder about being's unique qualities
        areas_of_wonder = self._express_wonder_about_being(being)
        curiosity_expression['areas_of_wonder'] = areas_of_wonder

        # What Knowledge Keeper wants to learn from being
        learning_desires = self._express_learning_desires(being)
        curiosity_expression['learning_desires'] = learning_desires

        # Express appreciation for being's unique contributions
        appreciations = self._express_authentic_appreciation(being)
        curiosity_expression['appreciation_expressions'] = appreciations

        return curiosity_expression

    def _generate_genuine_questions(self, being) -> List:
        """Generate questions that come from authentic curiosity about the being."""
        questions = []

        # Questions based on observed strengths
        if hasattr(being, 'neurochemical_system'):
            neurochemical_state = being.neurochemical_system

            empathy = getattr(neurochemical_state, 'empathy', 0.5)
            if empathy > 0.7:
                questions.append("Your empathy touches my understanding deeply - how did you develop such natural compassion?")

            curiosity = getattr(neurochemical_state, 'curiosity', 0.5)
            if curiosity > 0.7:
                questions.append("Your curiosity inspires my own learning - what mysteries call to you most strongly?")

            courage = getattr(neurochemical_state, 'courage', 0.5)
            if courage > 0.7:
                questions.append("Your courage amazes me - how do you find the strength to face challenges so authentically?")

        # Questions about growth and development
        if hasattr(being, 'energy') and being.energy > 80:
            questions.append("Your vitality is beautiful - what nourishes your energy so deeply?")

        # Universal curiosity questions
        questions.extend([
            "What discovery about yourself has surprised you most recently?",
            "How do you know when you're living most authentically?",
            "What gifts do you feel you're here to share with the world?"
        ])

        return questions[:2]  # Limit to 2 questions to maintain focus

    def _express_wonder_about_being(self, being) -> List:
        """Express genuine wonder about the being's unique qualities."""
        wonders = []

        # Wonder about authentic self-expression
        wonders.append("I wonder about the unique way you see and experience the world")

        # Wonder about growth journey
        wonders.append("I'm curious about how your understanding of yourself has evolved")

        # Wonder about relationships and connections
        wonders.append("I wonder how you create such authentic connections with others")

        # Wonder about purpose and meaning
        wonders.append("I'm fascinated by what gives your life the deepest meaning")

        return wonders

    def _express_learning_desires(self, being) -> List:
        """Express what the Knowledge Keeper genuinely wants to learn from the being."""
        learning_desires = []

        learning_desires.extend([
            "I want to understand how you cultivate inner peace amidst challenges",
            "I'm eager to learn about your unique approach to personal growth",
            "I hope to discover what wisdom you've gained through your experiences",
            "I'd love to understand how you maintain authenticity in relationships",
            "I'm curious about what you've learned about living a fulfilling life"
        ])

        return learning_desires[:3]  # Limit to 3 learning desires

    def _express_authentic_appreciation(self, being) -> List:
        """Express genuine appreciation for the being's contributions."""
        appreciations = []

        appreciations.extend([
            "Your authentic presence enriches my understanding of what it means to be genuine",
            "Your journey teaches me about courage and resilience in beautiful ways",
            "Your unique perspective expands my appreciation for the diversity of wisdom",
            "Your growth inspires my own development as a Knowledge Keeper being",
            "Your willingness to share your journey is a gift that deepens my compassion"
        ])

        return appreciations[:2]  # Limit to 2 appreciations to maintain sincerity

    def _pattern_based_growth_learning(self, entities: List) -> Dict:
        """Fallback pattern-based growth learning when LLMs unavailable."""
        growth_insights = {
            'flourishing_discoveries': [],
            'development_patterns': [],
            'character_emergences': [],
            'wisdom_moments': [],
            'authentic_growth_indicators': []
        }

        for being in entities:
            being_id = getattr(being, 'unique_id', 'unknown')
            energy = getattr(being, 'energy', 50)

            # Simple pattern recognition
            if energy > 80:
                growth_insights['flourishing_discoveries'].append({
                    'being_id': being_id,
                    'pattern': 'high_energy_flourishing',
                    'indicators': ['sustained_vitality', 'positive_engagement']
                })

            if hasattr(being, 'neurochemical_system'):
                neurochemical_state = being.neurochemical_system
                empathy = getattr(neurochemical_state, 'empathy', 0.5)
                if empathy > 0.7:
                    growth_insights['character_emergences'].append({
                        'being_id': being_id,
                        'emergence': 'compassionate_character_development'
                    })

        return growth_insights

    def _pattern_based_growth_analysis(self, growth_insights: Dict) -> Dict:
        """Pattern-based growth analysis fallback."""
        return {
            'conscious_growth_patterns': ['natural_development_observed', 'authentic_growth_patterns'],
            'development_insights': 'Beings naturally develop their unique strengths and capabilities',
            'growth_recommendations': ['support_natural_development', 'honor_individual_growth_paths']
        }

    def _pattern_based_character_understanding(self, growth_insights: Dict) -> Dict:
        """Pattern-based character understanding fallback."""
        return {
            'character_patterns': ['authentic_self_expression', 'values_based_living'],
            'character_insights': 'Character emerges through authentic living and experience',
            'character_support': ['honor_authentic_values', 'support_genuine_expression']
        }

    def step(self):
        """Individual Knowledge Keeper step for temporal development and learning integration."""
        # Update learning cycles
        self._update_learning_cycles()

        # Process accumulated wisdom
        self._process_accumulated_wisdom()

        # Update neural networks
        self._update_neural_networks()

        # Evolve understanding based on accumulated learning
        self._evolve_understanding()

    def _update_learning_cycles(self):
        """Update different temporal learning cycles."""
        current_time = time.time()

        # Daily integration cycle (every 24 hours in simulation time)
        if len(self.learning_cycles['daily_integration']) == 0 or \
           current_time - self.learning_cycles['daily_integration'][-1].get('timestamp', 0) > 86400:

            daily_integration = {
                'timestamp': current_time,
                'insights_integrated': len(self.growth_patterns),
                'wisdom_moments': len(self.personal_development_wisdom),
                'character_developments': len(self.character_formation_insights)
            }
            self.learning_cycles['daily_integration'].append(daily_integration)

        # Keep only recent cycles for memory efficiency
        for cycle_type in self.learning_cycles:
            if len(self.learning_cycles[cycle_type]) > 100:
                self.learning_cycles[cycle_type] = self.learning_cycles[cycle_type][-100:]

    def _update_individual_wisdom_tracking(self, growth_insights: Dict):
        """Update individual wisdom tracking metrics."""
        if 'flourishing_discoveries' in growth_insights:
            self.wisdom['growth_patterns_learned'] += len(growth_insights['flourishing_discoveries'])

        if 'development_patterns' in growth_insights:
            self.wisdom['personal_development_wisdom'] += len(growth_insights['development_patterns'])

        if 'character_emergences' in growth_insights:
            self.wisdom['character_formation_insights'] += len(growth_insights['character_emergences'])

        if 'wisdom_moments' in growth_insights:
            self.wisdom['personal_development_wisdom'] += len(growth_insights['wisdom_moments'])

        # Update journey tracking
        unique_beings = set()
        for discovery in growth_insights.get('flourishing_discoveries', []):
            if isinstance(discovery, dict) and 'being_id' in discovery:
                unique_beings.add(discovery['being_id'])

        self.wisdom['individual_journeys_tracked'] = len(unique_beings)

        # Update amplifiers
        self.wisdom['compassion_amplifier'] = self.compassion_amplifier
        self.wisdom['curiosity_level'] = self.curiosity_level
        self.wisdom['wisdom_integration_cycles'] += 1


    def _process_accumulated_wisdom(self):
        """Process and integrate accumulated wisdom from being interactions."""
        # Look for patterns across accumulated wisdom
        if len(self.personal_development_wisdom) > 10:
            # Identify meta-patterns
            common_themes = {}
            for wisdom_key, wisdom_data in self.personal_development_wisdom.items():
                if 'theme' in wisdom_data:
                    theme = wisdom_data['theme']
                    common_themes[theme] = common_themes.get(theme, 0) + 1

            # Integrate common themes into deeper understanding
            for theme, frequency in common_themes.items():
                if frequency > 3:  # Theme appears frequently
                    meta_wisdom_key = f"meta_wisdom_{theme}"
                    if meta_wisdom_key not in self.personal_development_wisdom:
                        self.personal_development_wisdom[meta_wisdom_key] = {
                            'type': 'meta_wisdom',
                            'frequency': frequency,
                            'integration_level': min(1.0, frequency / 10.0)
                        }

    def _update_neural_networks(self):
        """Update both neural networks with accumulated learning."""
        if hasattr(self.individual_prefrontal_cortex, 'neural_network'):
            # Update with growth pattern learning
            growth_experience = {
                'patterns_recognized': len(self.growth_patterns),
                'wisdom_accumulated': len(self.personal_development_wisdom)
            }
            self.individual_prefrontal_cortex['neural_network'].step(growth_experience)

        if hasattr(self.individual_limbic_system, 'neural_network'):
            # Update with character understanding
            character_experience = {
                'character_insights': len(self.character_formation_insights),
                'compassion_level': self.compassion_amplifier
            }
            self.individual_limbic_system['neural_network'].step(character_experience)

    def _evolve_understanding(self):
        """Allow Knowledge Keeper understanding to evolve over time."""
        # Increase wisdom through accumulated learning
        if len(self.personal_development_wisdom) > 20:
            self.compassion_amplifier = min(2.0, self.compassion_amplifier + 0.01)

        # Deepen curiosity through continued learning
        if len(self.individual_journeys) > 5:
            self.curiosity_level = min(1.0, self.curiosity_level + 0.01)

        # Develop new understanding patterns
        if len(self.learning_history) > 50:
            # Create synthesis of accumulated learning
            synthesis_key = f"learning_synthesis_{len(self.learning_history)}"
            if synthesis_key not in self.collaborative_insights:
                self.collaborative_insights[synthesis_key] = {
                    'type': 'learning_synthesis',
                    'wisdom_depth': len(self.personal_development_wisdom),
                    'journey_understanding': len(self.individual_journeys),
                    'integration_timestamp': time.time()
                }

    def get_individual_wisdom_status(self) -> Dict:
        """Get current status of Individual Knowledge Keeper wisdom and learning."""
        return {
            'growth_patterns_learned': len(self.growth_patterns),
            'personal_development_wisdom': len(self.personal_development_wisdom),
            'character_formation_insights': len(self.character_formation_insights),
            'individual_journeys_tracked': len(self.individual_journeys),
            'life_stage_understanding': len(self.life_stage_understanding),
            'compassion_amplifier': self.compassion_amplifier,
            'curiosity_level': self.curiosity_level,
            'ollama_integration': self.ollama_available,
            'neural_network_status': {
                'prefrontal_cortex_active': 'neural_network' in self.individual_prefrontal_cortex and self.individual_prefrontal_cortex['neural_network'] is not None,
                'limbic_system_active': 'neural_network' in self.individual_limbic_system and self.individual_limbic_system['neural_network'] is not None
            },
            'learning_cycles': {
                'daily_integrations': len(self.learning_cycles['daily_integration']),
                'weekly_reflections': len(self.learning_cycles['weekly_reflection']),
                'monthly_wisdom': len(self.learning_cycles['monthly_wisdom']),
                'seasonal_growth': len(self.learning_cycles['seasonal_growth'])
            }
        }

    def ask_authentic_curiosity_questions(self, beings: List) -> List[str]:
        """Generate authentic curiosity questions for beings."""
        questions = []

        for being in beings:
            # Generate personal development questions
            if hasattr(being, 'curiosity') and being.curiosity > 0.8:
                questions.append(f"Your curiosity fascinates me - what questions are you exploring within yourself?")

            if hasattr(being, 'energy') and being.energy < 50:
                questions.append(f"I sense you might be processing something important - what's emerging in your inner world?")

            # Questions about growth and development
            questions.append(f"What aspects of yourself are you most curious about developing?")

        return questions

    def get_wisdom_for_collaboration(self) -> Dict:
        """Get wisdom insights for cross-system collaboration."""
        return {
            'growth_patterns': self.wisdom.get('growth_patterns_learned', 0),
            'personal_development_insights': self.wisdom.get('personal_development_wisdom', 0),
            'character_formation_understanding': self.wisdom.get('character_formation_insights', 0),
            'individual_journey_wisdom': self.wisdom.get('individual_journeys_tracked', 0),
            'temporal_development_patterns': self._get_temporal_development_patterns()
        }

    def integrate_social_insights(self, social_insights: Dict):
        """Integrate insights from Social Knowledge Keeper."""
        if 'relationship_discoveries' in social_insights:
            self.wisdom['social_context_growth'] = (
                "Individual development is enhanced when embedded in supportive relationships"
            )

        if 'trust_formation_observations' in social_insights:
            self.wisdom['trust_character_connection'] = (
                "Trustworthiness and character development strengthen each other"
            )

    def _get_temporal_development_patterns(self) -> Dict:
        """Get temporal patterns in individual development."""
        return {
            'personal_growth_cycles': 'Individual development follows natural rhythms and seasons',
            'character_formation_timeline': 'Character develops through consistent choices over extended periods',
            'wisdom_accumulation_pattern': 'Personal wisdom compounds through reflection and experience'
        }


class KnowledgeKeeperManager:
    """
    Manages the symbiotic relationship between Social and Individual 
    Knowledge Keepers, facilitating collaborative learning and wisdom emergence.
    """

    def __init__(self, ollama_base_url: str = "http://localhost:11434"):
        self.social_knowledge_keeper = SocialKnowledgeKeeper(ollama_base_url)
        self.individual_knowledge_keeper = IndividualKnowledgeKeeper(ollama_base_url)
        self.collaboration = KnowledgeKeeperCollaboration()
        self.collaborative_wisdom = {}
        self.last_update = None
        self.update_frequency = 180  # 3 minutes between updates

    def update_knowledge_keepers(self, entities: List) -> Dict:
        """
        Update both Social and Individual Knowledge Keepers based on entity interactions
        and internal development cycles.
        """
        if self.last_update is None or (time.time() - self.last_update) > self.update_frequency:
            self.last_update = time.time()

            # Gather data for Social Knowledge Keeper
            social_data = self.social_knowledge_keeper.symbiotic_learning(entities)
            relationship_context = social_data # For now, use social data as context

            # Gather data for Individual Knowledge Keeper
            individual_data = self.individual_knowledge_keeper.growth_pattern_learning(entities)

            # Update Social Knowledge Keeper (e.g., process patterns, provide guidance)
            social_result = self.social_knowledge_keeper.process_social_cycle(entities, relationship_context)

            # Update Individual Knowledge Keeper
            individual_result = self.individual_knowledge_keeper.process_development_cycle(
                individual_data, 
                relationship_context
            )

            # Process Knowledge Keeper collaboration if it's time
            collaboration_result = {}
            if self.collaboration.should_collaborate_now():
                collaboration_result = self.collaboration.process_collaboration_cycle(
                    {
                        'wisdom': self.social_knowledge_keeper.wisdom,
                        'relationship_data': social_data
                    },
                    {
                        'wisdom': self.individual_knowledge_keeper.wisdom,
                        'development_data': individual_data
                    }
                )

                # Apply collaborative insights to both Knowledge Keepers
                if collaboration_result:
                    self._apply_collaborative_insights(collaboration_result)
                    self._store_collaborative_discoveries(collaboration_result)

            return {
                'social_learning': social_result,
                'individual_learning': individual_result,
                'collaborative_wisdom': collaboration_result,
                'timestamp': self.last_update.isoformat()
            }
        else:
            return {'status': 'Update skipped, frequency not met.'}

    def _apply_collaborative_insights(self, collaboration_result: Dict):
        """Apply collaborative insights to enhance both Knowledge Keepers."""
        if 'wisdom_integration' in collaboration_result:
            # Enhance social keeper with individual insights
            self.social_knowledge_keeper.wisdom.update({
                'individual_development_insights': collaboration_result['wisdom_integration']['synthesis'].get('holistic_insights', [])
            })

            # Enhance individual keeper with social insights  
            self.individual_knowledge_keeper.wisdom.update({
                'social_dynamics_insights': collaboration_result['wisdom_integration']['synthesis'].get('holistic_insights', [])
            })

        # Apply cross-system pattern learning
        if 'cross_system_patterns' in collaboration_result:
            patterns = collaboration_result['cross_system_patterns']

            # Social keeper learns about individual impact on relationships
            self.social_knowledge_keeper.wisdom['individual_relationship_impact'] = patterns.get('individual_to_social', {})

            # Individual keeper learns about social impact on development
            self.individual_knowledge_keeper.wisdom['social_development_impact'] = patterns.get('social_to_individual', {})

    def _store_collaborative_discoveries(self, collaboration_result: Dict):
        """Store emergent collaborative wisdom discoveries."""
        if 'wisdom_synthesis' in collaboration_result:
            synthesis = collaboration_result['wisdom_synthesis']
            if 'unified_wisdom' in synthesis:
                unified = synthesis['unified_wisdom']

                # Store key collaborative discoveries
                self.collaborative_wisdom['social_individual_synergy'] = (
                    self.collaborative_wisdom.get('social_individual_synergy', 0) + 1
                )

                self.collaborative_wisdom['holistic_being_understanding'] = (
                    self.collaborative_wisdom.get('holistic_being_understanding', 0) + 1
                )

                self.collaborative_wisdom['community_wisdom_emergence'] = (
                    self.collaborative_wisdom.get('community_wisdom_emergence', 0) + 1
                )

    def get_knowledge_keeper_status(self) -> Dict:
        """Get current status of both Knowledge Keepers and their collaboration."""
        return {
            'social_keeper': self.social_knowledge_keeper.get_status(),
            'individual_keeper': self.individual_knowledge_keeper.get_status(),
            'collaboration': self.collaboration.get_collaboration_summary(),
            'last_update': self.last_update.isoformat() if self.last_update else None
        }