
"""
Population Analysis for the Neural Ecosystem.
Provides detailed insights into the beings and their development.
"""

def analyze_ecosystem_population(model):
    """
    Analyze the current population of the Neural Ecosystem.
    
    Args:
        model: The NeuralEcosystem model instance
        
    Returns:
        Dict containing comprehensive population analysis
    """
    if not model.agents:
        return {
            'total_population': 0,
            'message': 'No beings currently exist in the ecosystem'
        }
    
    analysis = {
        'total_population': len(model.agents),
        'beings_overview': [],
        'population_statistics': {},
        'development_stages': {},
        'energy_distribution': {},
        'social_dynamics': {},
        'wisdom_patterns': {},
        'neurochemical_profile': {},
        'knowledge_keeper_insights': {}
    }
    
    # Analyze each being
    energies = []
    wisdom_levels = []
    social_connections = []
    growth_stages = {}
    
    for being in model.agents:
        being_info = {
            'id': being.unique_id,
            'energy': getattr(being, 'energy', 0),
            'wisdom': getattr(being, 'accumulated_wisdom', 0),
            'experience': getattr(being, 'total_experience', 0),
            'growth_stage': getattr(being, 'current_growth_stage', 'unknown'),
            'social_connections': getattr(being, 'social_connections', 0),
            'position': being.pos if being.pos else 'unknown'
        }
        
        # Neurochemical profile
        if hasattr(being, 'neurochemical_system'):
            neuro = being.neurochemical_system
            being_info['neurochemicals'] = {
                'empathy': getattr(neuro, 'empathy', 0.5),
                'curiosity': getattr(neuro, 'curiosity', 0.5),
                'contentment': getattr(neuro, 'contentment', 0.5),
                'courage': getattr(neuro, 'courage', 0.5),
                'stress': getattr(neuro, 'stress', 0.3),
                'loneliness': getattr(neuro, 'loneliness', 0.3),
                'compassion_amplifier': getattr(neuro, 'compassion_amplifier', 1.0),
                'wisdom_integrator': getattr(neuro, 'wisdom_integrator', 1.0)
            }
        
        analysis['beings_overview'].append(being_info)
        
        # Collect statistics
        energies.append(being_info['energy'])
        wisdom_levels.append(being_info['wisdom'])
        social_connections.append(being_info['social_connections'])
        
        stage = being_info['growth_stage']
        growth_stages[stage] = growth_stages.get(stage, 0) + 1
    
    # Population statistics
    import numpy as np
    analysis['population_statistics'] = {
        'average_energy': np.mean(energies),
        'energy_range': [min(energies), max(energies)],
        'total_wisdom': sum(wisdom_levels),
        'average_wisdom': np.mean(wisdom_levels),
        'total_social_connections': sum(social_connections),
        'average_social_connections': np.mean(social_connections),
        'beings_flourishing': sum(1 for e in energies if e > 80),
        'beings_struggling': sum(1 for e in energies if e < 40)
    }
    
    # Development stages
    analysis['development_stages'] = growth_stages
    
    # Energy distribution
    energy_ranges = {
        'very_high (90-100)': sum(1 for e in energies if e >= 90),
        'high (75-89)': sum(1 for e in energies if 75 <= e < 90),
        'moderate (50-74)': sum(1 for e in energies if 50 <= e < 75),
        'low (25-49)': sum(1 for e in energies if 25 <= e < 50),
        'very_low (0-24)': sum(1 for e in energies if e < 25)
    }
    analysis['energy_distribution'] = energy_ranges
    
    # Social dynamics analysis
    connection_levels = {
        'highly_connected (3+)': sum(1 for c in social_connections if c >= 3),
        'moderately_connected (1-2)': sum(1 for c in social_connections if 1 <= c <= 2),
        'isolated (0)': sum(1 for c in social_connections if c == 0)
    }
    analysis['social_dynamics'] = connection_levels
    
    # Neurochemical profile analysis
    if analysis['beings_overview'] and 'neurochemicals' in analysis['beings_overview'][0]:
        neuro_stats = {}
        chemicals = ['empathy', 'curiosity', 'contentment', 'courage', 'stress', 'loneliness']
        
        for chemical in chemicals:
            values = []
            for being_info in analysis['beings_overview']:
                if 'neurochemicals' in being_info:
                    values.append(being_info['neurochemicals'].get(chemical, 0.5))
            
            if values:
                neuro_stats[chemical] = {
                    'average': np.mean(values),
                    'range': [min(values), max(values)],
                    'high_count': sum(1 for v in values if v > 0.7),
                    'low_count': sum(1 for v in values if v < 0.3)
                }
        
        analysis['neurochemical_profile'] = neuro_stats
    
    # Knowledge Keeper insights
    if hasattr(model, 'social_knowledge_keeper') and hasattr(model, 'individual_knowledge_keeper'):
        social_wisdom = getattr(model.social_knowledge_keeper, 'wisdom', {})
        individual_wisdom = getattr(model.individual_knowledge_keeper, 'wisdom', {})
        
        analysis['knowledge_keeper_insights'] = {
            'social_keeper': {
                'relationship_discoveries': social_wisdom.get('relationship_discoveries', 0),
                'trust_observations': social_wisdom.get('trust_formation_observations', 0),
                'empathy_emergences': social_wisdom.get('empathy_emergences', 0)
            },
            'individual_keeper': {
                'growth_patterns_learned': individual_wisdom.get('growth_patterns_learned', 0),
                'development_insights': individual_wisdom.get('personal_development_wisdom', 0),
                'journeys_tracked': individual_wisdom.get('individual_journeys_tracked', 0),
                'compassion_level': individual_wisdom.get('compassion_amplifier', 1.0),
                'curiosity_level': individual_wisdom.get('curiosity_level', 0.8)
            }
        }
    
    return analysis

def print_population_report(analysis):
    """Print a human-readable population report."""
    print("\n🌟 NEURAL ECOSYSTEM POPULATION REPORT")
    print("=" * 50)
    
    if analysis['total_population'] == 0:
        print("🚫 No beings currently exist in the ecosystem")
        return
    
    # Basic stats
    stats = analysis['population_statistics']
    print(f"\n👥 POPULATION OVERVIEW:")
    print(f"   Total beings: {analysis['total_population']}")
    print(f"   Average energy: {stats['average_energy']:.1f}")
    print(f"   Energy range: {stats['energy_range'][0]:.1f} - {stats['energy_range'][1]:.1f}")
    print(f"   Beings flourishing (>80 energy): {stats['beings_flourishing']}")
    print(f"   Beings struggling (<40 energy): {stats['beings_struggling']}")
    
    # Wisdom stats
    print(f"\n🧠 WISDOM & DEVELOPMENT:")
    print(f"   Total community wisdom: {stats['total_wisdom']:.2f}")
    print(f"   Average wisdom per being: {stats['average_wisdom']:.2f}")
    print(f"   Total social connections: {stats['total_social_connections']}")
    
    # Development stages
    if analysis['development_stages']:
        print(f"\n📈 DEVELOPMENT STAGES:")
        for stage, count in analysis['development_stages'].items():
            print(f"   {stage}: {count} beings")
    
    # Energy distribution
    print(f"\n⚡ ENERGY DISTRIBUTION:")
    for range_name, count in analysis['energy_distribution'].items():
        if count > 0:
            print(f"   {range_name}: {count} beings")
    
    # Social dynamics
    print(f"\n🤝 SOCIAL CONNECTIONS:")
    for level, count in analysis['social_dynamics'].items():
        if count > 0:
            print(f"   {level}: {count} beings")
    
    # Neurochemical highlights
    if analysis['neurochemical_profile']:
        print(f"\n🧪 NEUROCHEMICAL HIGHLIGHTS:")
        neuro = analysis['neurochemical_profile']
        for chemical, data in neuro.items():
            if data['high_count'] > 0:
                print(f"   High {chemical}: {data['high_count']} beings (avg: {data['average']:.2f})")
    
    # Knowledge Keeper insights
    if analysis['knowledge_keeper_insights']:
        print(f"\n🎓 KNOWLEDGE KEEPER INSIGHTS:")
        social = analysis['knowledge_keeper_insights']['social_keeper']
        individual = analysis['knowledge_keeper_insights']['individual_keeper']
        
        print(f"   Social discoveries: {social['relationship_discoveries']} relationships")
        print(f"   Individual growth patterns: {individual['growth_patterns_learned']}")
        print(f"   Compassion level: {individual['compassion_level']:.2f}")
        print(f"   Community curiosity: {individual['curiosity_level']:.2f}")
    
    # Individual being details
    print(f"\n👤 INDIVIDUAL BEINGS:")
    for being in analysis['beings_overview']:
        energy_status = "🌟" if being['energy'] > 80 else "⚡" if being['energy'] > 50 else "💤"
        wisdom_status = "🧠" if being['wisdom'] > 2 else "🌱" if being['wisdom'] > 0.5 else "🌿"
        
        print(f"   Being {being['id']}: {energy_status} Energy:{being['energy']:.1f} "
              f"{wisdom_status} Wisdom:{being['wisdom']:.2f} Stage:{being['growth_stage']}")
        
        if 'neurochemicals' in being:
            neuro = being['neurochemicals']
            print(f"      💙 Empathy:{neuro['empathy']:.2f} 🔍 Curiosity:{neuro['curiosity']:.2f} "
                  f"☮️ Peace:{neuro['contentment']:.2f} 💪 Courage:{neuro['courage']:.2f}")

if __name__ == "__main__":
    print("Population analysis module ready!")
