"""
Flask-based UI for Neural Ecosystem Simulation
Real-time dashboard with analogue-style displays
"""

from flask import Flask, render_template, jsonify, request
from flask_socketio import SocketIO, emit
import threading
import time
import json
import sys
import os

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import our ecosystem components
try:
    from population_analysis import analyze_ecosystem_population
    from temporal_development import TemporalDevelopmentTracker
    print("✅ Successfully imported ecosystem components")
except ImportError as e:
    print(f"⚠️ Import warning: {e}")

app = Flask(__name__)
app.config['SECRET_KEY'] = 'neural_ecosystem_secret'
socketio = SocketIO(app, async_mode='eventlet', cors_allowed_origins="*")

# Global variables
simulation_data = {
    'step_count': 0,
    'beings': [],
    'system_vitality': 85.0,
    'emergence_score': 60.0,
    'community_wisdom': 25.0,
    'current_stage': 'mature_community',
    'being_count': 3,
    'average_energy': 85.0,
    'relationships': [],
    'events': []
}

knowledge_keeper_data = {
    'social_keeper': {
        'relationship_discoveries': 15,
        'trust_observations': 8,
        'empathy_emergences': 5,
        'active': True
    },
    'individual_keeper': {
        'growth_patterns_learned': 12,
        'development_insights': 20,
        'journeys_tracked': 3,
        'compassion_level': 1.2,
        'curiosity_level': 0.8,
        'active': True
    },
    'collaboration': {
        'total_collaborations': 45,
        'development_phase': 'wisdom_integration',
        'collaboration_effectiveness': 0.85
    }
}

def update_simulation_data():
    """Update simulation data with real values if available."""
    global simulation_data

    # Update step count and basic metrics
    simulation_data['step_count'] += 1

    # Simulate dynamic values
    import random
    simulation_data['system_vitality'] = max(70, min(100, simulation_data['system_vitality'] + random.uniform(-2, 2)))
    simulation_data['emergence_score'] = max(40, min(100, simulation_data['emergence_score'] + random.uniform(-1, 1)))
    simulation_data['community_wisdom'] += random.uniform(0, 0.5)

    # Create sample beings data
    simulation_data['beings'] = [
        {
            'unique_id': i + 1,
            'energy': random.uniform(70, 100),
            'accumulated_wisdom': random.uniform(0, 50),
            'total_experience': random.uniform(50, 200),
            'current_growth_stage': random.choice(['emerging_awareness', 'active_exploration', 'wisdom_integration', 'mature_wisdom_sharing']),
            'neurochemical_state': {
                'empathy': random.uniform(0.6, 1.0),
                'curiosity': random.uniform(0.5, 1.0),
                'contentment': random.uniform(0.4, 0.9),
                'courage': random.uniform(0.3, 0.8),
                'stress': random.uniform(0.0, 0.3),
                'loneliness': random.uniform(0.0, 0.2),
                'compassion_amplifier': random.uniform(1.0, 1.5),
                'wisdom_integrator': random.uniform(1.0, 1.3)
            },
            'social_connections': random.randint(1, 5),
            'position': [random.randint(0, 6), random.randint(0, 6)],
            'recent_activities': [
                {'type': 'explored_environment', 'impact': 0.7},
                {'type': 'rested_peacefully', 'impact': 0.5}
            ],
            'recent_interactions': [
                {'type': 'supportive_care', 'partner': (i % 2) + 1, 'outcome': 0.8}
            ]
        } for i in range(3)
    ]

    # Update events
    event_types = [
        "Being discovered new growth pattern",
        "Community milestone achieved",
        "Knowledge Keepers collaborated",
        "Wisdom emergence detected",
        "Empathy resonance observed"
    ]

    if random.random() < 0.3:  # 30% chance of new event
        simulation_data['events'].append({
            'timestamp': simulation_data['step_count'],
            'event': f"Step {simulation_data['step_count']}: {random.choice(event_types)}"
        })

    # Keep only recent events
    simulation_data['events'] = simulation_data['events'][-20:]

# Simulation thread
def simulation_loop():
    """Continuous simulation updates."""
    while True:
        try:
            update_simulation_data()
            socketio.emit('update', {'type': 'full_update'})
            time.sleep(3)  # 3-second updates
        except Exception as e:
            print(f"Simulation loop error: {e}")
            time.sleep(5)

# Start simulation thread
threading.Thread(target=simulation_loop, daemon=True).start()

# Routes
@app.route('/')
def index():
    """Main dashboard page."""
    return render_template('dashboard.html')

@app.route('/api/ecosystem_state')
def get_ecosystem_state():
    """Get current ecosystem state."""
    try:
        # Create grid representation
        grid_data = [[None for _ in range(7)] for _ in range(7)]
        for being in simulation_data['beings']:
            if being['position']:
                x, y = being['position']
                if 0 <= x < 7 and 0 <= y < 7:
                    grid_data[x][y] = {
                        'id': being['unique_id'],
                        'energy': being['energy'],
                        'wisdom': being['accumulated_wisdom']
                    }

        return jsonify({
            'step_count': simulation_data['step_count'],
            'system_vitality': simulation_data['system_vitality'],
            'emergence_score': simulation_data['emergence_score'],
            'being_count': simulation_data['being_count'],
            'average_energy': simulation_data['average_energy'],
            'community_wisdom': simulation_data['community_wisdom'],
            'grid': grid_data,
            'beings': simulation_data['beings'],
            'current_stage': simulation_data['current_stage']
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/population_analysis')
def get_population_analysis():
    """Get population analysis data."""
    return jsonify({
        'total_population': len(simulation_data['beings']),
        'population_statistics': {
            'average_energy': sum(b['energy'] for b in simulation_data['beings']) / len(simulation_data['beings']) if simulation_data['beings'] else 0,
            'total_wisdom': sum(b['accumulated_wisdom'] for b in simulation_data['beings']),
            'beings_flourishing': sum(1 for b in simulation_data['beings'] if b['energy'] > 80),
            'beings_struggling': sum(1 for b in simulation_data['beings'] if b['energy'] < 50)
        },
        'development_stages': {
            stage: sum(1 for b in simulation_data['beings'] if b['current_growth_stage'] == stage)
            for stage in ['emerging_awareness', 'active_exploration', 'wisdom_integration', 'mature_wisdom_sharing']
        },
        'energy_distribution': {
            'very_low': sum(1 for b in simulation_data['beings'] if b['energy'] < 20),
            'low': sum(1 for b in simulation_data['beings'] if 20 <= b['energy'] < 40),
            'medium': sum(1 for b in simulation_data['beings'] if 40 <= b['energy'] < 60),
            'high': sum(1 for b in simulation_data['beings'] if 60 <= b['energy'] < 80),
            'very_high': sum(1 for b in simulation_data['beings'] if b['energy'] >= 80)
        }
    })

@app.route('/api/knowledge_keeper_status')
def get_knowledge_keeper_status():
    """Get Knowledge Keeper status."""
    return jsonify(knowledge_keeper_data)

@app.route('/api/temporal_development')
def get_temporal_development():
    """Get temporal development data."""
    return jsonify({
        'current_stage': simulation_data['current_stage'],
        'daily_patterns': [{'step': i, 'metrics': {'average_energy': 75 + i % 20}} for i in range(10)],
        'weekly_patterns': [{'week': i, 'metrics': {'average_energy': 70 + i * 2}} for i in range(4)],
        'stage_transitions': [
            {'from': 'forming_community', 'to': 'early_exploration', 'step': 50},
            {'from': 'early_exploration', 'to': 'active_development', 'step': 150},
            {'from': 'active_development', 'to': 'mature_community', 'step': 300}
        ],
        'milestones': [
            {'type': 'community_formation', 'description': 'First stable relationships formed', 'step': 25},
            {'type': 'wisdom_emergence', 'description': 'Collective wisdom patterns emerged', 'step': 180},
            {'type': 'mature_collaboration', 'description': 'Advanced being-Knowledge Keeper synergy', 'step': 350}
        ]
    })

@app.route('/api/relationships')
def get_relationships():
    """Get relationship data."""
    relationships = []
    for being in simulation_data['beings']:
        for interaction in being['recent_interactions']:
            relationships.append({
                'from': being['unique_id'],
                'to': interaction.get('partner', 1),
                'type': interaction['type'],
                'strength': interaction['outcome'],
                'timestamp': simulation_data['step_count']
            })
    return jsonify(relationships)

@app.route('/api/live_events')
def get_live_events():
    """Get live events."""
    events = [event['event'] for event in simulation_data['events']]
    return jsonify(events)

@app.route('/api/being/<int:being_id>')
def get_being(being_id):
    """Get detailed being information."""
    for being in simulation_data['beings']:
        if being['unique_id'] == being_id:
            return jsonify(being)
    return jsonify({'error': 'Being not found'}), 404

@app.route('/api/query', methods=['POST'])
def query():
    """Handle Knowledge Keeper queries."""
    try:
        data = request.json
        query_text = data.get('query', '')

        # Generate Knowledge Keeper response
        responses = [
            f"Individual Knowledge Keeper: Observing growth patterns in response to '{query_text}'",
            f"Social Knowledge Keeper: Community dynamics show {len(simulation_data['beings'])} beings flourishing",
            f"Collaborative Insight: Wisdom emerges through authentic being interactions"
        ]

        response = " | ".join(responses)
        return jsonify({'response': response})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# WebSocket events
@socketio.on('connect')
def handle_connect():
    """Handle client connection."""
    print('Client connected to Neural Ecosystem UI')
    emit('update', {'type': 'initial'})

@socketio.on('disconnect')
def handle_disconnect():
    """Handle client disconnection."""
    print('Client disconnected from Neural Ecosystem UI')

if __name__ == '__main__':
    print("🌟 Starting Neural Ecosystem UI Dashboard...")
    print("🔗 Access at: http://0.0.0.0:5000")
    socketio.run(app, host='0.0.0.0', port=5000, debug=False)