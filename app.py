"""
Flask web server for the RAG music recommender system.

Provides REST API endpoints for:
- /api/chat - Get music recommendations from natural language queries
- /api/feedback - Log user feedback (thumbs up/down) on recommendations
- /api/playlist - Generate playlists from current preferences
- /api/profile - Save and load user profiles
- / - Serve the main HTML UI
"""

import os
import json
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from datetime import datetime
import uuid
from dotenv import load_dotenv
from src.recommender import create_conversational_recommender

load_dotenv()

# Initialize Flask app
app = Flask(__name__, static_folder='static', static_url_path='')
CORS(app)

# Global state for user sessions
SESSIONS = {}  # session_id -> ConversationalRecommender instance
PROFILES_DIR = 'data/profiles'

# Ensure profiles directory exists
os.makedirs(PROFILES_DIR, exist_ok=True)

# Load the recommender system (cached globally)
RECOMMENDER = None

def get_recommender():
    """Load recommender system lazily (on first request)."""
    global RECOMMENDER
    if RECOMMENDER is None:
        print("Loading recommendation system...")
        csv_path = 'data/songs.csv'
        RECOMMENDER = create_conversational_recommender(
            csv_path,
            use_llm=True,
            llm_type='gemini'
        )
        print("✓ Recommendation system loaded")
    return RECOMMENDER

def get_or_create_session(session_id: str = None):
    """
    Get or create a user session with a ConversationalRecommender instance.
    
    Args:
        session_id: Existing session ID (optional)
    
    Returns:
        (session_id, ConversationalRecommender)
    """
    if session_id is None:
        session_id = str(uuid.uuid4())
    
    if session_id not in SESSIONS:
        # Create new session by copying the global recommender
        base_recommender = get_recommender()
        # Create a new instance for this session
        from src.recommender import ConversationalRecommender
        session_recommender = ConversationalRecommender(
            base_recommender.hybrid,
            llm=base_recommender.llm
        )
        SESSIONS[session_id] = session_recommender
    
    return session_id, SESSIONS[session_id]


# ============================================================================
# API Routes
# ============================================================================

@app.route('/', methods=['GET'])
def serve_index():
    """Serve main HTML UI."""
    return send_from_directory('static', 'index.html')

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    return jsonify({
        "status": "ok",
        "timestamp": datetime.now().isoformat()
    }), 200


@app.route('/api/chat', methods=['POST'])
def api_chat():
    """
    Get music recommendations from a natural language query.
    
    Request JSON:
    {
        "query": "I want chill lofi for studying",
        "session_id": "optional-session-uuid",
        "k": 5
    }
    
    Response JSON:
    {
        "session_id": "session-uuid",
        "recommendations": [...],
        "llm_explanation": "...",
        "user_profile": {...}
    }
    """
    try:
        data = request.get_json()
        query = data.get('query', '').strip()
        session_id = data.get('session_id')
        k = data.get('k', 5)
        
        if not query:
            return jsonify({"error": "Query is required"}), 400
        
        # Get or create session
        session_id, recommender = get_or_create_session(session_id)
        
        # Get recommendations
        response = recommender.chat(query, k=k)
        
        # Add session_id to response
        response['session_id'] = session_id
        
        return jsonify(response), 200
    
    except Exception as e:
        print(f"Error in /api/chat: {e}")
        return jsonify({"error": str(e)}), 500


@app.route('/api/feedback', methods=['POST'])
def api_feedback():
    """
    Log feedback (thumbs up/down) on a recommendation.
    
    Request JSON:
    {
        "session_id": "session-uuid",
        "song_id": 1,
        "song_title": "Song Title",
        "rating": 1,  # +1 for thumbs up, -1 for thumbs down
        "query": "original query that led to this recommendation"
    }
    
    Response JSON:
    {
        "success": true,
        "feedback": {...},
        "stats": {...}
    }
    """
    try:
        data = request.get_json()
        session_id = data.get('session_id')
        song_id = data.get('song_id')
        song_title = data.get('song_title', '')
        rating = data.get('rating', 0)
        query = data.get('query', '')
        
        if not session_id or song_id is None:
            return jsonify({"error": "session_id and song_id are required"}), 400
        
        if rating not in [-1, 1]:
            return jsonify({"error": "rating must be 1 (thumbs up) or -1 (thumbs down)"}), 400
        
        # Get session
        session_id, recommender = get_or_create_session(session_id)
        
        # Log feedback
        feedback = recommender.log_feedback(
            song_id=song_id,
            rating=rating,
            query=query,
            song_title=song_title
        )
        
        # Get stats
        stats = recommender.get_feedback_stats()
        
        return jsonify({
            "success": True,
            "feedback": feedback,
            "stats": stats
        }), 200
    
    except Exception as e:
        print(f"Error in /api/feedback: {e}")
        return jsonify({"error": str(e)}), 500


@app.route('/api/playlist', methods=['POST'])
def api_playlist():
    """
    Generate a playlist based on current user profile.
    
    Request JSON:
    {
        "session_id": "session-uuid",
        "duration_minutes": 30,
        "k": 10
    }
    
    Response JSON:
    {
        "playlist": {...},
        "session_id": "session-uuid"
    }
    """
    try:
        data = request.get_json()
        session_id = data.get('session_id')
        duration_minutes = data.get('duration_minutes', 30)
        k = data.get('k', 10)
        
        if not session_id:
            return jsonify({"error": "session_id is required"}), 400
        
        # Get session
        session_id, recommender = get_or_create_session(session_id)
        
        # Generate playlist
        playlist = recommender.generate_playlist(
            duration_minutes=duration_minutes,
            k=k
        )
        
        return jsonify({
            "playlist": playlist,
            "session_id": session_id
        }), 200
    
    except Exception as e:
        print(f"Error in /api/playlist: {e}")
        return jsonify({"error": str(e)}), 500


@app.route('/api/profile', methods=['GET', 'POST'])
def api_profile():
    """
    Get current user profile or save/load profiles.
    
    GET /api/profile?session_id=...
    Returns current user profile for the session
    
    POST /api/profile
    Request JSON:
    {
        "session_id": "session-uuid",
        "action": "save" or "load",
        "user_id": "username" (for save/load)
    }
    """
    try:
        if request.method == 'GET':
            # Get current profile for session
            session_id = request.args.get('session_id')
            if not session_id:
                return jsonify({"error": "session_id is required"}), 400
            
            session_id, recommender = get_or_create_session(session_id)
            profile = {
                "favorite_genres": recommender.user_profile.favorite_genres,
                "favorite_moods": recommender.user_profile.favorite_moods,
                "target_energy": recommender.user_profile.target_energy,
            }
            return jsonify({"profile": profile, "session_id": session_id}), 200
        
        elif request.method == 'POST':
            # Save or load profile
            data = request.get_json()
            action = data.get('action', 'save')
            session_id = data.get('session_id')
            user_id = data.get('user_id', 'default').replace('/', '_')  # Sanitize
            
            if not session_id:
                return jsonify({"error": "session_id is required"}), 400
            
            session_id, recommender = get_or_create_session(session_id)
            profile_path = os.path.join(PROFILES_DIR, f'{user_id}_profile.json')
            
            if action == 'save':
                # Save profile to file
                profile_data = {
                    "user_id": user_id,
                    "timestamp": datetime.now().isoformat(),
                    "profile": {
                        "favorite_genres": recommender.user_profile.favorite_genres,
                        "favorite_moods": recommender.user_profile.favorite_moods,
                        "target_energy": recommender.user_profile.target_energy,
                        "target_valence": recommender.user_profile.target_valence,
                        "target_tempo_bpm": recommender.user_profile.target_tempo_bpm,
                        "target_danceability": recommender.user_profile.target_danceability,
                    },
                    "feedback_log": recommender.feedback_log,
                    "session_history": recommender.session_history[-10:],  # Keep last 10 queries
                }
                with open(profile_path, 'w') as f:
                    json.dump(profile_data, f, indent=2)
                
                return jsonify({
                    "success": True,
                    "message": f"Profile saved for {user_id}",
                    "session_id": session_id
                }), 200
            
            elif action == 'load':
                # Load profile from file
                if not os.path.exists(profile_path):
                    return jsonify({
                        "success": False,
                        "message": f"No profile found for {user_id}",
                        "session_id": session_id
                    }), 404
                
                with open(profile_path, 'r') as f:
                    profile_data = json.load(f)
                
                # Restore profile to session
                prof = profile_data['profile']
                recommender.user_profile.favorite_genres = prof['favorite_genres']
                recommender.user_profile.favorite_moods = prof['favorite_moods']
                recommender.user_profile.target_energy = prof['target_energy']
                recommender.user_profile.target_valence = prof['target_valence']
                recommender.user_profile.target_tempo_bpm = prof['target_tempo_bpm']
                recommender.user_profile.target_danceability = prof['target_danceability']
                
                # Restore feedback log and session history
                recommender.feedback_log = profile_data.get('feedback_log', [])
                recommender.session_history = profile_data.get('session_history', [])
                
                return jsonify({
                    "success": True,
                    "message": f"Profile loaded for {user_id}",
                    "profile": prof,
                    "session_id": session_id
                }), 200
            
            else:
                return jsonify({"error": f"Unknown action: {action}"}), 400
    
    except Exception as e:
        print(f"Error in /api/profile: {e}")
        return jsonify({"error": str(e)}), 500


@app.route('/api/profiles', methods=['GET'])
def api_list_profiles():
    """
    List all available saved user profiles.
    
    Returns:
    {
        "profiles": ["user1", "user2", ...]
    }
    """
    try:
        profiles = []
        if os.path.exists(PROFILES_DIR):
            for filename in os.listdir(PROFILES_DIR):
                if filename.endswith('_profile.json'):
                    user_id = filename.replace('_profile.json', '')
                    profiles.append(user_id)
        
        return jsonify({"profiles": sorted(profiles)}), 200
    
    except Exception as e:
        print(f"Error in /api/profiles: {e}")
        return jsonify({"error": str(e)}), 500


@app.route('/api/export/<user_id>', methods=['GET'])
def api_export_feedback(user_id):
    """
    Export feedback log as CSV for a saved user profile.
    
    Returns CSV file with feedback entries
    """
    try:
        user_id = user_id.replace('/', '_')  # Sanitize
        profile_path = os.path.join(PROFILES_DIR, f'{user_id}_profile.json')
        
        if not os.path.exists(profile_path):
            return jsonify({"error": f"No profile found for {user_id}"}), 404
        
        with open(profile_path, 'r') as f:
            profile_data = json.load(f)
        
        # Generate CSV
        feedback_log = profile_data.get('feedback_log', [])
        if not feedback_log:
            return jsonify({"message": "No feedback to export"}), 200
        
        csv_lines = [
            "timestamp,song_id,song_title,rating,query"
        ]
        for fb in feedback_log:
            line = f"{fb['timestamp']},{fb['song_id']},\"{fb['song_title']}\",{fb['rating']},\"{fb['query']}\""
            csv_lines.append(line)
        
        csv_content = "\n".join(csv_lines)
        
        return csv_content, 200, {
            'Content-Type': 'text/csv',
            'Content-Disposition': f'attachment; filename=feedback_{user_id}.csv'
        }
    
    except Exception as e:
        print(f"Error in /api/export: {e}")
        return jsonify({"error": str(e)}), 500


@app.route('/api/session', methods=['GET'])
def api_session_summary():
    """
    Get session summary.
    
    Request: GET /api/session?session_id=...
    Returns: Session info and stats
    """
    try:
        session_id = request.args.get('session_id')
        if not session_id:
            return jsonify({"error": "session_id is required"}), 400
        
        session_id, recommender = get_or_create_session(session_id)
        summary = recommender.get_session_summary()
        
        return jsonify({
            "session_id": session_id,
            "summary": summary
        }), 200
    
    except Exception as e:
        print(f"Error in /api/session: {e}")
        return jsonify({"error": str(e)}), 500


# ============================================================================
# Main
# ============================================================================

if __name__ == '__main__':
    print("\n" + "="*70)
    print("🎵 RAG Music Recommendation System - Flask Server")
    print("="*70)
    print("\nAPI Endpoints:")
    print("  POST  /api/chat         - Get recommendations")
    print("  POST  /api/feedback     - Log feedback")
    print("  POST  /api/playlist     - Generate playlist")
    print("  GET   /api/profile      - Get current profile")
    print("  POST  /api/profile      - Save/load profile")
    print("  GET   /api/profiles     - List saved profiles")
    print("  GET   /api/export/<id>  - Export feedback")
    print("  GET   /api/session      - Session summary")
    print("  GET   /health           - Health check")
    print("  GET   /                 - Serve UI")
    print("\nStarting server on http://localhost:5000")
    print("="*70 + "\n")
    
    # Pre-load recommender on startup
    get_recommender()
    
    # Run Flask app
    app.run(debug=True, host='0.0.0.0', port=5001)
