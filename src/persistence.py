"""
Persistence utilities for saving and loading user profiles and feedback.

Provides JSON-based serialization/deserialization of:
- User preferences (genres, moods, energy, etc.)
- Chat history
- Feedback logs
"""

import json
import os
from datetime import datetime
from typing import Dict, List, Optional
from src.recommender import UserProfile


def save_user_profile(
    user_id: str,
    user_profile: UserProfile,
    feedback_log: List[Dict],
    session_history: List[Dict],
    output_dir: str = 'data/profiles'
) -> str:
    """
    Save a user profile to JSON file.
    
    Args:
        user_id: User identifier (used in filename)
        user_profile: UserProfile instance
        feedback_log: List of feedback entries
        session_history: List of session history entries
        output_dir: Directory to save profiles
    
    Returns:
        Path to saved file
    """
    os.makedirs(output_dir, exist_ok=True)
    
    user_id_safe = user_id.replace('/', '_').replace('\\', '_')
    filepath = os.path.join(output_dir, f'{user_id_safe}_profile.json')
    
    profile_data = {
        "user_id": user_id,
        "timestamp": datetime.now().isoformat(),
        "profile": {
            "favorite_genres": user_profile.favorite_genres,
            "favorite_moods": user_profile.favorite_moods,
            "target_energy": user_profile.target_energy,
            "target_valence": user_profile.target_valence,
            "target_tempo_bpm": user_profile.target_tempo_bpm,
            "target_danceability": user_profile.target_danceability,
        },
        "feedback_log": feedback_log,
        "session_history": session_history[-20:],  # Keep last 20 sessions
    }
    
    with open(filepath, 'w') as f:
        json.dump(profile_data, f, indent=2)
    
    return filepath


def load_user_profile(
    user_id: str,
    input_dir: str = 'data/profiles'
) -> Optional[Dict]:
    """
    Load a user profile from JSON file.
    
    Args:
        user_id: User identifier
        input_dir: Directory containing profiles
    
    Returns:
        Profile data dict or None if not found
    """
    user_id_safe = user_id.replace('/', '_').replace('\\', '_')
    filepath = os.path.join(input_dir, f'{user_id_safe}_profile.json')
    
    if not os.path.exists(filepath):
        return None
    
    with open(filepath, 'r') as f:
        profile_data = json.load(f)
    
    return profile_data


def list_user_profiles(input_dir: str = 'data/profiles') -> List[str]:
    """
    List all saved user profile IDs.
    
    Args:
        input_dir: Directory containing profiles
    
    Returns:
        List of user IDs
    """
    if not os.path.exists(input_dir):
        return []
    
    profiles = []
    for filename in os.listdir(input_dir):
        if filename.endswith('_profile.json'):
            user_id = filename.replace('_profile.json', '')
            profiles.append(user_id)
    
    return sorted(profiles)


def save_feedback_log(
    user_id: str,
    feedback_log: List[Dict],
    output_dir: str = 'data/feedback'
) -> str:
    """
    Save feedback log to JSON file.
    
    Args:
        user_id: User identifier
        feedback_log: List of feedback entries
        output_dir: Directory to save logs
    
    Returns:
        Path to saved file
    """
    os.makedirs(output_dir, exist_ok=True)
    
    user_id_safe = user_id.replace('/', '_').replace('\\', '_')
    filepath = os.path.join(output_dir, f'{user_id_safe}_feedback.json')
    
    feedback_data = {
        "user_id": user_id,
        "timestamp": datetime.now().isoformat(),
        "entries": feedback_log,
    }
    
    with open(filepath, 'w') as f:
        json.dump(feedback_data, f, indent=2)
    
    return filepath


def load_feedback_log(
    user_id: str,
    input_dir: str = 'data/feedback'
) -> Optional[List[Dict]]:
    """
    Load feedback log from JSON file.
    
    Args:
        user_id: User identifier
        input_dir: Directory containing logs
    
    Returns:
        List of feedback entries or None if not found
    """
    user_id_safe = user_id.replace('/', '_').replace('\\', '_')
    filepath = os.path.join(input_dir, f'{user_id_safe}_feedback.json')
    
    if not os.path.exists(filepath):
        return None
    
    with open(filepath, 'r') as f:
        feedback_data = json.load(f)
    
    return feedback_data.get('entries', [])


def export_feedback_as_csv(
    user_id: str,
    feedback_log: List[Dict],
    output_dir: str = 'data/exports'
) -> str:
    """
    Export feedback log as CSV file.
    
    Args:
        user_id: User identifier
        feedback_log: List of feedback entries
        output_dir: Directory to save CSV
    
    Returns:
        Path to saved CSV file
    """
    os.makedirs(output_dir, exist_ok=True)
    
    user_id_safe = user_id.replace('/', '_').replace('\\', '_')
    filepath = os.path.join(output_dir, f'{user_id_safe}_feedback.csv')
    
    with open(filepath, 'w') as f:
        # Write CSV header
        f.write('timestamp,song_id,song_title,rating,query\n')
        
        # Write feedback entries
        for entry in feedback_log:
            timestamp = entry.get('timestamp', '')
            song_id = entry.get('song_id', '')
            song_title = entry.get('song_title', '').replace('"', '""')  # Escape quotes
            rating = entry.get('rating', '')
            query = entry.get('query', '').replace('"', '""')  # Escape quotes
            
            f.write(f'{timestamp},{song_id},"{song_title}",{rating},"{query}"\n')
    
    return filepath
