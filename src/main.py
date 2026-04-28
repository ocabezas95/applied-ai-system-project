"""
Command line runner for the Music Recommender Simulation.

This file helps you quickly run and test your recommender.

You will implement the functions in recommender.py:
- load_songs
- score_song
- recommend_songs
"""

from src.recommender import load_songs, recommend_songs


def main() -> None:
    songs = load_songs("data/songs.csv") 

    # ========== TEST PROFILES ==========
    
    # Starter example profile
    user_prefs = {"genre": "pop", "mood": "happy", "energy": 0.8}

    # ADVERSARIAL PROFILE 1: "The Contradicted User"
    # Tests conflicting preferences: high energy but melancholic mood
    # EXPOSES: No logic to reconcile contradictory preferences.
    # - Metal/Rock songs are high-energy (0.91-0.95) but NOT melancholic
    # - Classical/Lofi songs are melancholic (0.30) but NOT high-energy
    # System must choose between energy match or mood match, forcing poor compromises.
    user_prefs_contradicted = {
        "genre": "metal",
        "mood": "melancholic",
        "energy": 0.95
    }

    # ADVERSARIAL PROFILE 2: "The Genre Ghost"
    # Tests non-existent genre that doesn't appear in dataset.
    # EXPOSES: Graceful degradation when genre_weight always scores 0.
    # - "techno" genre doesn't exist in the dataset
    # - "ambient" exists, but request non-existent mood "ethereal" too
    # System falls back to energy scoring only. Also reveals that target_tempo_bpm
    # is defined in UserProfile but NEVER used in score_song() function.
    user_prefs_ghost_genre = {
        "genre": "techno",
        "mood": "ethereal",
        "energy": 0.80,
        "tempo_bpm": 130  # High tempo preference—unused in scoring!
    }

    # TEST PROFILE 3: "Chill Lofi"
    # Tests relaxing lofi aesthetic with low energy and slow tempo.
    user_prefs_chill_lofi = {
        "genre": "lofi",
        "mood": "chill",
        "energy": 0.3,
        "tempo_bpm": 75
    }

    # TEST PROFILE 4: "Deep Intense Rock"
    # Tests high-energy rock music with intense mood and fast tempo.
    user_prefs_intense_rock = {
        "genre": "rock",
        "mood": "intense",
        "energy": 0.9,
        "tempo_bpm": 140
    }

    # ========== RUN RECOMMENDATIONS ==========
    print("\n" + "="*50)
    print("STARTER PROFILE (Normal case)")
    print("="*50)
    recommendations = recommend_songs(user_prefs, songs, k=5)

    print("\nTop recommendations:\n")
    for rec in recommendations:
        # You decide the structure of each returned item.
        # A common pattern is: (song, score, explanation)
        song, score, explanation = rec
        print(f"{song['title']} - Score: {score:.2f}")
        print(f"{explanation}")
        print()

    print("\n" + "="*50)
    print("ADVERSARIAL PROFILE 1: Contradicted User")
    print("(High energy + Melancholic mood)")
    print("="*50)
    recommendations = recommend_songs(user_prefs_contradicted, songs, k=5)

    print("\nTop recommendations:\n")
    for rec in recommendations:
        song, score, explanation = rec
        print(f"{song['title']} - Score: {score:.2f}")
        print(f"{explanation}")
        print()

    print("\n" + "="*50)
    print("ADVERSARIAL PROFILE 2: Genre Ghost")
    print("(Non-existent genre 'techno' + unused tempo_bpm)")
    print("="*50)
    recommendations = recommend_songs(user_prefs_ghost_genre, songs, k=5)

    print("\nTop recommendations:\n")
    for rec in recommendations:
        song, score, explanation = rec
        print(f"{song['title']} - Score: {score:.2f}")
        print(f"{explanation}")
        print()

    print("\n" + "="*50)
    print("TEST PROFILE 3: Chill Lofi")
    print("(Low energy + Slow tempo)")
    print("="*50)
    recommendations = recommend_songs(user_prefs_chill_lofi, songs, k=5)

    print("\nTop recommendations:\n")
    for rec in recommendations:
        song, score, explanation = rec
        print(f"{song['title']} - Score: {score:.2f}")
        print(f"{explanation}")
        print()

    print("\n" + "="*50)
    print("TEST PROFILE 4: Deep Intense Rock")
    print("(High energy + Fast tempo)")
    print("="*50)
    recommendations = recommend_songs(user_prefs_intense_rock, songs, k=5)

    print("\nTop recommendations:\n")
    for rec in recommendations:
        song, score, explanation = rec
        print(f"{song['title']} - Score: {score:.2f}")
        print(f"{explanation}")
        print()



if __name__ == "__main__":
    main()
