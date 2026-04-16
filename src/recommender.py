import csv
from typing import List, Dict, Tuple
from dataclasses import dataclass
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

# Constants for the scoring weights (out of 3.5 total)
GENRE_WEIGHT = 1.5
MOOD_WEIGHT = 1.0
ENERGY_WEIGHT = 1.0


@dataclass
class Song:
    """
    Represents a song and its attributes.
    Required by tests/test_recommender.py
    """
    id: int
    title: str
    artist: str
    genre: str
    mood: str
    energy: float
    tempo_bpm: float
    valence: float
    danceability: float
    acousticness: float
    use_case: str = ""
    language: str = ""
    year: int = 0
    description: str = ""


@dataclass
class UserProfile:
    """
    Represents a user's taste preferences.
    Required by tests/test_recommender.py
    """
    favorite_genres: List[str]
    favorite_moods: List[str]
    target_energy: float
    target_valence: float
    target_tempo_bpm: float
    target_danceability: float


def score_song(user: UserProfile, song: Song) -> float:
    """
    Score a song against a user profile using weighted hybrid scoring.

    - Genre match: +1.5 points (binary)
    - Mood match:  +1.0 point  (binary)
    - Energy:      up to +1.0  (inverse distance)

    Returns a score between 0.0 and 3.5.
    """
    score = 0.0

    # Categorical scoring
    score += GENRE_WEIGHT if song.genre in user.favorite_genres else 0
    score += MOOD_WEIGHT if song.mood in user.favorite_moods else 0

    # Numerical scoring (energy is already on 0-1 scale)
    score += ENERGY_WEIGHT * (1.0 - abs(song.energy - user.target_energy))

    return round(score, 2)


def explain_score(user: UserProfile, song: Song) -> str:
    """Build a human-readable explanation of why a song was recommended."""
    reasons = []

    if song.genre in user.favorite_genres:
        reasons.append(f"genre is {song.genre} (+{GENRE_WEIGHT})")
    if song.mood in user.favorite_moods:
        reasons.append(f"mood is {song.mood} (+{MOOD_WEIGHT})")

    energy_pts = round(
        ENERGY_WEIGHT * (1.0 - abs(song.energy - user.target_energy)), 2)
    reasons.append(f"energy similarity (+{energy_pts}/{ENERGY_WEIGHT})")

    if not reasons:
        return "No strong matches, but still in your top results."

    return "Matched because: " + ", ".join(reasons)


class Recommender:
    """
    OOP implementation of the recommendation logic.
    Required by tests/test_recommender.py
    """

    def __init__(self, songs: List[Song]):
        self.songs = songs

    def recommend(self, user: UserProfile, k: int = 5) -> List[Song]:
        scored = [(song, score_song(user, song)) for song in self.songs]
        scored.sort(key=lambda x: x[1], reverse=True)
        return [song for song, _ in scored[:k]]

    def explain_recommendation(self, user: UserProfile, song: Song) -> str:
        total = score_song(user, song)
        explanation = explain_score(user, song)
        return f"Score: {total}/3.5. {explanation}"


def load_songs(csv_path: str) -> List[Dict]:
    """
    Loads songs from a CSV file.
    Required by src/main.py
    """
    songs = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if not row.get("id"):
                continue
            songs.append({
                "id": int(row["id"]),
                "title": row["title"],
                "artist": row["artist"],
                "genre": row["genre"],
                "mood": row["mood"],
                "energy": float(row["energy"]),
                "tempo_bpm": float(row["tempo_bpm"]),
                "valence": float(row["valence"]),
                "danceability": float(row["danceability"]),
                "acousticness": float(row["acousticness"]),
                "use_case": row.get("use_case", ""),
                "language": row.get("language", ""),
                "year": int(row.get("year", 0)) if row.get("year") else 0,
                "description": row.get("description", ""),
            })
    return songs


def recommend_songs(user_prefs: Dict, songs: List[Dict], k: int = 5) -> List[Tuple[Dict, float, str]]:
    """
    Functional implementation of the recommendation logic.
    Required by src/main.py

    Accepts a simple prefs dict (from main.py) and a list of song dicts.
    Returns list of (song_dict, score, explanation) tuples sorted by score descending.
    """
    # Build a UserProfile from the prefs dict, using sensible defaults
    user = UserProfile(
        favorite_genres=[user_prefs["genre"]] if "genre" in user_prefs else [],
        favorite_moods=[user_prefs["mood"]] if "mood" in user_prefs else [],
        target_energy=user_prefs.get("energy", 0.5),
        target_valence=user_prefs.get("valence", 0.5),
        target_tempo_bpm=user_prefs.get("tempo_bpm", 110),
        target_danceability=user_prefs.get("danceability", 0.5),
    )

    results = []
    for s in songs:
        song_obj = Song(
            id=s["id"], title=s["title"], artist=s["artist"],
            genre=s["genre"], mood=s["mood"], energy=s["energy"],
            tempo_bpm=s["tempo_bpm"], valence=s["valence"],
            danceability=s["danceability"], acousticness=s["acousticness"],
        )
        sc = score_song(user, song_obj)
        explanation = explain_score(user, song_obj)
        results.append((s, sc, explanation))

    results.sort(key=lambda x: x[1], reverse=True)
    return results[:k]


# ============================================================================
# PHASE 1: RAG (Retrieval-Augmented Generation) System
# ============================================================================

def create_song_document(song: Song) -> str:
    """
    Convert a Song object into one searchable document combining all metadata.

    This text will be embedded and used for semantic similarity search.
    Includes: title, artist, genre, mood, use_case, language, year, description
    """
    doc = f"""
Title: {song.title}
Artist: {song.artist}
Genre: {song.genre}
Mood: {song.mood}
Use Case: {song.use_case}
Language: {song.language}
Year: {song.year}
Description: {song.description}
""".strip()
    return doc


class MusicRAG:
    """
    RAG (Retrieval-Augmented Generation) system for music recommendations.

    This class handles:
    1. Creating embeddings from song metadata
    2. Storing them in a FAISS index
    3. Retrieving semantically similar songs via vector search
    """

    def __init__(self, songs: List[Song], model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize the RAG system.

        Args:
            songs: List of Song objects to index
            model_name: Sentence transformer model (all-MiniLM-L6-v2 is small & fast)
        """
        self.songs = songs
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)
        self.index = None
        self.document_embeddings = None

        # Build embeddings and index
        self._build_index()

    def _build_index(self):
        """Create embeddings for all songs and build FAISS index."""
        print(f"Building RAG index for {len(self.songs)} songs...")

        # Step 1: Convert each song to a document
        documents = [create_song_document(song) for song in self.songs]

        # Step 2: Embed documents (creates 384-dim vectors per song)
        print("Generating embeddings...")
        self.document_embeddings = self.model.encode(
            documents,
            convert_to_numpy=True,
            show_progress_bar=True
        )

        # Step 3: Build FAISS index for fast similarity search
        print("Building FAISS similarity index...")
        embedding_dim = self.document_embeddings.shape[1]  # 384 for MiniLM
        self.index = faiss.IndexFlatL2(
            embedding_dim)  # L2 = Euclidean distance
        self.index.add(self.document_embeddings.astype(np.float32))  # type: ignore[arg-type]

        print(f"✓ RAG ready! {len(self.songs)} songs indexed.")

    def retrieve(self, query: str, k: int = 5) -> List[Tuple[Song, float]]:
        """
        Retrieve top-k most similar songs for a query.

        Args:
            query: Natural language query (e.g., "energetic pop for working out")
            k: Number of results to return

        Returns:
            List of (Song, similarity_score) tuples ranked by relevance
        """
        # Step 1: Embed the user's query
        query_embedding = self.model.encode([query], convert_to_numpy=True).astype(np.float32)

        # Step 2: Search FAISS index for nearest neighbors
        # Returns distances (lower = more similar) and indices
        distances, indices = self.index.search(query_embedding, k)  # type: ignore[arg-type]

        # Step 3: Convert to (Song, score) format
        # Normalize distance to 0-1 similarity score (1.0 = perfect match)
        results = []
        for idx, distance in zip(indices[0], distances[0]):
            song = self.songs[idx]
            # Convert L2 distance to similarity (0-1 scale)
            similarity = 1.0 / (1.0 + distance)
            results.append((song, round(similarity, 3)))

        return results

    def explain_retrieval(self, query: str, song: Song, similarity: float) -> str:
        """Generate human-readable explanation of why song matched the query."""
        return f"Similarity: {similarity:.1%} - '{song.title}' matches '{query}'"


def create_rag_system(csv_path: str) -> MusicRAG:
    """Load songs from CSV and create a ready-to-use RAG system."""
    song_dicts = load_songs(csv_path)
    songs = [
        Song(
            id=s["id"], title=s["title"], artist=s["artist"],
            genre=s["genre"], mood=s["mood"], energy=s["energy"],
            tempo_bpm=s["tempo_bpm"], valence=s["valence"],
            danceability=s["danceability"], acousticness=s["acousticness"],
            use_case=s["use_case"], language=s["language"],
            year=s["year"], description=s["description"],
        )
        for s in song_dicts
    ]
    return MusicRAG(songs)


# ============================================================================
# PHASE 2: Hybrid Scoring (RAG + Content-Based)
# ============================================================================

class HybridRecommender:
    """
    Combines RAG (semantic similarity) with content-based scoring (weighted genre/mood/energy).

    Strategy:
    1. Use RAG to retrieve candidate songs (semantic expansion)
    2. Score candidates with existing weighted scoring
    3. Re-rank by hybrid score (blend of both approaches)
    4. Return top-k results

    This gives best of both worlds:
    - RAG finds semantically similar songs (understands natural language)
    - Weighted scoring enforces user preferences (genre, mood, energy)
    - Hybrid ranking balances discovery with preference matching
    """

    def __init__(self, rag_system: MusicRAG, rag_weight: float = 0.4, content_weight: float = 0.6):
        """
        Initialize hybrid recommender.

        Args:
            rag_system: Initialized MusicRAG instance
            rag_weight: Weight for RAG retrieval score (0.0-1.0)
            content_weight: Weight for content-based score (0.0-1.0)
        """
        self.rag = rag_system
        self.rag_weight = rag_weight
        self.content_weight = content_weight

        # Validate weights sum to 1.0
        total = rag_weight + content_weight
        if abs(total - 1.0) > 0.01:
            print(f"⚠️  Warning: weights sum to {total}, normalizing...")
            self.rag_weight = rag_weight / total
            self.content_weight = content_weight / total

    def recommend(self, query: str, user_profile: UserProfile, k: int = 5, expansion_factor: int = 3) -> List[Tuple[Song, float, Dict]]:
        """
        Get hybrid recommendations using RAG + content-based scoring.

        Args:
            query: Natural language query (e.g., "chill lofi for studying")
            user_profile: UserProfile with preferences (genre, mood, energy, etc.)
            k: Number of final recommendations to return
            expansion_factor: How many candidates to retrieve via RAG (k * expansion_factor)

        Returns:
            List of (Song, hybrid_score, metadata_dict) sorted by hybrid_score descending
        """
        # Step 1: Use RAG to retrieve candidate songs (expand search space)
        # At least 10 candidates
        candidate_count = max(k * expansion_factor, 10)
        rag_results = self.rag.retrieve(query, k=candidate_count)

        # Step 2: Score candidates with content-based scoring
        hybrid_results = []
        for song, rag_score in rag_results:
            # Get content-based score (0.0-3.5 scale)
            content_score = score_song(user_profile, song)

            # Normalize both scores to 0-1 scale
            norm_rag_score = rag_score  # Already 0-1
            norm_content_score = content_score / 3.5  # Normalize from 0-3.5 to 0-1

            # Hybrid score: weighted combination
            hybrid_score = (self.rag_weight * norm_rag_score +
                            self.content_weight * norm_content_score)

            metadata = {
                "rag_score": round(rag_score, 3),
                "content_score": round(content_score, 2),
                "hybrid_score": round(hybrid_score, 3),
                "explanation": self._explain_recommendation(user_profile, song, rag_score, content_score)
            }

            hybrid_results.append((song, hybrid_score, metadata))

        # Step 3: Re-rank by hybrid score and return top-k
        hybrid_results.sort(key=lambda x: x[1], reverse=True)
        return hybrid_results[:k]

    def _explain_recommendation(self, user_profile: UserProfile, song: Song, rag_score: float, content_score: float) -> str:
        """Generate explanation combining both scoring approaches."""
        reasons = []

        # RAG alignment
        if rag_score > 0.7:
            reasons.append(
                f"semantically similar to your query ({rag_score:.0%})")
        
        # Content score contribution
        norm_content_score = content_score / 3.5
        if norm_content_score > 0.5:
            reasons.append(f"content-based match ({norm_content_score:.0%})")

        # Content-based alignment
        if song.genre in user_profile.favorite_genres:
            reasons.append(f"matches your genre preference ({song.genre})")
        if song.mood in user_profile.favorite_moods:
            reasons.append(f"matches your mood preference ({song.mood})")

        energy_match = 1.0 - abs(song.energy - user_profile.target_energy)
        if energy_match > 0.8:
            reasons.append("energy level matches your preference")

        if not reasons:
            reasons.append("found through semantic search")

        return "Recommended because: " + ", ".join(reasons) + "."

    def explain_results(self, results: List[Tuple[Song, float, Dict]]) -> str:
        """Pretty-print recommendation results with explanations."""
        output = []
        for i, (song, hybrid_score, metadata) in enumerate(results, 1):
            output.append(
                f"{i}. {song.title} by {song.artist}\n"
                f"   Genre: {song.genre} | Mood: {song.mood} | Use Case: {song.use_case}\n"
                f"   Hybrid Score: {hybrid_score:.1%} "
                f"(RAG: {metadata['rag_score']:.1%}, Content: {metadata['content_score']/3.5:.1%})\n"
                f"   {metadata['explanation']}\n"
            )
        return "\n".join(output)


def create_hybrid_recommender(csv_path: str, rag_weight: float = 0.4, content_weight: float = 0.6) -> HybridRecommender:
    """One-liner to create a ready-to-use hybrid recommender from CSV."""
    rag = create_rag_system(csv_path)
    return HybridRecommender(rag, rag_weight=rag_weight, content_weight=content_weight)
