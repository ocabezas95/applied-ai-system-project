import csv
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from datetime import datetime

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


class LocalFallbackEmbedder:
    """
    Tiny local embedder used when sentence-transformer weights are unavailable.

    It builds a bag-of-words vocabulary from the song documents so retrieval still
    works offline and remains deterministic in tests.
    """

    def __init__(self, seed_documents: List[str]):
        self.vocabulary = self._build_vocabulary(seed_documents)

    def _build_vocabulary(self, documents: List[str]) -> Dict[str, int]:
        vocab: Dict[str, int] = {}
        for document in documents:
            for token in self._tokenize(document):
                if token not in vocab:
                    vocab[token] = len(vocab)
        return vocab

    def _tokenize(self, text: str) -> List[str]:
        cleaned = []
        for char in text.lower():
            cleaned.append(char if char.isalnum() else " ")
        return [token for token in "".join(cleaned).split() if token]

    def encode(self, texts, convert_to_numpy=True, show_progress_bar=False):
        vectors = np.zeros((len(texts), len(self.vocabulary)), dtype=np.float32)
        for row_idx, text in enumerate(texts):
            for token in self._tokenize(text):
                token_idx = self.vocabulary.get(token)
                if token_idx is not None:
                    vectors[row_idx, token_idx] += 1.0
        if convert_to_numpy:
            return vectors
        return vectors.tolist()


class MusicRAG:
    """
    RAG (Retrieval-Augmented Generation) system for music recommendations.

    This class handles:
    1. Creating embeddings from song metadata
    2. Storing them in a FAISS index
    3. Retrieving semantically similar songs via vector search
    """

    def __init__(
        self,
        songs: List[Song],
        model_name: str = "all-MiniLM-L6-v2",
        embedding_model=None,
    ):
        """
        Initialize the RAG system.

        Args:
            songs: List of Song objects to index
            model_name: Sentence transformer model (all-MiniLM-L6-v2 is small & fast)
        """
        self.songs = songs
        self.model_name = model_name
        self.documents = [create_song_document(song) for song in self.songs]
        self.using_fallback_embeddings = False
        self.model = embedding_model or self._load_embedding_model()
        self.index = None
        self.document_embeddings = None

        # Build embeddings and index
        self._build_index()

    def _load_embedding_model(self):
        try:
            return SentenceTransformer(self.model_name, local_files_only=True)
        except TypeError:
            try:
                return SentenceTransformer(self.model_name)
            except Exception:
                self.using_fallback_embeddings = True
                return LocalFallbackEmbedder(self.documents)
        except Exception:
            self.using_fallback_embeddings = True
            return LocalFallbackEmbedder(self.documents)

    def _normalize_embeddings(self, embeddings: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms = np.where(norms == 0.0, 1.0, norms)
        return embeddings / norms

    def _build_index(self):
        """Create embeddings for all songs and build FAISS index."""
        print(f"Building RAG index for {len(self.songs)} songs...")

        # Step 1: Embed documents
        print("Generating embeddings...")
        embeddings = self.model.encode(
            self.documents,
            convert_to_numpy=True,
            show_progress_bar=True
        ).astype(np.float32)
        self.document_embeddings = self._normalize_embeddings(embeddings)

        # Step 2: Build FAISS index using normalized vectors for cosine-style similarity
        print("Building FAISS similarity index...")
        embedding_dim = self.document_embeddings.shape[1]  # 384 for MiniLM
        self.index = faiss.IndexFlatIP(embedding_dim)
        self.index.add(self.document_embeddings.astype(np.float32))  # type: ignore[arg-type]

        print(f"✓ RAG ready! {len(self.songs)} songs indexed.")

    def retrieve(self, query: str, k: int = 5) -> List[Tuple[Song, float, str]]:
        """
        Retrieve top-k most similar songs for a query.

        Args:
            query: Natural language query (e.g., "energetic pop for working out")
            k: Number of results to return

        Returns:
            List of (Song, similarity_score, document_text) tuples ranked by relevance
        """
        # Step 1: Embed the user's query
        query_embedding = self.model.encode([query], convert_to_numpy=True).astype(np.float32)
        query_embedding = self._normalize_embeddings(query_embedding)

        # Step 2: Search FAISS index for nearest neighbors
        top_k = min(k, len(self.songs))
        scores, indices = self.index.search(query_embedding, top_k)  # type: ignore[arg-type]

        # Step 3: Convert to (Song, score) format
        results = []
        for idx, score in zip(indices[0], scores[0]):
            if idx < 0:
                continue
            song = self.songs[idx]
            similarity = float(np.clip(score, 0.0, 1.0))
            results.append((song, round(similarity, 3), self.documents[idx]))

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
        for song, rag_score, document in rag_results:
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
                "document": document,
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


# ============================================================================
# PHASE 3: LLM Integration + Conversational Interface
# ============================================================================

class LLMInterface:
    """
    Abstract interface for LLM backends (OpenAI or Ollama).
    Subclasses implement actual LLM calls.
    """
    
    def generate(self, prompt: str, max_tokens: int = 500) -> str:
        """Generate text from a prompt."""
        raise NotImplementedError
    
    def summarize_songs(self, songs: List[Song], query: str, retrieval_context: List[Dict]) -> str:
        """Generate natural language summary of song recommendations."""
        raise NotImplementedError


class OllamaLLM(LLMInterface):
    """
    Local LLM via Ollama (no API keys needed, runs locally).
    
    Requires:
    - Ollama installed: https://ollama.ai
    - Model pulled: ollama pull mistral (or llama2, neural-chat)
    - Service running: ollama serve
    """
    
    def __init__(self, model: str = "mistral", base_url: str = "http://localhost:11434"):
        """
        Args:
            model: Ollama model name (mistral, llama2, neural-chat, etc.)
            base_url: Ollama API endpoint
        """
        self.model = model
        self.base_url = base_url
        self._verify_connection()
    
    def _verify_connection(self):
        """Check if Ollama service is running."""
        try:
            import requests
            response = requests.get(f"{self.base_url}/api/tags", timeout=2)
            if response.status_code == 200:
                print(f"✓ Connected to Ollama on {self.base_url}")
            else:
                print(f"⚠️  Ollama connection returned status {response.status_code}")
        except Exception as e:
            print(f"⚠️  Ollama not reachable at {self.base_url}. Start with 'ollama serve'")
            print(f"   Error: {e}")
    
    def generate(self, prompt: str, max_tokens: int = 500) -> str:
        """Generate text using Ollama."""
        try:
            import requests
            response = requests.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "stream": False,
                    "num_predict": max_tokens,
                },
                timeout=30
            )
            if response.status_code == 200:
                return response.json()["response"].strip()
            else:
                return f"[LLM Error: {response.status_code}]"
        except ImportError:
            return "[Error: requests library not installed. pip install requests]"
        except Exception as e:
            return f"[LLM Error: {str(e)}]"
    
    def summarize_songs(self, songs: List[Song], query: str, retrieval_context: List[Dict]) -> str:
        """Generate natural language summary of recommendations."""
        song_list = "\n".join([
            f"- {s.title} by {s.artist} ({s.genre}, {s.mood}, {s.use_case}): {s.description}"
            for s in songs
        ])
        context_blocks = "\n\n".join([
            (
                f"Title: {item['title']}\n"
                f"RAG score: {item['rag_score']}\n"
                f"Content score: {item['content_score']}\n"
                f"Source document:\n{item['document']}"
            )
            for item in retrieval_context
        ])

        prompt = f"""Given the user query: "{query}"

Here are my recommendations:
{song_list}

Retrieved evidence:
{context_blocks}

Write a brief, friendly explanation (2-3 sentences) of why these songs match the request.
Use only the retrieved evidence above, and do not invent attributes that are not present.
Be conversational and avoid technical jargon."""
        
        return self.generate(prompt, max_tokens=300)


class ConversationalRecommender:
    """
    Conversational music recommender combining:
    1. Hybrid recommender (RAG + content-based scoring)
    2. LLM for natural language explanations
    3. Chat history management
    
    Provides a chat-like interface where users can ask for recommendations
    in natural language and get detailed, personalized responses.
    """
    
    def __init__(self, hybrid_recommender: HybridRecommender, llm: Optional[LLMInterface] = None):
        """
        Args:
            hybrid_recommender: HybridRecommender instance
            llm: LLMInterface (OpenAI, Ollama, etc.). If None, uses basic explanations.
        """
        self.hybrid = hybrid_recommender
        self.llm = llm
        self.chat_history: List[Dict] = []
        self.user_profile = UserProfile(
            favorite_genres=[],
            favorite_moods=[],
            target_energy=0.7,
            target_valence=0.7,
            target_tempo_bpm=110,
            target_danceability=0.7
        )
    
    def extract_preferences_from_query(self, query: str) -> Dict:
        """
        Parse user query to extract preferences (heuristic-based).
        
        Examples:
        - "pop songs" -> genres: [pop]
        - "happy upbeat music" -> moods: [happy]
        - "slow relaxing songs" -> energy: 0.3
        """
        query_lower = query.lower()
        extracted = {
            "genres": [],
            "moods": [],
            "energy": None,
        }
        
        # Genre extraction (simple heuristic)
        genres_map = {
            "pop": "pop", "rock": "rock", "jazz": "jazz",
            "indie": "indie pop", "lofi": "lofi", "classical": "classical",
            "metal": "metal", "reggae": "reggae", "hip-hop": "hip-hop",
            "electronic": "electronic", "house": "house", "r&b": "r&b",
            "country": "country", "funk": "funk", "soul": "soul"
        }
        for keyword, genre in genres_map.items():
            if keyword in query_lower:
                extracted["genres"].append(genre)
        
        # Mood extraction (simple heuristic)
        moods_map = {
            "happy": "happy", "sad": "melancholic", "chill": "chill",
            "relaxed": "relaxed", "energetic": "energetic", "intense": "intense",
            "peaceful": "peaceful", "romantic": "romantic", "focused": "focused"
        }
        for keyword, mood in moods_map.items():
            if keyword in query_lower:
                extracted["moods"].append(mood)
        
        # Energy extraction (heuristic)
        if any(word in query_lower for word in ["slow", "chill", "relax", "calm"]):
            extracted["energy"] = 0.3
        elif any(word in query_lower for word in ["energetic", "upbeat", "high energy", "intense"]):
            extracted["energy"] = 0.85
        
        return extracted
    
    def chat(self, user_input: str, k: int = 5) -> Dict:
        """
        Process user query and return recommendations with LLM explanation.
        
        Args:
            user_input: User's natural language query
            k: Number of recommendations to return
        
        Returns:
            Dict with recommendations, explanations, and chat history
        """
        timestamp = datetime.now().isoformat()
        
        # Step 1: Extract preferences from user input
        prefs = self.extract_preferences_from_query(user_input)
        
        # Step 2: Update user profile with extracted preferences
        if prefs["genres"]:
            self.user_profile.favorite_genres = prefs["genres"]
        if prefs["moods"]:
            self.user_profile.favorite_moods = prefs["moods"]
        if prefs["energy"] is not None:
            self.user_profile.target_energy = prefs["energy"]
        
        # Step 3: Get hybrid recommendations
        recommendations = self.hybrid.recommend(
            user_input,
            self.user_profile,
            k=k,
            expansion_factor=3
        )

        # Step 4: Generate LLM explanation (if LLM available)
        songs = [song for song, _, _ in recommendations]
        retrieval_context = [
            {
                "title": song.title,
                "document": metadata["document"],
                "rag_score": metadata["rag_score"],
                "content_score": metadata["content_score"],
            }
            for song, _, metadata in recommendations
        ]
        if self.llm:
            llm_explanation = self.llm.summarize_songs(songs, user_input, retrieval_context)
        else:
            llm_explanation = self._default_explanation(songs, user_input)
        
        # Step 5: Build response
        response = {
            "timestamp": timestamp,
            "user_query": user_input,
            "extracted_preferences": prefs,
            "recommendations": [
                {
                    "rank": i + 1,
                    "song": {
                        "id": song.id,
                        "title": song.title,
                        "artist": song.artist,
                        "genre": song.genre,
                        "mood": song.mood,
                        "use_case": song.use_case,
                        "description": song.description,
                    },
                    "score": round(score, 3),
                    "scores": {
                        "rag": metadata["rag_score"],
                        "content": round(metadata["content_score"] / 3.5, 3),
                        "hybrid": metadata["hybrid_score"]
                    }
                }
                for i, (song, score, metadata) in enumerate(recommendations)
            ],
            "llm_explanation": llm_explanation,
            "user_profile": {
                "favorite_genres": self.user_profile.favorite_genres,
                "favorite_moods": self.user_profile.favorite_moods,
                "target_energy": self.user_profile.target_energy,
            }
        }
        
        # Step 6: Add to chat history
        self.chat_history.append({
            "role": "user",
            "content": user_input,
            "timestamp": timestamp
        })
        self.chat_history.append({
            "role": "assistant",
            "content": response,
            "timestamp": timestamp
        })
        
        return response
    
    def _default_explanation(self, songs: List[Song], query: str) -> str:
        """Generate explanation without LLM."""
        genres = ", ".join(set(s.genre for s in songs))
        moods = ", ".join(set(s.mood for s in songs))
        return (
            f"I found {len(songs)} recommendations for your request. "
            f"These songs span {genres} genres with {moods} moods. "
            f"All match your preferences and the semantic meaning of '{query}'."
        )
    
    def print_recommendations(self, response: Dict) -> str:
        """Pretty-print recommendations with LLM explanation."""
        output = []
        output.append("\n" + "="*70)
        output.append("🎵 MUSIC RECOMMENDATIONS")
        output.append("="*70)
        output.append(f"\nYour Request: \"{response['user_query']}\"")
        output.append(f"\nAI's Explanation:\n{response['llm_explanation']}\n")
        output.append("Top Recommendations:")
        output.append("-" * 70)
        
        for rec in response["recommendations"]:
            song = rec["song"]
            output.append(
                f"\n{rec['rank']}. {song['title']} by {song['artist']}\n"
                f"   Genre: {song['genre']} | Mood: {song['mood']} | Use: {song['use_case']}\n"
                f"   Score: {rec['score']:.1%} (RAG: {rec['scores']['rag']:.1%}, "
                f"Content: {rec['scores']['content']:.1%})\n"
                f"   \"{song['description']}\""
            )
        
        output.append("\n" + "="*70)
        return "\n".join(output)
    
    def get_chat_history(self) -> str:
        """Get formatted chat history."""
        if not self.chat_history:
            return "No conversation history yet."
        
        output = []
        for msg in self.chat_history:
            if msg["role"] == "user":
                output.append(f"\nYou: {msg['content']}")
            else:
                # Assistant message - show summary
                rec = msg["content"]
                output.append(f"\nAssistant: Found {len(rec['recommendations'])} songs")
                output.append(f"Summary: {rec['llm_explanation'][:100]}...")
        
        return "\n".join(output)


def create_conversational_recommender(csv_path: str, use_llm: bool = False, llm_type: str = "ollama") -> ConversationalRecommender:
    """
    Create a ready-to-use conversational recommender.
    
    Args:
        csv_path: Path to songs CSV file
        use_llm: Whether to use LLM for explanations
        llm_type: "ollama" (free, local) or "openai" (requires API key)
    
    Returns:
        ConversationalRecommender ready for chat
    """
    hybrid = create_hybrid_recommender(csv_path, rag_weight=0.4, content_weight=0.6)
    
    llm = None
    if use_llm:
        if llm_type == "ollama":
            llm = OllamaLLM(model="mistral")
        else:
            print(f"⚠️  LLM type '{llm_type}' not yet implemented. Using basic explanations.")
    
    return ConversationalRecommender(hybrid, llm=llm)
