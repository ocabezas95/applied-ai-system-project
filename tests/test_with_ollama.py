from src.recommender import (
    ConversationalRecommender,
    HybridRecommender,
    MusicRAG,
    Song,
    UserProfile,
    create_song_document,
)


def make_songs():
    return [
        Song(
            id=1,
            title="Sunrise Sprint",
            artist="Neon Echo",
            genre="pop",
            mood="energetic",
            energy=0.9,
            tempo_bpm=128,
            valence=0.8,
            danceability=0.82,
            acousticness=0.12,
            use_case="workout",
            language="en",
            year=2024,
            description="Energetic pop for gym sessions and morning runs.",
        ),
        Song(
            id=2,
            title="Library Rain",
            artist="Paper Lanterns",
            genre="lofi",
            mood="chill",
            energy=0.3,
            tempo_bpm=72,
            valence=0.55,
            danceability=0.48,
            acousticness=0.91,
            use_case="study",
            language="en",
            year=2021,
            description="Quiet lofi for reading, studying, and focus.",
        ),
        Song(
            id=3,
            title="Iron Anthem",
            artist="Voltline",
            genre="rock",
            mood="intense",
            energy=0.88,
            tempo_bpm=150,
            valence=0.44,
            danceability=0.61,
            acousticness=0.08,
            use_case="training",
            language="en",
            year=2023,
            description="Heavy guitars and driving drums for hard workouts.",
        ),
    ]


class FakeSentenceTransformer:
    def __init__(self, model_name: str):
        self.model_name = model_name

    def encode(self, texts, convert_to_numpy=True, show_progress_bar=False):
        vectors = []
        for text in texts:
            lowered = text.lower()
            vectors.append(
                [
                    2.0 if "pop" in lowered else 0.0,
                    2.0 if "workout" in lowered or "gym" in lowered else 0.0,
                    2.0 if "study" in lowered or "focus" in lowered else 0.0,
                    2.0 if "rock" in lowered or "guitar" in lowered else 0.0,
                ]
            )
        if convert_to_numpy:
            from numpy import array, float32

            return array(vectors, dtype=float32)
        return vectors


class RecordingLLM:
    def __init__(self):
        self.prompts = []

    def generate(self, prompt: str, max_tokens: int = 500) -> str:
        self.prompts.append(prompt)
        return "Grounded summary."

    def summarize_songs(self, songs, query, retrieval_context):
        song_list = "\n".join(f"- {song.title}" for song in songs)
        context = "\n".join(
            f"- {item['title']} | doc={item['document']} | score={item['rag_score']}"
            for item in retrieval_context
        )
        return self.generate(
            f"Query: {query}\nSongs:\n{song_list}\nContext:\n{context}",
            max_tokens=300,
        )


def test_music_rag_falls_back_to_local_embedder_when_transformer_load_fails(monkeypatch):
    from src import recommender as recommender_module

    class BrokenSentenceTransformer:
        def __init__(self, model_name: str):
            raise OSError("network unavailable")

    monkeypatch.setattr(
        recommender_module,
        "SentenceTransformer",
        BrokenSentenceTransformer,
    )

    rag = MusicRAG(make_songs())
    results = rag.retrieve("energetic pop workout", k=2)

    assert len(results) == 2
    assert results[0][0].title == "Sunrise Sprint"
    assert 0.0 <= results[0][1] <= 1.0
    assert getattr(rag, "using_fallback_embeddings") is True


def test_music_rag_uses_normalized_similarity_scores():
    rag = MusicRAG(make_songs(), embedding_model=FakeSentenceTransformer("fake"))

    results = rag.retrieve("pop workout music", k=3)

    assert results[0][0].title == "Sunrise Sprint"
    assert results[0][1] > results[1][1]
    assert results[0][1] <= 1.0


def test_hybrid_recommender_combines_retrieval_and_content_scores():
    rag = MusicRAG(make_songs(), embedding_model=FakeSentenceTransformer("fake"))
    hybrid = HybridRecommender(rag, rag_weight=0.4, content_weight=0.6)
    user = UserProfile(
        favorite_genres=["pop"],
        favorite_moods=["energetic"],
        target_energy=0.9,
        target_valence=0.8,
        target_tempo_bpm=125,
        target_danceability=0.8,
    )

    results = hybrid.recommend("upbeat pop for the gym", user, k=2)

    assert len(results) == 2
    top_song, top_score, metadata = results[0]
    assert top_song.title == "Sunrise Sprint"
    assert top_score == metadata["hybrid_score"]
    assert metadata["rag_score"] <= 1.0
    assert metadata["content_score"] > 0


def test_conversational_recommender_passes_grounded_context_to_llm():
    rag = MusicRAG(make_songs(), embedding_model=FakeSentenceTransformer("fake"))
    hybrid = HybridRecommender(rag)
    llm = RecordingLLM()
    recommender = ConversationalRecommender(hybrid, llm=llm)

    response = recommender.chat("I need energetic pop for workouts", k=2)

    assert response["llm_explanation"] == "Grounded summary."
    assert llm.prompts
    assert "Context:" in llm.prompts[0]
    assert create_song_document(make_songs()[0]) in llm.prompts[0]
    assert "score=" in llm.prompts[0]


def test_src_main_imports_recommender_from_package():
    import src.main as main_module

    assert callable(main_module.main)
