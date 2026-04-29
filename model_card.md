# Model card: RhythmFlow

## 1. Model name

RhythmFlow music recommender

## 2. Intended use

RhythmFlow recommends songs from a small local catalog. It is meant for a class project and for testing recommender behavior, not for production music discovery.

A user can ask for music in natural language, such as "chill lofi for studying" or "high energy rock for a workout." The system reads that request, searches the song catalog, scores candidate songs, and returns a ranked list. The app can also use Gemini to explain the results, but Gemini does not decide the ranking.

This project is useful because the recommendation logic is small enough to inspect. You can see how a single weight or missing catalog entry changes the output.

## 3. How the system works

RhythmFlow has two ranking signals.

First, it uses a RAG layer. Each song is turned into a short text document with its title, artist, genre, mood, use case, language, year, and description. Those documents are embedded and stored in a FAISS index. When a user types a query, the system retrieves songs whose metadata is semantically close to the query.

Second, it uses content scoring. Each candidate song can earn up to 3.5 points:

- Genre match: 1.5 points
- Mood match: 1.0 point
- Energy similarity: up to 1.0 point

Energy similarity uses distance from the user's target energy. If the user wants `0.8` energy and the song is also `0.8`, the song gets the full energy point. If the song is `0.5`, it gets partial credit.

The hybrid recommender blends the RAG score and content score, then returns the top results. By default, content scoring matters more than semantic search. That keeps the results explainable, but it also means the hand-picked weights have a real effect on what users see.

## 4. Data

The dataset is `data/songs.csv`. It contains 18 songs across genres such as pop, lofi, rock, ambient, jazz, synthwave, indie pop, metal, reggae, classical, R&B, electronic, country, hip-hop, and latin.

Each row includes:

- Basic metadata: `id`, `title`, `artist`, `genre`, `mood`
- Audio-style fields: `energy`, `tempo_bpm`, `valence`, `danceability`, `acousticness`
- Extra context: `use_case`, `language`, `year`, `description`

The catalog is big enough to test the recommender, but it is uneven. Pop and lofi have more examples than metal, jazz, or classical. That matters. If a genre only has one song, the system cannot return a varied set of recommendations for that genre.

## 5. What works well

RhythmFlow works best when the user's preferences agree with each other. A request like "intense rock with high energy" is easy because rock, intense mood, and high energy all point in roughly the same direction.

Energy scoring is also useful. It gives the system a sliding scale instead of only yes/no matches. A chill lofi profile with `0.3` target energy gets very different results from a gym profile with `0.9` target energy. That part behaved the way I expected.

The RAG layer helps with natural language. A user does not have to know the exact genre or mood tag in the CSV. Queries like "music for coding late at night" can still retrieve relevant songs because the search reads descriptions and use cases, not only exact labels.

## 6. Limitations and bias

The genre weight is strong. Since genre is worth 1.5 out of 3.5 content points, genre-matched songs can beat songs that might fit the mood better. This creates a small version of a filter bubble: the system keeps pulling users back toward the genre they named.

The catalog is uneven too. Pop and lofi users get more variety because those genres have more songs. Users asking for metal, jazz, or classical have fewer choices, so the system either repeats narrow results or drifts into other genres.

Some fields are underused. The CSV includes tempo, valence, danceability, and acousticness, but the main weighted score still uses only genre, mood, and energy. That means a user can care about tempo, but the core score will not fully respect it yet.

The system also does not handle contradictions well. If a user asks for "melancholic metal with high energy," the recommender does not stop and say, "That combination is not really in this catalog." It just ranks the closest compromises.

Gemini explanations can fail separately from recommendations. If the API key is missing or the model is busy, the app can still rank songs, but the explanation may be unavailable or replaced with a fallback message.

## 7. Evaluation

I tested the system with normal profiles and edge cases.

Test profiles:

- Starter profile: pop, happy mood, `0.8` energy
- Chill lofi: lofi, chill mood, `0.3` energy
- Deep intense rock: rock, intense mood, `0.9` energy
- Contradicted user: metal, melancholic mood, `0.95` energy
- Genre ghost: techno, ethereal mood, `0.8` energy

What I saw:

- The starter profile returned the expected kind of upbeat pop recommendations.
- Chill lofi shifted the results toward lower-energy tracks like `Library Rain` and `Midnight Coding`.
- Deep intense rock pushed high-energy songs like `Storm Runner` toward the top.
- The contradicted profile exposed a weakness. The system had to choose between metal, melancholic mood, and high energy because the catalog does not really contain all three together.
- The genre ghost profile showed the fallback behavior. When genre and mood do not exist in the catalog, the system leans on energy and semantic similarity, which can produce results that feel less coherent.

Automated tests cover the basic scoring order, recommendation explanations, lofi behavior, Gemini fallback behavior, and the dashboard element IDs used by the frontend.

## 8. Future work

- Add more songs, especially for genres with only one example.
- Use tempo, valence, danceability, and acousticness in the scoring function.
- Detect conflicting requests before ranking.
- Tell the user when the catalog does not contain a requested genre or mood.
- Add diversity logic so the top results are not all pulled from the same narrow area.
- Store profiles and feedback in a database instead of local JSON files.
- Let users adjust the weights from the UI.

## 9. Responsible use notes

RhythmFlow should not be treated like a full music platform. It is a small recommender demo with a tiny dataset. Its results reflect the catalog and the weights I chose. If the catalog is unbalanced, the recommendations will be unbalanced too.
