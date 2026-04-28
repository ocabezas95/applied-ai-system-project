from src.recommender import create_conversational_recommender

# Initialize with Ollama enabled
print("Initializing Conversational Recommender with Ollama...")
recommender = create_conversational_recommender(
    'data/songs.csv',
    use_llm=True,           # ← Enable LLM
    llm_type='ollama'       # ← Use Ollama
)

# Test 1: Natural language query
print("\n" + "="*70)
print("QUERY 1: Energetic Workout Music")
print("="*70)
response1 = recommender.chat("I need energetic upbeat music for my gym workout", k=5)
print(recommender.print_recommendations(response1))

# Test 2: Different mood
print("\n" + "="*70)
print("QUERY 2: Chill Focus Music")
print("="*70)
response2 = recommender.chat("relaxing lofi for deep focus and studying", k=3)
print(recommender.print_recommendations(response2))

# Test 3: Show chat history
print("\n" + "="*70)
print("CHAT HISTORY")
print("="*70)
print(recommender.get_chat_history())