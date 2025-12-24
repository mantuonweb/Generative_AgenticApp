from langchain_ollama import OllamaLLM, OllamaEmbeddings
from langchain_community.vectorstores import FAISS

# Setup AI
llm = OllamaLLM(model="llama2")
memory = []

# Knowledge Base
knowledge = [
    "AI Agents are programs that can think and act autonomously",
    "LangChain helps build AI applications easily",
    "FAISS is a vector database for fast similarity search",
    "Ollama runs AI models locally on your computer",
    "Python is a popular programming language for AI",
]

print("📚 Creating knowledge base (this may take a moment)...")
embeddings = OllamaEmbeddings(model="llama2")
kb = FAISS.from_texts(knowledge, embeddings)
print("✅ Knowledge base ready!")

print("\n🤖 Agent Ready!")
print("Commands: 'quit', 'clear', 'search: your query'")
print("-" * 60)

while True:
    user_input = input("\n💬 You: ").strip()
    
    if user_input.lower() in ['quit', 'exit']:
        print("👋 Bye!")
        break
    
    if user_input.lower() == 'clear':
        memory = []
        print("🧹 Memory cleared!")
        continue
    
    # Search knowledge
    if user_input.lower().startswith('search:'):
        query = user_input[7:].strip()
        results = kb.similarity_search(query, k=1)
        print(f"\n📚 Found: {results[0].page_content}")
        continue
    
    # Chat with memory
    context = "\n".join(memory[-6:])
    prompt = f"{context}\nHuman: {user_input}\nAssistant:" if context else user_input
    
    print("\n🤖 Agent:", end=" ", flush=True)
    response = llm.invoke(prompt)
    print(response)
    
    memory.append(f"Human: {user_input}")
    memory.append(f"Assistant: {response}")