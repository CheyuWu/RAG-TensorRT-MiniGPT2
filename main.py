from src.pipeline.rag_system import RAGSystem


DATA_PATH = "src/data/sample_docs.txt"


def main():
    """Main function - includes while loop for interactive inference"""

    print("=== RAG System with GPT-2 ===")
    print("Initializing system...")

    # Initialize RAG system
    rag = RAGSystem(
        retriever_model="all-MiniLM-L6-v2",
        gpt2_model="gpt2-medium",  # Can be changed to "gpt2-medium", "gpt2-large", "gpt2-xl"
    )

    # Load sample documents from file
    with open(DATA_PATH, "r", encoding="utf-8") as f:
        sample_docs = [line.strip() for line in f if line.strip()]
    rag.add_documents(sample_docs)

    print("\n" + "=" * 50)
    print("RAG System is ready!")
    print("You can ask questions about the documents.")
    print("Type 'quit', 'exit', or 'q' to stop.")
    print("Type 'docs' to see all documents.")
    print("Type 'clear' to clear the screen.")
    print("=" * 50)

    # Main while loop
    while True:
        try:
            # Get user input
            user_input = input("\n🤖 Ask me anything: ").strip()

            # Check exit conditions
            if user_input.lower() in ["quit", "exit", "q"]:
                print("👋 Goodbye!")
                break

            # Handle special commands
            if user_input.lower() == "docs":
                print("\n📚 Current documents:")
                for i, doc in enumerate(rag.retriever.docs, 1):
                    print(f"{i}. {doc}")
                continue

            if user_input.lower() == "clear":
                import os

                os.system("cls" if os.name == "nt" else "clear")
                continue

            # Check for empty input
            if not user_input:
                print("❌ Please enter a question.")
                continue

            print(f"\n🔍 Searching for relevant information...")

            # Perform RAG query
            result = rag.query(question=user_input, top_k=3, max_length=150)

            # Display results
            print(f"\n📖 Retrieved Documents:")
            if result["retrieved_docs"]:
                for i, doc in enumerate(result["retrieved_docs"], 1):
                    print(f"   {i}. {doc}")
            else:
                print("   No relevant documents found.")

            print(f"\n💬 Generated Response:")
            print(f"   {result['response']}")

            print("-" * 50)

        except KeyboardInterrupt:
            print("\n\n👋 Interrupted by user. Goodbye!")
            break
        except Exception as e:
            print(f"\n❌ Error occurred: {e}")
            print("Please try again with a different question.")


if __name__ == "__main__":
    main()
