import logging
from src.config.gpt_config import (
    DATA_PATH,
    GPT2_MODEL,
    MAX_LENGTH,
    RETRIEVER_MODEL,
    TOP_K,
)
from src.pipeline.rag_system import RAGSystem
import sys


def setup_logging():
    """Setup logging system - output to both file and console"""
    # Create logger
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # Clear existing handlers
    logger.handlers.clear()

    # Setup formatter
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    color_formatter = logging.Formatter("%(message)s")
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(color_formatter)
    logger.addHandler(console_handler)

    # Export handler - output to log.txt
    file_handler = logging.FileHandler("log.txt", mode="a", encoding="utf-8")
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger


def main():
    """Main function - includes while loop for interactive inference"""
    logger = setup_logging()
    logger.info("=== RAG System with GPT-2 ===")
    logger.info("Initializing system...")

    # Initialize RAG system
    rag = RAGSystem(
        retriever_model=RETRIEVER_MODEL,
        gpt2_model=GPT2_MODEL,
    )

    # Load sample documents from file
    with open(DATA_PATH, "r", encoding="utf-8") as f:
        sample_docs = [line.strip() for line in f if line.strip()]
    rag.add_documents(sample_docs)

    logger.info("RAG System is ready!")
    logger.info("You can ask questions about the documents.")
    logger.info("Type 'quit', 'exit', or 'q' to stop.")
    logger.info("Type 'docs' to see all documents.")
    logger.info("Type 'clear' to clear the screen.")

    # Main while loop
    while True:
        try:
            # Get user input
            user_input = input("\n🤖 Ask me anything: ").strip()

            # Check exit conditions
            if user_input.lower() in ["quit", "exit", "q"]:
                logger.info("👋 Goodbye!")
                break

            # Handle special commands
            if user_input.lower() == "docs":
                logger.info("📚 Current documents:")
                for i, doc in enumerate(rag.retriever.docs, 1):
                    logger.info(f"{i}. {doc}")
                continue

            if user_input.lower() == "clear":
                import os

                os.system("cls" if os.name == "nt" else "clear")
                continue

            # Check for empty input
            if not user_input:
                logger.warning("❌ Please enter a question.")
                continue

            logger.info("🔍 Searching for relevant information...")

            # Perform RAG query
            result = rag.query(question=user_input, top_k=TOP_K, max_length=MAX_LENGTH)

            # Display results
            logger.info("📖 Retrieved Documents:")
            if result["retrieved_docs"]:
                for i, doc in enumerate(result["retrieved_docs"], 1):
                    logger.info(f"   {i}. {doc}")
            else:
                logger.info("   No relevant documents found.")

            logger.info("\n💬 Generated Response:")
            logger.info(f"   {result['response']}")

            logger.info("-" * 50)

        except KeyboardInterrupt:
            logger.info("👋 Interrupted by user. Goodbye!")
            break
        except Exception as e:
            logger.error(f"❌ Error occurred: {e}")
            logger.info("Please try again with a different question.")


if __name__ == "__main__":
    main()
