# Retrieval-Augmented Generation with GPT

🚀 A Retrieval-Augmented Generation (RAG) pipeline using **Hugging Face Transformers**.  
This project demonstrates how to combine **retrieval** with **Hugging Face models** for question answering.

## ✨ Features

- 🔎 **Retrieval-Augmented Generation (RAG)** with FAISS vector store
- � **Hugging Face Transformers** for model inference
- 📊 End-to-end demo: query → retrieval → model inference → answer

## 📌 Motivation

Modern LLM-based applications require not only accurate answers (via RAG)

## Installation

```bash
$ git clone https://github.com/CheyuWu/Retrieval-Augmented-Generation-with-GPT.git
$ cd Retrieval-Augmented-Generation-with-GPT
$ pip install -r requirements.txt
```

## Setup Your Documents and Models

1. Place your documents in the `src/data` directory.
2. Update the `DATA_PATH` variable in `src/config/gpt_config.py` to point to your document file.
3. (Optional) Change the retriever and GPT-2 model in `src/config/gpt_config.py`:
   - `RETRIEVER_MODEL`: e.g., `"all-MiniLM-L6-v2"`
   - `GPT2_MODEL`: e.g., `"gpt2-medium"`, `"gpt2-large"`, `"gpt2-xl"`
4. (Optional) Adjust `TOP_K` and `MAX_LENGTH` in `src/config/gpt_config.py` for retrieval and generation settings.

## Launch the Demo

> You can change the model and dataset in the `main.py` file

```bash
$ python main.py
```

## Results

### Sample Output

```console
=== RAG System with GPT-2 ===
Initializing system...
Use pytorch device_name: cuda:0
Load pretrained SentenceTransformer: all-MiniLM-L6-v2
Loading GPT-2 model: gpt2-medium
Model loaded on device: cuda
Batches: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 4/4 [00:00<00:00, 13.22it/s]
Added 119 documents to the retriever.
RAG System is ready!
You can ask questions about the documents.
Type 'quit', 'exit', or 'q' to stop.
Type 'docs' to see all documents.
Type 'clear' to clear the screen.

🤖 Ask me anything: What is the capital of France?
🔍 Searching for relevant information...
Batches: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:00<00:00, 41.08it/s]
/home/user/miniconda3/envs/llm/lib/python3.13/site-packages/transformers/generation/configuration_utils.py:679: UserWarning: `num_beams` is set to 1. However, `early_stopping` is set to `True` -- this flag is only used in beam-based generation modes. You should set `num_beams>1` or unset `early_stopping`.
  warnings.warn(
The attention mask is not set and cannot be inferred from input because pad token is same as eos token. As a consequence, you may observe unexpected behavior. Please pass your input's `attention_mask` to obtain reliable results.
📖 Retrieved Documents:
   1. The Bash shell is a command-line interpreter for Unix systems.
   2. The attention mechanism allows models to focus on relevant parts of the input sequence.
   3. The LangChain framework helps build applications with LLMs and external data sources.

💬 Generated Response:
   Paris
In this question, you can also answer: A capital city is an international city with its own name, or a city that is part of a continent. If you answer the second question correctly, then you are considered to be a French person. (Note: In the first question you have to provide the correct answer, but the answers to all the questions are valid).
 Question : What kind of person do
--------------------------------------------------
```
