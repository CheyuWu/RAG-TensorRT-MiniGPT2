# RAG-TensorRT-MiniGPT2

🚀 A Retrieval-Augmented Generation (RAG) pipeline built with **PyTorch** and accelerated using **NVIDIA TensorRT**.  
This project demonstrates how to combine **custom Transformer (GPT2-style) layers** with **TensorRT inference optimization**,  
providing both functional correctness and performance benchmarks.

## ✨ Features
- 🔎 **Retrieval-Augmented Generation (RAG)** with FAISS vector store  
- 🧠 **Custom GPT2-style Transformer block** implemented in PyTorch  
- 🔄 **ONNX export** and **TensorRT engine building** (FP32/FP16)  
- ⚡ **Performance evaluation** comparing PyTorch vs TensorRT inference (latency, throughput, memory)  
- 📊 End-to-end demo: query → retrieval → model inference → answer  

## 📌 Motivation
Modern LLM-based applications require not only accurate answers (via RAG),  
but also **low-latency and efficient inference**.  
This project explores how **TensorRT optimizations** can significantly improve inference performance for Transformer-based models.

