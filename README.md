# Advanced RAG Techniques

This repository contains a collection of scripts demonstrating advanced Retrieval-Augmented Generation (RAG) techniques using LlamaIndex and OpenAI. 
Each lesson focuses on a specific aspect of modern RAG systems, providing hands-on examples and evaluation methods and evaluation capabilities using **TruLens**.

## 📚 Overview

The scripts in this repository showcase a progression of increasingly sophisticated RAG approaches:

1. **Basic Vector Search** → **Sentence Window** → **Hierarchical Chunking** → **Auto-merging Retrieval**
2. **Evaluation** using the **RAG Triad of Metrics**: Answer Relevance, Context Relevance, and Groundedness

## 📊 Integrated Evaluation System

Each lesson integrates robust evaluation capabilities using **TruLens**, allowing you to:

- **Quantitatively measure RAG performance** across different architectures
- **Visualize metrics** through an interactive dashboard
- **Compare different approaches** side-by-side using consistent metrics

The evaluation system analyzes each RAG architecture using the RAG Triad metrics:
1. **Answer Relevance**: How well the response addresses the query
2. **Context Relevance**: How well retrieved passages match the query
3. **Groundedness**: How faithfully the response represents retrieved information

This evaluation framework helps identify which RAG approach works best for different types of queries and content, 
enabling data-driven decisions when designing production RAG systems.

## 🧩 Repository Structure

### Lesson 1: Advanced RAG Pipeline

**Purpose:** Demonstrates building and comparing different RAG architectures with LlamaIndex.

**Key Components:**
- 📄 Document loading and preprocessing with PyPDF2
- 🔍 Three retrieval architectures implemented side-by-side
- 📊 Comprehensive evaluation system using TruLens
- ⚙️ Configuration management with environment variables

**What You'll Learn:**
- How to build a basic Vector Search RAG pipeline
- How to implement Sentence Window retrieval for improved context awareness
- How to create a Hierarchical RAG system with multi-level chunking
- How to compare and evaluate different RAG approaches using quantitative metrics
- How to structure a complex RAG application with clean separation of concerns

### Lesson 2: RAG Triad of Metrics

**Purpose:** Explores systematic evaluation of RAG performance using three critical metrics.

**Key Components:**
- 📏 TruLens evaluation framework integration
- 🎯 Custom feedback functions for each metric
- 🔍 Sentence window retrieval implementation
- 📊 Detailed visualization of evaluation results

**What You'll Learn:**
- How to measure **Answer Relevance** - whether the response addresses the query
- How to evaluate **Context Relevance** - whether retrieved passages match the query
- How to assess **Groundedness** - whether the answer is supported by context
- How to create custom evaluation pipelines for RAG systems
- How to interpret and visualize RAG performance metrics
- How to use evaluation results to improve RAG system design

### Lesson 3: Sentence Window Retrieval

**Purpose:** Deep dive into the Sentence Window technique for context-aware retrieval.

**Key Components:**
- 📑 SentenceWindowNodeParser for context-aware text chunking
- 🔄 MetadataReplacementPostProcessor for enhancing retrieval context
- 🏆 SentenceTransformerRerank for improving result relevance
- 📊 Comparative analysis of different window sizes

**What You'll Learn:**
- How sentence window parsing creates contextual overlaps between chunks
- How to configure optimal window sizes for different content types
- How reranking can enhance the quality of retrieved passages
- How metadata replacement provides more complete context
- How to compare different window size configurations (1 vs 3)
- The impact of context window size on RAG performance metrics

### Lesson 4: Auto-merging Retrieval

**Purpose:** Demonstrates the Auto-merging approach for dynamically combining chunks at different granularity levels.

**Key Components:**
- 📚 HierarchicalNodeParser for multi-level document chunking
- 🔀 AutoMergingRetriever for dynamic context aggregation
- ⚙️ Customizable chunk size configurations
- 📊 Comparative evaluation of different hierarchical structures

**What You'll Learn:**
- How hierarchical chunking creates a tree structure of document segments
- How to configure optimal chunk sizes for hierarchical parsing
- How auto-merging dynamically combines relevant chunks across granularity levels
- How to compare different hierarchical structures (2 vs 3 layers)
- The benefits of retrieving at multiple levels of granularity
- How to evaluate and optimize hierarchical RAG systems
- When to use Auto-merging Retrieval versus other RAG techniques

## 🛠️ Getting Started

### Prerequisites

- Python 3.8+
- OpenAI API key
- PyPDF2
- LlamaIndex
- TruLens

### Installation

```bash
# Install required packages
pip install -r requirements.txt

# Set up your OpenAI API key
# Either create a .env file with OPENAI_API_KEY=your-api-key
# Or export it in your shell:
export OPENAI_API_KEY=your-api-key