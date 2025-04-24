"""
Auto-merging Retrieval

This script demonstrates the implementation of Auto-merging Retrieval
approach in a RAG pipeline. The approach involves:

1. Using hierarchical chunking with multiple levels of granularity
2. Automatically merging relevant chunks based on context
3. Using re-ranking to improve retrieval quality
4. Comparing different hierarchical structures (2 vs 3 layers)

This technique improves RAG performance by retrieving chunks at multiple 
levels of granularity and dynamically merging them to provide a more 
comprehensive context for answering queries.
"""

from typing import List, Optional, Tuple
import os
import warnings
from pathlib import Path

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
warnings.filterwarnings("ignore", category=UserWarning, module="torch.cuda")
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import openai
import PyPDF2
from dotenv import load_dotenv

from llama_index.core import Document, VectorStoreIndex, Settings, StorageContext
from llama_index.core import load_index_from_storage
from llama_index.core.node_parser import HierarchicalNodeParser, get_leaf_nodes
from llama_index.core.retrievers import AutoMergingRetriever
from llama_index.core.postprocessor import SentenceTransformerRerank
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.llms.openai import OpenAI
from trulens_eval import Tru, Feedback, TruLlama
from trulens_eval.feedback import OpenAI as TruOpenAI


def setup_environment() -> None:
    """
    Set up the environment variables and OpenAI API key.
    """
    load_dotenv()

    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        raise ValueError("OPENAI_API_KEY is not set in environment variables or .env file")
    
    openai.api_key = openai_api_key
    print("OpenAI API key configured successfully.")


def find_pdf_file(file_path: str) -> str:
    """
    Find the PDF file by checking both the given path and potential 'books' subdirectory.
    
    Args:
        file_path: Path to the file to look for
        
    Returns:
        Validated path to the PDF file
    """
    if os.path.exists(file_path):
        return file_path

    books_path = os.path.join("books", os.path.basename(file_path))
    if os.path.exists(books_path):
        return books_path

    if not file_path.endswith('.pdf'):
        pdf_path = file_path + '.pdf'
        if os.path.exists(pdf_path):
            return pdf_path
        
        books_pdf_path = os.path.join("books", os.path.basename(pdf_path))
        if os.path.exists(books_pdf_path):
            return books_pdf_path
    
    raise FileNotFoundError(f"Document not found at {file_path} or in the books directory")


def load_document(file_path: str) -> Document:
    """
    Load a document from a file.
    
    Args:
        file_path: Path to the file to load
        
    Returns:
        Document object containing the text
    """
    validated_path = find_pdf_file(file_path)
    print(f"Loading document from: {validated_path}")

    with open(validated_path, "rb") as f:
        pdf_reader = PyPDF2.PdfReader(f)
        text = ""
        for page_num in range(len(pdf_reader.pages)):
            page = pdf_reader.pages[page_num]
            text += page.extract_text() + "\n"
    
    document = Document(text=text)
    print(f"Created document with {len(document.text)} characters")
    return document


def build_automerging_index(
    document: Document,
    chunk_sizes: Optional[List[int]] = None,
    save_dir: str = "merging_index"
) -> VectorStoreIndex:
    """
    Build an auto-merging index for the document with hierarchical chunking.
    
    Args:
        document: The document to index
        chunk_sizes: List of chunk sizes for hierarchical parsing
        save_dir: Directory to save the index
        
    Returns:
        Auto-merging vector store index
    """
    print(f"\n## Building auto-merging index...")
    chunk_sizes = chunk_sizes or [2048, 512, 128]
    print(f"Using chunk sizes: {chunk_sizes}")

    llm = OpenAI(model="gpt-3.5-turbo", temperature=0.1)
    Settings.llm = llm
    Settings.embed_model = "local:BAAI/bge-small-en-v1.5"

    node_parser = HierarchicalNodeParser.from_defaults(chunk_sizes=chunk_sizes)

    nodes = node_parser.get_nodes_from_documents([document])
    leaf_nodes = get_leaf_nodes(nodes)
    
    print(f"Generated {len(nodes)} total nodes, {len(leaf_nodes)} leaf nodes")

    save_dir_path = Path(save_dir)

    storage_context = StorageContext.from_defaults()
    storage_context.docstore.add_documents(nodes)
    
    if not save_dir_path.exists():
        print(f"Creating new index at {save_dir_path}")
        automerging_index = VectorStoreIndex(leaf_nodes, storage_context=storage_context)
        automerging_index.storage_context.persist(persist_dir=str(save_dir_path))
        print(f"Index saved to directory: {save_dir_path.absolute()}")
    else:
        print(f"Loading existing index from {save_dir_path}")
        storage_context = StorageContext.from_defaults(persist_dir=str(save_dir_path))
        automerging_index = load_index_from_storage(storage_context)
    
    return automerging_index


def create_automerging_query_engine(
    index: VectorStoreIndex,
    similarity_top_k: int = 12,
    rerank_top_n: int = 6
) -> RetrieverQueryEngine:
    """
    Create a query engine for the auto-merging index.
    
    Args:
        index: The vector store index to query
        similarity_top_k: Number of top similar nodes to retrieve
        rerank_top_n: Number of top nodes to keep after reranking
        
    Returns:
        Query engine configured for auto-merging retrieval with reranking
    """
    print(f"Creating auto-merging query engine with top_k={similarity_top_k}, rerank_top_n={rerank_top_n}")

    base_retriever = index.as_retriever(similarity_top_k=similarity_top_k)

    retriever = AutoMergingRetriever(
        base_retriever, 
        index.storage_context, 
        verbose=True
    )

    reranker = SentenceTransformerRerank(
        top_n=rerank_top_n, 
        model="BAAI/bge-reranker-base"
    )

    query_engine = RetrieverQueryEngine.from_args(
        retriever=retriever,
        node_postprocessors=[reranker]
    )
    
    return query_engine


def setup_trulens_evaluation() -> Optional[Tru]:
    """
    Set up TruLens for evaluation if it's available.
    
    Returns:
        Tru instance if TruLens is available
    """
    print("\nSetting up TruLens evaluation...")
    tru = Tru()
    tru.reset_database()
    return tru


def create_feedback_functions(provider: TruOpenAI) -> Tuple[Feedback, Feedback, Feedback]:
    """
    Create feedback functions for the RAG triad metrics.
    
    Args:
        provider: TruLens provider for evaluations
        
    Returns:
        Tuple of (answer_relevance, context_relevance, groundedness) feedback functions
    """
    print("Creating feedback functions for evaluation...")

    f_qa_relevance = Feedback(
        provider.relevance_with_cot_reasons,
        name="Answer Relevance"
    ).on_input_output()

    context_selection = TruLlama.select_source_nodes().node.text
    
    f_qs_relevance = (
        Feedback(provider.qs_relevance_with_cot_reasons,
                name="Context Relevance")
        .on_input()
        .on(context_selection)
        .aggregate(np.mean)
    )

    f_groundedness = (
        Feedback(provider.groundedness_measure_with_cot_reasons,
                 name="Groundedness")
        .on(context_selection)
        .on_output()
    )
    
    return f_qa_relevance, f_qs_relevance, f_groundedness


def create_trulens_recorder(
    query_engine: RetrieverQueryEngine,
    feedback_functions: List[Feedback],
    app_id: str = "auto_merging_engine"
) -> Optional[TruLlama]:
    """
    Create a TruLens recorder for evaluating the query engine.
    
    Args:
        query_engine: The query engine to evaluate
        feedback_functions: List of feedback functions to use
        app_id: ID for the application in TruLens
        
    Returns:
        TruLens recorder if TruLens is available
    """
    print(f"Creating TruLens recorder with app_id: {app_id}")
    
    tru_recorder = TruLlama(
        query_engine,
        app_id=app_id,
        feedbacks=feedback_functions
    )
    
    return tru_recorder


def load_evaluation_questions(
    file_path: str = 'generated_questions.text',
    custom_questions: Optional[List[str]] = None
) -> List[str]:
    """
    Load evaluation questions from a file and optionally add custom questions.
    
    Args:
        file_path: Path to the file containing questions
        custom_questions: Optional list of additional questions
        
    Returns:
        List of evaluation questions
    """
    print(f"Loading evaluation questions from {file_path}")
    
    eval_questions = []
    try:
        with open(file_path, 'r') as file:
            for line in file:
                item = line.strip()
                if item:
                    eval_questions.append(item)
        
        if not eval_questions:
            print(f"Warning: No questions found in {file_path}")
            
        if custom_questions:
            for question in custom_questions:
                eval_questions.append(question)
            
        print(f"Loaded {len(eval_questions)} questions")
        for i, q in enumerate(eval_questions):
            print(f"  {i+1}. {q}")
        
        if not eval_questions:

            default_question = "What is the importance of networking in AI?"
            eval_questions.append(default_question)
            print(f"Added default question: {default_question}")
            
    except FileNotFoundError:
        print(f"Warning: Questions file {file_path} not found, using default questions")
        if custom_questions:
            eval_questions = custom_questions
        else:
            eval_questions = ["What is the importance of networking in AI?"]
            
    return eval_questions


def run_evaluation(
    tru_recorder: Optional[TruLlama],
    query_engine: RetrieverQueryEngine,
    questions: List[str],
    tru: Optional[Tru] = None
) -> None:
    """
    Run evaluation on a list of questions.
    
    Args:
        tru_recorder: TruLens recorder
        query_engine: Query engine to evaluate
        questions: List of questions to evaluate
        tru: Tru instance for recording results
    """
    print(f"\nRunning evaluation on {len(questions)} questions...")
    
    if tru_recorder is None or tru is None:
        print("TruLens not available, running without evaluation...")
        for question in questions:
            print(f"Processing question: '{question}'")
            response = query_engine.query(question)
            print(f"Response: {response.response}\n")
        return
    
    for question in questions:
        print(f"Evaluating question: '{question}'")
        with tru_recorder:
            query_engine.query(question)

    records, feedback = tru.get_records_and_feedback(app_ids=[])
    print("\nEvaluation records:")
    print(records.head())

    pd.set_option("display.max_colwidth", None)
    feedback_df = records[["input", "output"] + feedback]
    print("\nDetailed feedback:")
    print(feedback_df)

    print("\nLeaderboard:")
    print(tru.get_leaderboard(app_ids=[]))


def compare_chunk_structures(
    document: Document,
    eval_questions: List[str],
    chunk_structures: List[List[int]] = [[2048, 512], [2048, 512, 128]],
    tru: Optional[Tru] = None
) -> None:
    """
    Compare different chunk structures in the auto-merging approach.
    
    Args:
        document: Document to index
        eval_questions: List of evaluation questions
        chunk_structures: List of chunk size lists to compare
        tru: Tru instance for recording results
    """
    print(f"\n## Comparing chunk structures:")
    for i, chunk_sizes in enumerate(chunk_structures):
        print(f"Structure {i+1}: {chunk_sizes}")
    
    for i, chunk_sizes in enumerate(chunk_structures):
        print(f"\n### Testing chunk structure {i+1}: {chunk_sizes}")

        save_dir = f"merging_index_{i}"
        automerging_index = build_automerging_index(
            document,
            chunk_sizes=chunk_sizes,
            save_dir=save_dir
        )

        query_engine = create_automerging_query_engine(automerging_index)

        provider = TruOpenAI()
        feedback_functions = create_feedback_functions(provider)

        tru_recorder = create_trulens_recorder(
            query_engine,
            feedback_functions,
            app_id=f'auto_merging_engine_{i}'
        )

        run_evaluation(tru_recorder, query_engine, eval_questions, tru)


def main() -> None:
    """
    Main function to run the auto-merging retrieval demonstration.
    """
    try:
        setup_environment()

        tru = setup_trulens_evaluation()

        document_path = "books/eBook-How-to-Build-a-Career-in-AI.pdf"
        document = load_document(document_path)

        eval_questions = load_evaluation_questions(
            file_path="generated_questions.text",
            custom_questions=["What is the importance of networking in AI?"]
        )

        chunk_structures = [
            [2048, 512],
            [2048, 512, 128]
        ]
        compare_chunk_structures(document, eval_questions, chunk_structures, tru)

        print("\nLaunching TruLens dashboard...")
        tru.run_dashboard()

        
    except Exception as e:
        print(f"Error in main function: {e}")


if __name__ == "__main__":
    main()
