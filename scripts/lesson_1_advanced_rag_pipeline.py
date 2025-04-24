"""
Advanced RAG Pipeline

This script demonstrates building different RAG pipelines with LlamaIndex:
1. Basic Vector Search RAG
2. Sentence Window RAG
3. Hierarchical RAG

Each approach uses a different retrieval mechanism to improve context
relevance and answer quality.
"""

from typing import List, Optional, Dict, Tuple
import os
import warnings
from pathlib import Path

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # Force CPU only mode
warnings.filterwarnings("ignore", category=UserWarning, module="torch.cuda")

import openai
import PyPDF2
from dotenv import load_dotenv

from llama_index.core import Document, VectorStoreIndex, StorageContext, Settings
from llama_index.core.node_parser import (
    SentenceWindowNodeParser,
    HierarchicalNodeParser,
    get_leaf_nodes,
)
from llama_index.core.postprocessor import MetadataReplacementPostProcessor
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.callbacks import CallbackManager, LlamaDebugHandler
from llama_index.llms.openai import OpenAI
from trulens_eval import Tru, Feedback, TruLlama
from trulens_eval.feedback import OpenAI as TruOpenAI


def setup_environment() -> None:
    """
    Set up the environment variables and configurations.
    """
    load_dotenv()

    openai.api_key = os.getenv("OPENAI_API_KEY")
    if not openai.api_key:
        raise ValueError("OPENAI_API_KEY is not set in environment variables or .env file")

    print("OpenAI API key configured successfully.")


def load_pdf_document(file_path: str) -> Document:
    """
    Load a PDF document and convert it to a LlamaIndex Document.
    
    Args:
        file_path: Path to the PDF file
        
    Returns:
        Document object containing the PDF text
    """
    print(f"Loading PDF document from: {file_path}")
    
    with open(file_path, "rb") as f:
        pdf_reader = PyPDF2.PdfReader(f)
        text = ""
        for page_num in range(len(pdf_reader.pages)):
            page = pdf_reader.pages[page_num]
            text += page.extract_text() + "\n"
    
    document = Document(text=text)
    print(f"Created document with {len(text)} characters")
    
    return document


def find_pdf_file(possible_paths: List[str]) -> str:
    """
    Find the PDF file from a list of possible paths.
    
    Args:
        possible_paths: List of possible file paths
        
    Returns:
        Path to the first existing PDF file
        
    Raises:
        FileNotFoundError: If no file is found in any of the paths
    """
    for path in possible_paths:
        if os.path.exists(path):
            return path
    
    raise FileNotFoundError(f"Could not find PDF file in any of these locations: {possible_paths}")


def load_evaluation_questions(file_path: str, custom_questions: Optional[List[str]] = None) -> List[str]:
    """
    Load evaluation questions from a file and add any custom questions.
    
    Args:
        file_path: Path to the file containing questions
        custom_questions: Optional list of additional questions
        
    Returns:
        List of all questions
    """
    print(f"\nLoading evaluation questions from: {file_path}")
    
    questions = []
    with open(file_path, 'r') as file:
        for line in file:
            item = line.strip()
            if item:
                questions.append(item)
                print(f"  - {item}")

    if custom_questions:
        for question in custom_questions:
            questions.append(question)
            print(f"Added custom question: {question}")
    
    print(f"Loaded {len(questions)} total questions")
    return questions


def create_basic_rag(document: Document) -> Tuple[VectorStoreIndex, RetrieverQueryEngine]:
    """
    Create a basic RAG pipeline with vector search.
    
    Args:
        document: The document to index
        
    Returns:
        Tuple of (index, query_engine)
    """
    print("\n## Building basic RAG pipeline...")

    llm = OpenAI(model="gpt-3.5-turbo", temperature=0.1)
    Settings.llm = llm
    Settings.embed_model = "local:BAAI/bge-small-en-v1.5"

    index = VectorStoreIndex.from_documents([document])

    query_engine = index.as_query_engine()
    
    print("Basic RAG pipeline built successfully.")
    return index, query_engine


def create_sentence_window_rag(
    document: Document,
    window_size: int = 3,
    similarity_top_k: int = 6,
    save_dir: str = "sentence_index",
) -> Tuple[VectorStoreIndex, RetrieverQueryEngine]:
    """
    Create a Sentence Window RAG pipeline.
    
    Args:
        document: The document to index
        window_size: Size of context window around each sentence
        similarity_top_k: Number of nodes to retrieve
        save_dir: Directory to save/load the index
        
    Returns:
        Tuple of (index, query_engine)
    """
    print("\n## Building RAG pipeline with Sentence Window approach...")

    node_parser = SentenceWindowNodeParser.from_defaults(
        window_size=window_size,
        window_metadata_key="window",
        original_text_metadata_key="original_text",
    )

    Settings.node_parser = node_parser

    sentence_index = VectorStoreIndex.from_documents([document])

    save_dir_path = Path(save_dir)
    sentence_index.storage_context.persist(persist_dir=str(save_dir_path))
    print(f"Index saved to directory: {save_dir_path.absolute()}")

    retriever = sentence_index.as_retriever(similarity_top_k=similarity_top_k)
    metadata_replacement_postprocessor = MetadataReplacementPostProcessor(target_metadata_key="window")
    
    sentence_window_engine = RetrieverQueryEngine.from_args(
        retriever=retriever,
        node_postprocessors=[metadata_replacement_postprocessor]
    )
    
    print("Sentence Window RAG pipeline built successfully.")
    return sentence_index, sentence_window_engine


def create_hierarchical_rag(
    document: Document,
    chunk_sizes: List[int] = None,
    similarity_top_k: int = 12,
    save_dir: str = "merging_index",
) -> Tuple[VectorStoreIndex, RetrieverQueryEngine]:
    """
    Create a Hierarchical RAG pipeline with multi-level chunking.
    
    Args:
        document: The document to index
        chunk_sizes: List of chunk sizes for hierarchical parsing
        similarity_top_k: Number of nodes to retrieve
        save_dir: Directory to save/load the index
        
    Returns:
        Tuple of (index, query_engine)
    """
    print("\n## Building Hierarchical RAG pipeline...")

    if chunk_sizes is None:
        chunk_sizes = [2048, 512, 128]

    hierarchical_parser = HierarchicalNodeParser.from_defaults(
        chunk_sizes=chunk_sizes
    )

    nodes = hierarchical_parser.get_nodes_from_documents([document])

    leaf_nodes = get_leaf_nodes(nodes)

    storage_context = StorageContext.from_defaults()
    storage_context.docstore.add_documents(nodes)

    Settings.node_parser = hierarchical_parser

    hierarchical_index = VectorStoreIndex(
        leaf_nodes,
        storage_context=storage_context,
    )

    save_dir_path = Path(save_dir)
    hierarchical_index.storage_context.persist(persist_dir=str(save_dir_path))
    print(f"Index saved to directory: {save_dir_path.absolute()}")

    base_retriever = hierarchical_index.as_retriever(similarity_top_k=similarity_top_k)

    debug_handler = LlamaDebugHandler()
    callback_manager = CallbackManager([debug_handler])

    hierarchical_engine = RetrieverQueryEngine.from_args(
        retriever=base_retriever,
        callback_manager=callback_manager,
    )
    
    print("Hierarchical RAG pipeline built successfully.")
    return hierarchical_index, hierarchical_engine


def test_query_engine(query_engine: RetrieverQueryEngine, question: str) -> None:
    """
    Test a query engine with a single question.
    
    Args:
        query_engine: The query engine to test
        question: The question to ask
    """
    print(f"Question: {question}")
    response = query_engine.query(question)
    print(f"Answer: {str(response)}")
    print("-" * 80)


def setup_trulens_evaluation() -> Optional[Tru]:
    """
    Set up TruLens for evaluation if it's available.
    
    Returns:
        Tru instance if TruLens is available
    """
    print("\n## Setting up TruLens for evaluation...")
    tru = Tru()
    tru.reset_database()
    return tru


def create_trulens_recorder(query_engine: RetrieverQueryEngine, app_id: str) -> Optional[TruLlama]:
    """
    Create a TruLens recorder for the given query engine.
    
    Args:
        query_engine: The query engine to evaluate
        app_id: The app ID for TruLens
        
    Returns:
        TruLlama recorder if TruLens is available
    """
    provider = TruOpenAI()

    f_qa_relevance = Feedback(
        provider.relevance, name="Answer Relevance"
    ).on_input_output()

    context_selection = TruLlama.select_source_nodes().node.text
    
    import numpy as np
    f_context_relevance = (
        Feedback(
            provider.relevance, name="Context Relevance"
        )
        .on_input()
        .on(context_selection)
        .aggregate(np.mean)
    )

    recorder = TruLlama(
        query_engine,
        app_id=app_id,
        app_name=app_id,
        feedbacks=[f_qa_relevance, f_context_relevance]
    )
    
    return recorder


def run_trulens_evaluation(
    query_engines: Dict[str, RetrieverQueryEngine],
    eval_questions: List[str],
    tru: Optional[Tru] = None
) -> None:
    """
    Run TruLens evaluation on multiple query engines.
    
    Args:
        query_engines: Dictionary of {engine_name: query_engine}
        eval_questions: List of questions to evaluate
        tru: Tru instance
    """
    print("\n## Running TruLens evaluation...")

    recorders = {}
    for engine_name, engine in query_engines.items():
        recorders[engine_name] = create_trulens_recorder(engine, app_id=engine_name)

    for engine_name, recorder in recorders.items():
        print(f"\nEvaluating {engine_name}...")
        engine = query_engines[engine_name]

        eval_subset = eval_questions[:5]
        for question in eval_subset:
            with recorder:
                engine.query(question)
                print(f"Processed: {question}")

    try:
        print("\nTruLens Leaderboard:")
        app_ids = list(query_engines.keys())
        leaderboard = tru.get_leaderboard(app_ids=app_ids)
        print(leaderboard)
    except Exception as e:
        print(f"Could not generate leaderboard: {str(e)}")

    print("\n## Launching TruLens Dashboard")
    print("Dashboard will be available at: http://localhost:8501/")
    print("If the browser doesn't open automatically, please navigate to the URL above.")

    import threading
    
    def open_dashboard():
        import time
        time.sleep(2)
        print("Browser should open automatically with the TruLens dashboard.")
        print("In the dashboard you will see:")
        print("- Summary of all evaluated systems")
        print("- Detailed statistics for each RAG pipeline")
        print("- Answer and context relevance metrics")
        print("- Comparative analysis of different approaches")
        print("\nNOTE: If you encounter errors in the dashboard, try these troubleshooting steps:")
        print("1. Make sure all questions have been processed before switching tabs")
        print("2. Try clicking 'Refresh' in the dashboard")
        print("3. Look at the Records tab where more data might be available")
    
    # Launch dashboard
    def run_dashboard():
        try:
            tru.run_dashboard()
        except Exception as e:
            print(f"Dashboard error: {str(e)}")
            print("You can manually view results with:")
            print("  from trulens_eval import Tru")
            print("  db = Tru().get_records_and_feedback()")
            print("  print(db)")

    dashboard_thread = threading.Thread(target=run_dashboard)
    dashboard_thread.daemon = True
    dashboard_thread.start()

    browser_thread = threading.Thread(target=open_dashboard)
    browser_thread.daemon = True
    browser_thread.start()
    
    print("\nDashboard is starting... Please wait a few seconds.")
    print("Press Ctrl+C to stop the program and dashboard.")

    try:
        while dashboard_thread.is_alive():
            import time
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nProgram stopped by user.")


def evaluate_query_engines(
    query_engines: Dict[str, RetrieverQueryEngine],
    questions: List[str]
) -> None:
    """
    Evaluate multiple query engines on a list of questions.
    
    Args:
        query_engines: Dictionary of {engine_name: query_engine}
        questions: List of questions to evaluate
    """
    print("\n## Evaluating all RAG pipelines on test questions...")
    
    for engine_name, engine in query_engines.items():
        print(f"\n{engine_name} Results:")
        for question in questions[:3]:  # Only evaluate first 3 questions for brevity
            print(f"Question: {question}")
            response = engine.query(question)
            print(f"Answer: {str(response)}")
            print("-" * 50)
    
    print("\nText evaluation completed!")


def main():
    """Main function to run the RAG pipeline demonstration."""
    setup_environment()

    possible_paths = [
        "books/eBook-How-to-Build-a-Career-in-AI.pdf",
        "eBook-How-to-Build-a-Career-in-AI.pdf"
    ]
    pdf_path = find_pdf_file(possible_paths)
    document = load_pdf_document(pdf_path)

    eval_questions = load_evaluation_questions(
        '../eval_questions.txt',
        custom_questions=["What is the right AI job for me?"]
    )

    _, basic_query_engine = create_basic_rag(document)

    print("\nTesting basic RAG pipeline...")
    test_query_engine(
        basic_query_engine,
        "What are steps to take when finding projects to build your experience?"
    )

    _, sentence_window_engine = create_sentence_window_rag(document)

    print("\nTesting Sentence Window pipeline...")
    test_query_engine(
        sentence_window_engine,
        "how do I get started on a personal project in AI?"
    )
    _, hierarchical_engine = create_hierarchical_rag(document)

    print("\nTesting Hierarchical pipeline...")
    test_query_engine(
        hierarchical_engine,
        "How do I build a portfolio of AI projects?"
    )

    query_engines = {
        "Basic RAG": basic_query_engine,
        "Sentence Window RAG": sentence_window_engine,
        "Hierarchical RAG": hierarchical_engine
    }

    evaluate_query_engines(query_engines, eval_questions)

    tru = setup_trulens_evaluation()
    if tru:
        run_trulens_evaluation(query_engines, eval_questions, tru)


if __name__ == "__main__":
    main()
