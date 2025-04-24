"""
RAG Triad of Metrics

This script demonstrates the evaluation of a RAG system using three key metrics:
1. Answer Relevance - How well the answer addresses the query
2. Context Relevance - How well the retrieved context relates to the query
3. Groundedness - How well the answer is grounded in the retrieved context

The evaluation is performed using TruLens, which provides detailed feedback
and visualization of the RAG pipeline performance.
"""

from typing import List, Optional, Tuple
import os
import warnings
from pathlib import Path

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # Force CPU only mode
warnings.filterwarnings("ignore", category=UserWarning, module="torch.cuda")
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import openai
import PyPDF2
from dotenv import load_dotenv

from llama_index.core import Document, VectorStoreIndex, Settings
from llama_index.core.node_parser import SentenceWindowNodeParser
from llama_index.core.postprocessor import MetadataReplacementPostProcessor
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


def load_document(file_path: str) -> Document:
    """
    Load a document from a file.
    
    Args:
        file_path: Path to the file to load
        
    Returns:
        Document object containing the text
    """
    print(f"Loading document from: {file_path}")

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Document not found at {file_path}")

    with open(file_path, "rb") as f:
        pdf_reader = PyPDF2.PdfReader(f)
        text = ""
        for page_num in range(len(pdf_reader.pages)):
            page = pdf_reader.pages[page_num]
            text += page.extract_text() + "\n"
    
    document = Document(text=text)
    print(f"Created document with {len(document.text)} characters")
    return document


def build_sentence_window_index(
    document: Document,
    window_size: int = 3,
    similarity_top_k: int = 6,
    save_dir: str = "sentence_index"
) -> Tuple[VectorStoreIndex, RetrieverQueryEngine]:
    """
    Build a sentence window index for the document.
    
    Args:
        document: The document to index
        window_size: Size of the context window around each sentence
        similarity_top_k: Number of nodes to retrieve
        save_dir: Directory to save the index
        
    Returns:
        Tuple of (index, query_engine)
    """
    print(f"\n## Building sentence window index with window size {window_size}...")

    llm = OpenAI(model="gpt-3.5-turbo", temperature=0.1)
    Settings.llm = llm
    Settings.embed_model = "local:BAAI/bge-small-en-v1.5"

    node_parser = SentenceWindowNodeParser.from_defaults(
        window_size=window_size,
        window_metadata_key="window",
        original_text_metadata_key="original_text"
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


def test_query(query_engine: RetrieverQueryEngine, question: str) -> str:
    """
    Test the query engine with a question.
    
    Args:
        query_engine: The query engine to test
        question: The question to ask
        
    Returns:
        The response from the query engine
    """
    print(f"\nTesting query engine with question: '{question}'")
    response = query_engine.query(question)
    print(f"Response: {response.response}")
    
    return response.response


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
    print("Creating feedback functions for the RAG triad metrics...")

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
    app_id: str = "App_1"
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
    file_path: str = 'eval_questions.txt',
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
    with open(file_path, 'r') as file:
        for line in file:
            item = line.strip()
            if item:
                eval_questions.append(item)
    
    if custom_questions:
        for question in custom_questions:
            eval_questions.append(question)
    
    print(f"Loaded {len(eval_questions)} questions")
    for i, q in enumerate(eval_questions):
        print(f"  {i+1}. {q}")
    
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


def main() -> None:
    """
    Main function to run the RAG evaluation with the triad of metrics.
    """
    try:
        setup_environment()

        tru = setup_trulens_evaluation()

        document_path = "books/eBook-How-to-Build-a-Career-in-AI.pdf"
        document = load_document(document_path)

        sentence_index, sentence_window_engine = build_sentence_window_index(
            document,
            window_size=3,
            similarity_top_k=6,
            save_dir="sentence_index"
        )

        test_query(
            sentence_window_engine, 
            "How do you create your AI portfolio?"
        )
        
        if tru is not None:
            provider = TruOpenAI()

            f_qa_relevance, f_qs_relevance, f_groundedness = create_feedback_functions(provider)

            tru_recorder = create_trulens_recorder(
                sentence_window_engine,
                [f_qa_relevance, f_qs_relevance, f_groundedness]
            )

            eval_questions = load_evaluation_questions(
                custom_questions=["How can I be successful in AI?"]
            )

            run_evaluation(tru_recorder, sentence_window_engine, eval_questions, tru)

            print("\nLaunching TruLens dashboard...")
            tru.run_dashboard()
        else:
            eval_questions = load_evaluation_questions(
                custom_questions=["How can I be successful in AI?"]
            )
            run_evaluation(None, sentence_window_engine, eval_questions)
        
    except Exception as e:
        print(f"Error in main function: {e}")


if __name__ == "__main__":
    main()
