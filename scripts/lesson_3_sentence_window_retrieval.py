"""
Sentence Window Retrieval

This script demonstrates the implementation of a Sentence Window Retrieval
approach in a RAG pipeline. The approach involves:

1. Parsing text into sentences
2. Creating context windows around each sentence
3. Using metadata replacement for effective retrieval
4. Optional re-ranking to improve retrieval quality
5. Comparing different window sizes (1 vs 3)

This technique improves RAG performance by providing more context around 
retrieved snippets and allowing for more precise relevance scoring.
"""

from typing import List, Dict, Any, Optional, Tuple
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

from llama_index.core import Document, VectorStoreIndex, Settings, StorageContext
from llama_index.core import load_index_from_storage
from llama_index.core.node_parser import SentenceWindowNodeParser
from llama_index.core.schema import NodeWithScore, TextNode, QueryBundle
from llama_index.core.postprocessor import MetadataReplacementPostProcessor, SentenceTransformerRerank
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


def build_sentence_window_index(
    document: Document,
    window_size: int = 3,
    save_dir: str = "sentence_index"
) -> Tuple[VectorStoreIndex, Dict[str, Any]]:
    """
    Build a sentence window index for the document.
    
    Args:
        document: The document to index
        window_size: Size of the context window around each sentence
        save_dir: Directory to save the index
        
    Returns:
        Tuple of (index, context)
    """
    print(f"\n## Building sentence window index with window size {window_size}...")

    llm = OpenAI(model="gpt-3.5-turbo", temperature=0.1)

    Settings.llm = llm
    Settings.embed_model = "local:BAAI/bge-small-en-v1.5"

    node_parser = SentenceWindowNodeParser.from_defaults(
        window_size=window_size,
        window_metadata_key="window",
        original_text_metadata_key="original_text",
    )

    Settings.node_parser = node_parser

    save_dir_path = Path(save_dir)
    if not save_dir_path.exists():
        print(f"Creating new index at {save_dir_path}")
        sentence_index = VectorStoreIndex.from_documents([document])
        sentence_index.storage_context.persist(persist_dir=str(save_dir_path))
        print(f"Index saved to directory: {save_dir_path.absolute()}")
    else:
        print(f"Loading existing index from {save_dir_path}")
        storage_context = StorageContext.from_defaults(persist_dir=str(save_dir_path))
        sentence_index = load_index_from_storage(storage_context)

    context = {
        "llm": llm,
        "embed_model": Settings.embed_model,
        "node_parser": node_parser
    }
    
    return sentence_index, context


def create_sentence_window_query_engine(
    index: VectorStoreIndex,
    similarity_top_k: int = 6,
    rerank_top_n: int = 2
) -> RetrieverQueryEngine:
    """
    Create a query engine for the sentence window index.
    
    Args:
        index: The vector store index to query
        similarity_top_k: Number of top similar nodes to retrieve
        rerank_top_n: Number of top nodes to keep after reranking
        
    Returns:
        Query engine configured for sentence window retrieval with reranking
    """
    print(f"Creating sentence window query engine with top_k={similarity_top_k}, rerank_top_n={rerank_top_n}")

    metadata_replacement_postprocessor = MetadataReplacementPostProcessor(
        target_metadata_key="window"
    )

    reranker = SentenceTransformerRerank(
        top_n=rerank_top_n, 
        model="BAAI/bge-reranker-base"
    )

    retriever = index.as_retriever(similarity_top_k=similarity_top_k)
    query_engine = RetrieverQueryEngine.from_args(
        retriever=retriever,
        node_postprocessors=[metadata_replacement_postprocessor, reranker]
    )
    
    return query_engine

def demo_sentence_parsing() -> None:
    """
    Demonstrate how the sentence window parser works.
    """
    print("\n## Demonstrating sentence window parsing...")

    node_parser = SentenceWindowNodeParser.from_defaults(
        window_size=3,
        window_metadata_key="window",
        original_text_metadata_key="original_text",
    )

    example_text = "hello. how are you? I am fine!  "
    nodes = node_parser.get_nodes_from_documents([Document(text=example_text)])
    
    print("Example 1: Parsing text with different sentence endings")
    print("Original text:", example_text)
    print("Extracted sentences:", [x.text for x in nodes])
    print("Window for node[1]:", nodes[1].metadata["window"])

    example_text_2 = "hello. foo bar. cat dog. mouse"
    nodes_2 = node_parser.get_nodes_from_documents([Document(text=example_text_2)])
    
    print("\nExample 2: Parsing text with consistent sentence endings")
    print("Original text:", example_text_2)
    print("Extracted sentences:", [x.text for x in nodes_2])
    print("Window for node[0]:", nodes_2[0].metadata["window"])


def demo_reranker() -> None:
    """
    Demonstrate how the reranker works.
    """
    print("\n## Demonstrating reranker functionality...")

    reranker = SentenceTransformerRerank(
        top_n=2, model="BAAI/bge-reranker-base"
    )

    query = QueryBundle("I want a dog.")
    scored_nodes = [
        NodeWithScore(node=TextNode(text="This is a cat"), score=0.6),
        NodeWithScore(node=TextNode(text="This is a dog"), score=0.4),
    ]

    reranked_nodes = reranker.postprocess_nodes(
        scored_nodes, query_bundle=query
    )
    
    print("Original nodes with scores:")
    for i, node in enumerate(scored_nodes):
        print(f"{i+1}. {node.text} (score: {node.score:.2f})")
    
    print("\nReranked nodes with scores:")
    for i, node in enumerate(reranked_nodes):
        print(f"{i+1}. {node.text} (score: {node.score:.2f})")


def setup_trulens_evaluation() -> Optional[Tru]:
    """
    Set up TruLens for evaluation if it's available.
    
    Returns:
        Tru instance if TruLens is available, None otherwise
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
    app_id: str = "sentence_window_engine"
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
                if item:  # Skip empty lines
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
            # Add a default question if no questions were loaded
            default_question = "What are the keys to building a career in AI?"
            eval_questions.append(default_question)
            print(f"Added default question: {default_question}")
            
    except FileNotFoundError:
        print(f"Warning: Questions file {file_path} not found, using default questions")
        if custom_questions:
            eval_questions = custom_questions
        else:
            eval_questions = ["What are the keys to building a career in AI?"]
            
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
    
    # Display evaluation results
    records, feedback = tru.get_records_and_feedback(app_ids=[])
    print("\nEvaluation records:")
    print(records.head())

    pd.set_option("display.max_colwidth", None)
    feedback_df = records[["input", "output"] + feedback]
    print("\nDetailed feedback:")
    print(feedback_df)

    print("\nLeaderboard:")
    print(tru.get_leaderboard(app_ids=[]))


def compare_window_sizes(
    document: Document,
    eval_questions: List[str],
    window_sizes: List[int] = [1, 3],
    tru: Optional[Tru] = None
) -> None:
    """
    Compare different window sizes in the sentence window approach.
    
    Args:
        document: Document to index
        eval_questions: List of evaluation questions
        window_sizes: List of window sizes to compare
        tru: Tru instance for recording results
    """
    print(f"\n## Comparing window sizes: {window_sizes}")
    
    for window_size in window_sizes:
        print(f"\n### Testing window size = {window_size}")

        save_dir = f"sentence_index_{window_size}"
        sentence_index, _ = build_sentence_window_index(
            document,
            window_size=window_size,
            save_dir=save_dir
        )

        query_engine = create_sentence_window_query_engine(sentence_index)

        provider = TruOpenAI()
        feedback_functions = create_feedback_functions(provider)

        tru_recorder = create_trulens_recorder(
            query_engine,
            feedback_functions,
            app_id=f'sentence_window_engine_{window_size}'
        )

        run_evaluation(tru_recorder, query_engine, eval_questions, tru)



def main() -> None:
    """
    Main function to run the sentence window retrieval demonstration.
    """
    try:
        setup_environment()

        tru = setup_trulens_evaluation()

        document_path = "books/eBook-How-to-Build-a-Career-in-AI.pdf"
        document = load_document(document_path)

        demo_sentence_parsing()

        demo_reranker()

        eval_questions = load_evaluation_questions(
            file_path="generated_questions.text",
            custom_questions=["What are the keys to building a career in AI?"]
        )

        compare_window_sizes(document, eval_questions, [1, 3], tru)

        print("\nLaunching TruLens dashboard...")
        tru.run_dashboard()
        
    except Exception as e:
        print(f"Error in main function: {e}")


if __name__ == "__main__":
    main()
