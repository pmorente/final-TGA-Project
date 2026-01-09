import time
import logging
from typing import List, Optional, Literal, Tuple
from sentence_transformers import SentenceTransformer
import numpy as np

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


class EmbeddingGenerator:
    def __init__(
        self, 
        model_name: Literal[
            "sentence-transformers/all-mpnet-base-v2",
            "sentence-transformers/all-MiniLM-L6-v2",
            "sentence-transformers/all-roberta-large-v1"
        ] = "sentence-transformers/all-MiniLM-L6-v2"
    ):
        """
        Initialize the embedding model.
        
        Args:
            model_name: One of the supported model names:
                - "sentence-transformers/all-mpnet-base-v2"
                - "sentence-transformers/all-MiniLM-L6-v2"
                - "sentence-transformers/all-roberta-large-v1"
        """
        logger.info(f"Initializing EmbeddingGenerator with model: {model_name}")
        logger.info("Loading model (this may take a while if downloading for the first time)...")
        load_start = time.time()
        self.model = SentenceTransformer(model_name)
        load_time = time.time() - load_start
        logger.info(f"Model loaded successfully in {load_time:.2f} seconds")
        self.model_name = model_name

    def encode(
        self,
        sentences: List[str],
        mode: Literal["sequential", "batch"] = "sequential",
        batch_size: Optional[int] = None,
        show_progress_bar: bool = True
    ) -> Tuple[np.ndarray, float]:
        """
        Encode sentences into embeddings with different strategies.

        Args:
            sentences (List[str]): Input sentences.
            mode (str): "sequential" or "batch".
            batch_size (int, optional): Batch size for batch mode.
            show_progress_bar (bool): Show progress bar during encoding.

        Returns:
            Tuple[np.ndarray, float]: (Embeddings matrix, elapsed time in seconds)
        """
        start_time = time.time()

        if mode == "sequential":
            # Encode one by one
            embeddings = [self.model.encode([s]) for s in sentences]
            embeddings = np.vstack(embeddings)

        elif mode == "batch":
            if not batch_size:
                raise ValueError("You must provide a batch_size for batch mode.")
            embeddings = self.model.encode(
                sentences,
                batch_size=batch_size,
                show_progress_bar=show_progress_bar
            )

        else:
            raise ValueError("Invalid mode. Choose from 'sequential' or 'batch'.")

        elapsed_sec = time.time() - start_time
        print(f"Encoding completed in {elapsed_sec:.5f} seconds using mode={mode}")

        return embeddings, elapsed_sec