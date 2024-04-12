from antiberty import AntiBERTyRunner
import torch
import logging

from tqdm import tqdm
from numpy import ndarray
from biotrainer.embedders import CustomEmbedder
from typing import Generator, Iterable, Optional

logger = logging.getLogger(__name__)


class AntiBERTyEmbedder(CustomEmbedder):
    """
    pip installed from https://pypi.org/project/antiberty/
    paper: https://arxiv.org/abs/2112.07782
    """

    name: str = "antiberty_embedder"

    def embed_many(
            self, sequences: Iterable[str], batch_size: Optional[int] = None
    ) -> Generator[ndarray, None, None]:
        """
        Method to embed all sequences from the provided iterable.
        This is the function that should be overwritten by most custom embedders, because it allows full control
        over the whole embeddings generation process. Other functions are optional to use and overwrite, except
        reduce_per_protein (if necessary).

        Yields embedding for one sequence at a time.

        :param sequences: List of proteins as AA strings
        :param batch_size: For embedders that profit from batching, this is maximum number of AA per batch

        :return: A list object with embeddings of the sequences.
        """

        logger.info(f"AntiBERTy: Embedding protein sequences!")

        model = AntiBERTyRunner()
        
        for seq in sequences:
            yield model.embed([seq])[0][1:-1, :]

    @staticmethod
    def reduce_per_protein(embedding: ndarray) -> ndarray:
        return torch.mean(torch.tensor(embedding).squeeze(), dim=0).numpy()
