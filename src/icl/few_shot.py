import os
import json
from typing import List, Tuple
from tqdm import tqdm
from multiprocessing import Pool
import numpy as np
from .zero_shot import ZeroShotInference
from utils import *


class FewShotInference(ZeroShotInference):
    def __init__(
        self,
        data_path: str,
        retriever_set_path: str,
        prompt_path: str,
        text_col: str,
        label_col: str,
        id2label_path: str,
        max_workers: int = 4,
        model_type="gpt-3.5-turbo",
    ) -> None:
        super().__init__(
            data_path, prompt_path, text_col, label_col, model_type, max_workers
        )
        # Get the path to the retriever set excluding the filename
        self.retriever_path = retriever_set_path[: retriever_set_path.rfind("/")]
        self.prompt = read_txt(prompt_path)
        self.retriever_set = load_data(retriever_set_path, text_col, label_col)
        self.id2label = read_json(id2label_path)
        self.sentences = [row[0] for row in self.retriever_set]
        self.label_names = [self.id2label[row[1]] for row in self.retriever_set]
        self.retriever_embeddings = self.generate_retriever_embeddings()
        self.query_embeddings = self.generate_query_embeddings()

    def generate_prediction(self, query: Tuple[str, List[float]]) -> str:
        text, query_embedding, k = query
        # Calculate the cosine similarity between the query and each sentence in the retriever set
        similarities = [
            euclidean_distance(query_embedding, retriever_embedding)
            for retriever_embedding in self.retriever_embeddings
        ]
        top_k = sorted(range(len(similarities)), key=lambda i: similarities[i])[:k]
        # Reverse top k
        top_k = top_k[::-1]
        # Print the text, top k most similar sentences and their similarity scores
        examples = "\n\n".join(
            [
                "Text: " + self.sentences[i] + "\nLabel: " + self.label_names[i]
                for i in top_k
            ]
        )
        # Concat the prompt with the examples and text into a single string
        input = self.prompt.format(examples, f"Text: {text}" + "\nLabel: ")
        # Generate the prediction
        prediction = openai_service(input, self.model_type)
        return prediction

    def generate_predictions(self, k: int = 1) -> List[str]:
        # Retrieve the top k most similar sentences from the retriever set with respect to each query
        queries = [
            (text, query_embedding, k)
            for text, query_embedding in zip(self.texts, self.query_embeddings)
        ]
        with Pool() as pool:
            self.predictions = list(
                tqdm(
                    pool.imap(self.generate_prediction, queries, chunksize=1),
                    total=len(queries),
                    desc="Generating predictions",
                )
            )
        return self.predictions

    def generate_retriever_embeddings(self) -> List[List[float]]:
        if os.path.exists(self.retriever_path + "/retriever_embeddings.json"):
            logger.info("Loading retriever embeddings from %s", self.retriever_path)
            with open(self.retriever_path + "/retriever_embeddings.json", "r") as f:
                embeddings = json.load(f)
        else:
            with Pool(self.max_workers) as pool:
                embeddings = list(
                    tqdm(
                        pool.imap(openai_embedding, self.sentences),
                        total=len(self.sentences),
                        desc="Generating retrieval embeddings",
                    )
                )
            with open(self.retriever_path + "/retriever_embeddings.json", "w") as f:
                json.dump(embeddings, f)
        return embeddings

    def generate_query_embeddings(self) -> List[List[float]]:
        if os.path.exists(self.retriever_path + "/query_embeddings.json"):
            logger.info("Loading query embeddings from %s", self.retriever_path)
            with open(self.retriever_path + "/query_embeddings.json", "r") as f:
                embeddings = json.load(f)
        else:
            with Pool(self.max_workers) as pool:
                embeddings = list(
                    tqdm(
                        pool.imap(openai_embedding, self.texts),
                        total=len(self.texts),
                        desc="Generating query embeddings",
                    )
                )
            with open(self.retriever_path + "/query_embeddings.json", "w") as f:
                json.dump(embeddings, f)
        return embeddings
