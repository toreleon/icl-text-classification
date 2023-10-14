import os
import json
from typing import List
from tqdm import tqdm
from multiprocessing import Pool

# from scipy.spatial.distance import cdist
from utils import *
from .zero_shot import ZeroShotInference


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
        self.distances = self.generate_distances()

    def generate_distances(self) -> np.ndarray:
        # Calculate the distances between all pairs of query and retriever embeddings
        distances = dist(self.query_embeddings, self.retriever_embeddings)
        return distances

    def generate_prompts(self, k: int) -> List[str]:
        if k == 0:
            return [
                self.prompt.format(f"Text: {text}" + "\nLabel: ", "").strip() for text in self.texts
            ]
        else:
            prompts = []
            for idx in tqdm(range(len(self.distances)), desc="Generating prompts"):
                text = self.texts[idx]
                similarities = self.distances[idx]
                top_k = sorted(range(len(similarities)), key=lambda i: similarities[i])[
                    :k
                ]
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
                prompt = self.prompt.format(examples, f"Text: {text}" + "\nLabel: ")
                prompts.append(prompt)
            return prompts

    def generate_predictions(self, k: int) -> List[str]:
        self.prompts = self.generate_prompts(k)
        inputs = [(prompt, self.model_type) for prompt in self.prompts]
        logger.info(inputs[231][1])
        logger.info(
            f"Generating predictions for {len(self.prompts)} prompts with {k}-shots"
        )
        with Pool(self.max_workers) as pool:
            self.predictions = list(
                tqdm(
                    pool.imap(openai_service, inputs[231], chunksize=1),
                    total=len(inputs),
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
            with Pool(4) as pool:
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
            with Pool(4) as pool:
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
