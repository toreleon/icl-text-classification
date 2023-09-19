import logging
import csv
from functools import partial
from typing import List
from tqdm import tqdm
from multiprocessing import Pool
from utils import *

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ZeroShotInference:
    def __init__(
        self,
        data_path: str,
        prompt_path: str,
        text_col: str,
        label_col: str,
        model_type: str = "gpt-3.5-turbo",
        max_workers: int = 4,
    ) -> None:
        self.prompt = read_txt(prompt_path)
        self.data = load_data(data_path, text_col, label_col)
        self.texts = [row[0] for row in self.data]
        self.labels = [row[1] for row in self.data]
        self.predictions = []
        self.max_workers = max_workers
        if model_type == "gpt-3.5-turbo" or model_type == "gpt-4":
            self.model_type = model_type
            logger.info(f"Using model: {self.model_type}")
        else:
            raise ValueError("Model must be either gpt-3.5-turbo or gpt-4")

    def generate_predictions(self) -> List[str]:
        inputs = [self.prompt.format(text) for text in self.texts]
        logger.info(f"Generating predictions for {len(inputs)} inputs")
        with Pool(self.max_workers) as pool:
            self.predictions = list(
                tqdm(
                    pool.imap(
                        partial(openai_service, model_type=self.model_type),
                        inputs,
                        chunksize=1,
                    ),
                    total=len(inputs),
                    desc="Generating predictions",
                )
            )
        return self.predictions

    def save_predictions(self, path: str) -> None:
        with open(path, "w") as f:
            writer = csv.writer(f)
            writer.writerow(["text", "label", "prediction"])
            for text, label, prediction in zip(
                self.texts, self.labels, self.predictions
            ):
                writer.writerow([text, label, prediction])

    def sample_prediction(self, num_samples: int = 10) -> None:
        for text, label in zip(self.texts[:num_samples], self.labels[:num_samples]):
            # Generate the prediction for the given text
            input = self.prompt.format(text)
            prediction = openai_service(input, model_type=self.model)
            logger.info(f"Prompt: {text}")
            logger.info(f"Text: {text}")
            logger.info(f"Label: {label}")
            logger.info(f"Prediction: {prediction}")
