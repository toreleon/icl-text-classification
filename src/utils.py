import os
import csv
import json
from tqdm.auto import tqdm
import logging
import requests
from typing import Dict, List, Any, Deque, Tuple
import time
from collections import deque
import numpy as np
import pandas as pd
import timeout_decorator
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score
import openai

tqdm.pandas()

# Configure logging for INFO, WARNING, ERROR from the root
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


class CognitiveService:
    def __init__(self, endpoint, token, max_requests_per_minute):
        self.endpoint = endpoint
        self.token = token
        self.max_requests_per_minute = max_requests_per_minute
        self.last_request_time = time.time()
        self.request_queue: Deque[float] = deque()

    def can_make_request(self):
        current_time = time.time()
        while self.request_queue and current_time - self.request_queue[0] > 60:
            self.request_queue.popleft()
        return len(self.request_queue) < self.max_requests_per_minute

    def made_request(self):
        self.request_queue.append(time.time())


CHATGPT_COGNITIVE_SERVICES = [
    CognitiveService(
        os.getenv("CHATGPT_ENDPOINT_1"), os.getenv("CHATGPT_TOKEN_1"), 240
    ),
    CognitiveService(
        os.getenv("CHATGPT_ENDPOINT_2"), os.getenv("CHATGPT_TOKEN_2"), 240
    ),
    CognitiveService(
        os.getenv("CHATGPT_ENDPOINT_3"), os.getenv("CHATGPT_TOKEN_3"), 240
    ),
]

GPT_4_COGNITIVE_SERVICES = [
    CognitiveService(os.getenv("GPT_4_ENDPOINT_1"), os.getenv("GPT_4_TOKEN_1"), 60),
    CognitiveService(os.getenv("GPT_4_ENDPOINT_2"), os.getenv("GPT_4_TOKEN_2"), 60),
    CognitiveService(os.getenv("GPT_4_ENDPOINT_3"), os.getenv("GPT_4_TOKEN_3"), 60),
]


@timeout_decorator.timeout(120)
def openai_request(prompt: str, model: str = "gpt-3.5-turbo") -> str:
    data = json.dumps(
        {
            "model": model,
            "messages": [{"role": "system", "content": prompt}],
            "temperature": 0.0,
            "top_p": 1.0,
            "frequency_penalty": 0.0,
            "presence_penalty": 0.0,
        }
    )
    headers = {
        "Content-Type": "application/json",
        "Authorization": "Bearer " + os.getenv("OPENAI_API_KEY"),
    }
    response = requests.post(os.getenv("OPENAI_ENDPOINT"), headers=headers, data=data)
    return response.json()


@timeout_decorator.timeout(120)
def openai_request_azure(prompt: str, cognitive_service: Dict[str, Any]) -> str:
    data = json.dumps(
        {
            "messages": [{"role": "system", "content": prompt}],
            "temperature": 0.0,
            "top_p": 1.0,
            "frequency_penalty": 0.0,
            "presence_penalty": 0.0,
        }
    )
    headers = {
        "Content-Type": "application/json",
        "api-key": cognitive_service["token"],
    }
    response = requests.post(cognitive_service["endpoint"], headers=headers, data=data)
    return response.json()


def openai_service(input: Tuple[str, str], max_retries: int = 3) -> str:
    prompt, model_type = input
    if model_type == "gpt-3.5-turbo":
        cognitive_services = CHATGPT_COGNITIVE_SERVICES
        model = "gpt-3.5-turbo-0301"
    elif model_type == "gpt-4":
        cognitive_services = GPT_4_COGNITIVE_SERVICES
        model = "gpt-4-0314"
    service = None
    try:
        for service in cognitive_services:
            if not service.can_make_request():
                logger.info("Service rate limit reached. Retrying with next service.")
                continue
            response = openai_request_azure(
                prompt, {"endpoint": service.endpoint, "token": service.token}
            )
            if "error" in response.keys():
                if response["error"]["code"] == "content_filter":
                    logger.info(
                        "The prompt violates Azure Cognitive service's content policy. Try to request to OpenAI API instead."
                    )
                    response = openai_request(prompt, model=model)
                else:
                    continue
            service.made_request()
            # logger.info(response)
            return response["choices"][0]["message"]["content"]
    except Exception as e:
        if max_retries > 0:
            logger.info(input)
            logger.info(f"Retrying... {max_retries} retries left")
            return openai_service(prompt, model_type, max_retries - 1)
    logger.info(f"Failed to generate response for prompt: {prompt}")
    return None  # if all retries failed


def read_json(path: str) -> Dict:
    with open(path, "r") as f:
        return json.load(f)


def read_txt(path: str) -> str:
    with open(path, "r") as f:
        return f.read()


@timeout_decorator.timeout(120)
def openai_embedding(text: str, max_retries: int = 3) -> List[float]:
    """
    Make a request to OpenAI API with the given prompt.
    """
    try:
        response = openai.Embedding.create(
            model="text-embedding-ada-002",
            input=text,
        )
        return response.data[0].embedding
    except Exception as e:
        if max_retries > 0:
            logger.info(e)
            logger.info(f"Retrying... {max_retries} retries left")
            return openai_embedding(text, max_retries - 1)
        else:
            logger.info(f"Failed to generate embedding for prompt: {text}")
            return None


def load_data(path: str, text_col: str, label_col: str) -> List[List[str]]:
    data = []
    with open(path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            data.append([row[text_col], row[label_col]])
    return data


def euclidean_distance(u: np.ndarray, v: np.ndarray, w: np.ndarray = None) -> float:
    if w is not None:
        # Compute weighted Euclidean distance
        distance = np.sqrt(np.sum(w * np.square(u - v), dtype=np.float64))
    else:
        # Compute Euclidean distance
        distance = np.linalg.norm(u - v).astype(np.float64)
    return distance


def dist(XA: List[List[float]], XB: List[List[float]]) -> List[List[float]]:
    # Ensure the inputs are NumPy arrays
    XA = np.array(XA, dtype=np.float64)
    XB = np.array(XB, dtype=np.float64)

    # Get the shapes of the input arrays
    m, n = XA.shape
    p, q = XB.shape

    # Ensure the input arrays have the same number of columns
    if n != q:
        raise ValueError("XA and XB must have the same number of columns")

    # Initialize an empty array to hold the output with 32 bit floating point precision
    D = np.empty((m, p), dtype=np.float64)

    # Compute the Euclidean distance between each pair of points
    for i in tqdm(range(m), desc="Computing distances"):
        for j in range(p):
            D[i, j] = euclidean_distance(XA[i], XB[j])
    return D.tolist()


def clean_prediction_label(prediction: str, label2id: Dict[str, int]) -> str:
    """
    Clean the prediction label by removing the label prefix and suffix.
    """
    prediction = prediction.lower().strip()
    prediction = prediction.rstrip("\n")
    for label, id in label2id.items():
        if label in prediction:
            return id
    else:
        return -1


def evaluate(
    path: str, prediction_col: str = "prediction", label2id_path: str = None
) -> Dict:
    """
    Evaluate the predictions in the given CSV file using accuracy, precision, recall, F1-score (macro and weighted), and ROC AUC.
    """
    label2id = read_json(label2id_path)
    label2id = {label.lower(): id for label, id in label2id.items()}
    labels = list(label2id.keys())
    df = pd.read_csv(path)
    df["prediction_id"] = df[prediction_col].progress_apply(
        lambda prediction: clean_prediction_label(prediction, label2id)
    )
    df.to_csv("test.csv", index=False)
    # read the labels and predictions
    labels = df["label"].tolist()
    predictions = df["prediction_id"].tolist()
    if -1 in predictions:
        logger.info(f"Some predictions are not in the label2id mapping for {path} in line {predictions.index(-1)}")
    # Compute metrics
    acc = accuracy_score(labels, predictions)
    f1_macro = f1_score(labels, predictions, average="macro")
    f1_weighted = f1_score(labels, predictions, average="weighted")
    # roc_auc = roc_auc_score(labels, predictions, average="macro", multi_class="ovo")
    # Return metrics
    return {
        "accuracy": acc,
        "f1_macro": f1_macro,
        "f1_weighted": f1_weighted,
        # "roc_auc": roc_auc,
    }


if __name__ == "__main__":
    prompt = "this is a test"
    response = openai_service(prompt, model_type="gpt-4")
    print(response)
