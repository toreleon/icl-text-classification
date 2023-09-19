import logging.config
from icl.few_shot import FewShotInference
from icl.zero_shot import ZeroShotInference
from utils import evaluate

# load the logging configuration from logging.ini
logging.config.fileConfig("src/logging.ini", disable_existing_loggers=False)
logger = logging.getLogger(__name__)


class CFG:
    K: int = 3
    prompt_path: str = "src/icl/prompts/uit-vsfc-sentiment-few-shot.txt"
    dataset: str = "uit-vsfc"
    text_col: str = "sentence"
    label_col: str = "sentiment"
    max_workers: int = 8
    model_type: str = "gpt-4"
    id2label_path: str = "data/uit-vsfc/sentiment_id2label.json"
    label2id_path: str = "data/uit-vsfc/sentiment_label2id.json"


# model = ZeroShotInference(
#     data_path=f"data/{CFG.dataset}/{CFG.dataset}-test.csv",
#     prompt_path="src/icl/prompts/uit-vsfc-topic-zeroshot.txt",
#     text_col=CFG.text_col,
#     label_col=CFG.label_col,
#     max_workers=CFG.max_workers,
#     model_type=CFG.model_type,
# )

# model.generate_predictions()
# model.save_predictions(
#     f"results/{CFG.dataset}-{CFG.label_col}-{CFG.model_type}-zero-shot-trial-0.csv"
# )

model = FewShotInference(
    data_path=f"data/{CFG.dataset}/{CFG.dataset}-test.csv",
    retriever_set_path=f"data/{CFG.dataset}/{CFG.dataset}-train.csv",
    prompt_path=CFG.prompt_path,
    text_col=CFG.text_col,
    label_col=CFG.label_col,
    id2label_path=CFG.id2label_path,
    max_workers=CFG.max_workers,
    model_type=CFG.model_type,
)

model.generate_predictions(k=CFG.K)
model.save_predictions(
    f"results/{CFG.dataset}-{CFG.label_col}-{CFG.model_type}-{CFG.K}-shot.csv"
)


print(
    evaluate(
        f"results/{CFG.dataset}-{CFG.label_col}-{CFG.model_type}-{CFG.K}-shot.csv",
        "prediction",
        CFG.label2id_path,
    )
)
