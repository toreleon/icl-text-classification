import os
import logging.config
from icl.few_shot import FewShotInference
from utils import evaluate

# load the logging configuration from logging.ini
logging.config.fileConfig("src/logging.ini", disable_existing_loggers=False)
logger = logging.getLogger(__name__)


class CFG:
    dataset: str = "uit-victsd"
    text_col: str = "comment"
    label_col: str = "constructiveness"
    max_workers: int = 17
    model_type: str = "gpt-3.5-turbo"
    prompt_path: str = f"src/icl/prompts/{dataset}-{label_col}-few-shot.txt"
    id2label_path: str = f"data/{dataset}/constructiveness_id2label.json"
    label2id_path: str = f"data/{dataset}/constructiveness_label2id.json"


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

for k in [30]:
    model.generate_predictions(k)
    model.save_predictions(
        f"results/{CFG.dataset}-{CFG.label_col}-{CFG.model_type}-{k}-shot.csv"
    )
    print(
        f'{CFG.dataset}-{CFG.label_col}-{CFG.model_type}-{k}-shot:{evaluate(f"results/{CFG.dataset}-{CFG.label_col}-{CFG.model_type}-{k}-shot.csv","prediction", CFG.label2id_path)}'
    )

# print(
#     f'{CFG.dataset}-{CFG.label_col}-{CFG.model_type}-0-shot:{evaluate(f"results/{CFG.dataset}-{CFG.label_col}-{CFG.model_type}-0-shot.csv","prediction", CFG.label2id_path)}'
# )


# files = os.listdir("results/")
# # Sort files by name
# files.sort()
# for file in files:
#     # Write the results to the results.txt
#     if file.endswith(".csv"):
#         if "uit-vsmec" in file:
#             label2id_path = "data/uit-vsmec/emotion_label2id.json"
#         elif "uit-vsfc" in file:
#             if "sentiment" in file:
#                 label2id_path = "data/uit-vsfc/sentiment_label2id.json"
#             else:
#                 label2id_path = "data/uit-vsfc/topic_label2id.json"
#         with open("results.txt", "a") as f:
#             f.write(file.replace(".csv", "") + str(evaluate(
#                     f"results/{file}",
#                     "prediction",
#                     label2id_path,
#                 ))
#             )
#             f.write("\n\n")