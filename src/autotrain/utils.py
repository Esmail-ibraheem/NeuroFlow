import json
import os
import subprocess

from commands import launch_command
from trainers.clm.params import LLMTrainingParams
from trainers.dreambooth.params import DreamBoothTrainingParams
from trainers.extractive_question_answering.params import ExtractiveQuestionAnsweringParams
from trainers.generic.params import GenericParams
from trainers.image_classification.params import ImageClassificationParams
from trainers.image_regression.params import ImageRegressionParams
from trainers.object_detection.params import ObjectDetectionParams
from trainers.sent_transformers.params import SentenceTransformersParams
from trainers.seq2seq.params import Seq2SeqParams
from trainers.tabular.params import TabularParams
from trainers.text_classification.params import TextClassificationParams
from trainers.text_regression.params import TextRegressionParams
from trainers.token_classification.params import TokenClassificationParams
from trainers.vlm.params import VLMTrainingParams


ALLOW_REMOTE_CODE = os.environ.get("ALLOW_REMOTE_CODE", "true").lower() == "true"


def run_training(params, task_id, local=False, wait=False):
    params = json.loads(params)
    if isinstance(params, str):
        params = json.loads(params)
    if task_id == 9:
        params = LLMTrainingParams(**params)
    elif task_id == 28:
        params = Seq2SeqParams(**params)
    elif task_id in (1, 2):
        params = TextClassificationParams(**params)
    elif task_id in (13, 14, 15, 16, 26):
        params = TabularParams(**params)
    elif task_id == 27:
        params = GenericParams(**params)
    elif task_id == 25:
        params = DreamBoothTrainingParams(**params)
    elif task_id == 18:
        params = ImageClassificationParams(**params)
    elif task_id == 4:
        params = TokenClassificationParams(**params)
    elif task_id == 10:
        params = TextRegressionParams(**params)
    elif task_id == 29:
        params = ObjectDetectionParams(**params)
    elif task_id == 30:
        params = SentenceTransformersParams(**params)
    elif task_id == 24:
        params = ImageRegressionParams(**params)
    elif task_id == 31:
        params = VLMTrainingParams(**params)
    elif task_id == 5:
        params = ExtractiveQuestionAnsweringParams(**params)
    else:
        raise NotImplementedError

    params.save(output_dir=params.project_name)
    cmd = launch_command(params=params)
    cmd = [str(c) for c in cmd]
    env = os.environ.copy()
    src_dir = os.path.abspath(os.path.join(os.path.dirname(os.path.dirname(__file__))))
    existing_pythonpath = env.get("PYTHONPATH", "")
    if src_dir not in existing_pythonpath.split(os.pathsep):
        env["PYTHONPATH"] = src_dir if not existing_pythonpath else os.pathsep.join([src_dir, existing_pythonpath])
    process = subprocess.Popen(cmd, env=env)
    if wait:
        process.wait()
    return process.pid
