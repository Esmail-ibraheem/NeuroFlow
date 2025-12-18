"""
Copyright 2023 The HuggingFace Team
"""

from dataclasses import dataclass
from typing import List, Union

from backends.base import AVAILABLE_HARDWARE
from backends.endpoints import EndpointsRunner
from backends.local import LocalRunner
from backends.ngc import NGCRunner
from backends.nvcf import NVCFRunner
from backends.spaces import SpaceRunner
from trainers.clm.params import LLMTrainingParams
from trainers.dreambooth.params import DreamBoothTrainingParams
from trainers.image_classification.params import ImageClassificationParams
from trainers.image_regression.params import ImageRegressionParams
from trainers.object_detection.params import ObjectDetectionParams
from trainers.sent_transformers.params import SentenceTransformersParams
from trainers.seq2seq.params import Seq2SeqParams
from trainers.tabular.params import TabularParams
from trainers.text_classification.params import TextClassificationParams
from trainers.text_regression.params import TextRegressionParams
from trainers.token_classification.params import TokenClassificationParams


@dataclass
class AutoTrainProject:
    params: Union[
        List[
            Union[
                LLMTrainingParams,
                TextClassificationParams,
                TabularParams,
                DreamBoothTrainingParams,
                Seq2SeqParams,
                ImageClassificationParams,
                TextRegressionParams,
                ObjectDetectionParams,
                TokenClassificationParams,
                SentenceTransformersParams,
                ImageRegressionParams,
            ]
        ],
        LLMTrainingParams,
        TextClassificationParams,
        TabularParams,
        DreamBoothTrainingParams,
        Seq2SeqParams,
        ImageClassificationParams,
        TextRegressionParams,
        ObjectDetectionParams,
        TokenClassificationParams,
        SentenceTransformersParams,
        ImageRegressionParams,
    ]
    backend: str

    def __post_init__(self):
        if self.backend not in AVAILABLE_HARDWARE:
            raise ValueError(f"Invalid backend: {self.backend}")

    def create(self):
        if self.backend.startswith("local"):
            runner = LocalRunner(params=self.params, backend=self.backend)
            return runner.create()
        elif self.backend.startswith("spaces-"):
            runner = SpaceRunner(params=self.params, backend=self.backend)
            return runner.create()
        elif self.backend.startswith("ep-"):
            runner = EndpointsRunner(params=self.params, backend=self.backend)
            return runner.create()
        elif self.backend.startswith("ngc-"):
            runner = NGCRunner(params=self.params, backend=self.backend)
            return runner.create()
        elif self.backend.startswith("nvcf-"):
            runner = NVCFRunner(params=self.params, backend=self.backend)
            return runner.create()
        else:
            raise NotImplementedError
