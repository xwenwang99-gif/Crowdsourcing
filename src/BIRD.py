# -*- coding: utf-8 -*-
"""
BIRD.py -- loader for the Bluebirds dataset (Welinder et al., NIPS 2010),
as distributed with Zhang et al., "Spectral Methods Meet EM" (github.com/zhangyuc/SpectralMethodsMeetEM).

108 images, 39 workers, 4,212 labels, 2 classes (Indigo Bunting vs Blue Grosbeak;
60 / 48). Fully dense: every worker labels every image.

Expected files (labels already shifted from 1/2 to 0/1):
    data/bird_answer.csv   columns: question, worker, answer
    data/bird_truth.csv    columns: question, truth

Returns the same tuple as get_DOG() / get_FACE().
"""

import os
from src.FACE import load_crowd_csv

_HERE = os.path.dirname(os.path.abspath(__file__))
_DEFAULT_DIR = os.path.join(_HERE, "..", "data")


def get_BIRD(data_dir=_DEFAULT_DIR):
    return load_crowd_csv(os.path.join(data_dir, "bird_answer.csv"),
                          os.path.join(data_dir, "bird_truth.csv"))
