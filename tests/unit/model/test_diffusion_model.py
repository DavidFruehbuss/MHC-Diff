import os
from uuid import uuid4
import random

import pandas
import torch

from model.diffusion_model import Conditional_Diffusion_Model


def test_save_rmsds():

    run_id = f"test-{uuid4()}"
    table_path = f"{run_id}-rmsds.csv"

    n = 10
    r = 100

    ts = list(reversed(range(r)))
    try:
        for t in ts:
            Conditional_Diffusion_Model._save_rmsds(run_id, t, torch.rand(n), n * ["test"])

        data = pandas.read_csv(table_path)

        assert data.shape[0] == r
        assert data.shape[1] == (n + 1)

        # should overwrite
        random.shuffle(ts)
        for t in ts:
            Conditional_Diffusion_Model._save_rmsds(run_id, t, torch.rand(n), n * ["test~"])

        data = pandas.read_csv(table_path)

        assert data.shape[0] == r
        assert data.shape[1] == (2 * n + 1)

    finally:
        if os.path.isfile(table_path):
            os.remove(table_path)
