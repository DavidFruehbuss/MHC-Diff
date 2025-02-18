import os
from uuid import uuid4

import pandas
import torch

from model.diffusion_model import Conditional_Diffusion_Model


def test_save_rmsds():

    run_id = f"test-{uuid4()}"
    table_path = f"{run_id}-rmsds.csv"

    n = 10

    try:
        Conditional_Diffusion_Model._save_rmsds(run_id, 0, torch.rand(n), n * ["test"])

        data = pandas.read_csv(table_path)

        assert data.shape[0] == 1
        assert data.shape[1] == (n + 1)

    finally:
        if os.path.isfile(table_path):
            os.remove(table_path)
