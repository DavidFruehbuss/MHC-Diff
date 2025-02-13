import argparse
from argparse import Namespace
from pathlib import Path
import yaml
import os
import time
import pickle
import gzip

import os
import sys

import torch
import pytorch_lightning as pl
from torch_scatter import scatter_add

if __name__ == "__main__":

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    desired_directory = '/gpfs/home4/dfruhbus/MHC-Diff/'
    os.chdir(desired_directory)
    sys.path.insert(0, desired_directory)
    from model.lightning_module import Structure_Prediction_Model
    from dataset_8k_xray import PDB_Dataset

    # read in config
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    args_dict = args.__dict__
    for key, value in config.items():
        if isinstance(value, dict):
            args_dict[key] = Namespace(**value)
        else:
            args_dict[key] = value

    num_samples = args.num_samples
    sample_batch_size = args.sample_batch_size
    sample_savepath = args.sample_savepath

    lightning_model = Structure_Prediction_Model.load_from_checkpoint(
                    args.checkpoint,
                    dataset=args.dataset,
                    data_dir=args.data_dir,
                    dataset_params=args.dataset_params,
                    task_params=args.task_params,
                    generative_model=args.generative_model,
                    generative_model_params=args.generative_model_params,
                    architecture=args.architecture,
                    network_params=args.network_params,
                    batch_size=args.batch_size,
                    lr=args.lr,
                    num_workers=args.num_workers,
                    device=args.device,
    )   

    lightning_model = lightning_model.to(device)
    lightning_model.setup('test')
    test_dataset = lightning_model.test_dataset

    ## calculate the test_dataset variance
    var = []
    for data in test_dataset:
        pos = data['peptide_positions'].to(torch.float32)
        var += [torch.sum((pos - torch.mean(pos, dim=0))**2, dim=0) / len(pos)]
    dataset_variance = sum(var) / len(var)
    print(dataset_variance)

    results = []
    saved_samples = {}
    saved_samples['graph_name'] = [data['graph_name'] for data in test_dataset]
    saved_samples['x_target'] = {}
    saved_samples['x_predicted'] = {}
    saved_samples['h'] = {}
    saved_samples['rmse'] = []
    saved_samples['rmse_mean'] = []
    saved_samples['rmse_best'] = []

    start_time_total = time.time()

    print(len(test_dataset))
    # temporary for spedtest ToDo: Modify Back

    for i in range(0, len(test_dataset), sample_batch_size):

        if i + sample_batch_size >= len(test_dataset): continue

        start_time = time.time()

        # prepare peptide-MHC
        mol_pro_list = [test_dataset[i+j] for _ in range(num_samples) for j in range(sample_batch_size)]

        mol_pro_samples = PDB_Dataset.collate_fn(mol_pro_list)

        # sample new peptide-MHC structures using trained model
        mol_pro_batch = lightning_model.get_molecule_and_protein(mol_pro_samples)
        molecule, protein_pocket = mol_pro_batch

        print(f'Prior to sampling time {time.time() - start_time}')

        xh_mol_final, xh_pro_final = lightning_model.model.sample_structure(num_samples, molecule, protein_pocket, args.sampling_without_noise, args.data_dir, args.run_name)

        print(f'After sampling time {time.time() - start_time}')

        # Safe resulting structures
        size_tuple = tuple(molecule['size'].tolist())
        true_pos = [torch.split(molecule['x'], size_tuple, dim=0)[i*num_samples] for i in range(sample_batch_size)] # [sample_batch_size, num_nodes, 3]
        true_h = [torch.split(molecule['h'], size_tuple, dim=0)[i*num_samples] for i in range(sample_batch_size)] # [sample_batch_size, num_nodes, 3]
        for j in range(sample_batch_size):
            key = i+j
            saved_samples['x_target'][key] = true_pos[j]
            # [num_all_nodes, 3] -> [sample_batch_size * samples, num_nodes, 3] -> [sample_batch_size, samples, num_nodes, 3]
            saved_samples['x_predicted'][key] = torch.split(xh_mol_final[:,:3], size_tuple, dim=0)[j*num_samples:(j+1)*num_samples]
            saved_samples['h'][key] = true_h[j]
        # Goal structure ['x_predicted']: [sample_key][10 * [9,3]], ['x_target']: [sample_key][1 * [9,3]]

        # Calculate the RMSE error
        error_mol = scatter_add(torch.sum((molecule['x'] - xh_mol_final[:,:3])**2, dim=-1), molecule['idx'], dim=0)
        rmse = torch.sqrt(error_mol / (molecule['size']))

        rmse_sample_mean = [rmse[j*num_samples:(j+1)*num_samples].mean(0) for j in range(sample_batch_size)]
        rmse_sample_best = [rmse[j*num_samples:(j+1)*num_samples].min(0)[0] for j in range(sample_batch_size)]

        end_time = time.time()

        saved_samples['rmse'] += [rmse[j*num_samples:(j+1)*num_samples] for j in range(sample_batch_size)]
        print(len(saved_samples['rmse']), rmse.shape)
        saved_samples['rmse_mean'] += [rmse_sample_mean[j] for j in range(sample_batch_size)]
        saved_samples['rmse_best'] += [rmse_sample_best[j] for j in range(sample_batch_size)]

        print(f'Time: {end_time - start_time}')

    end_time_total = time.time()
    time_total = end_time_total - start_time_total

    saved_samples['rmse_mean'] = torch.stack(saved_samples['rmse_mean'], dim=0)
    saved_samples['rmse_best'] = torch.stack(saved_samples['rmse_best'], dim=0)
    rmse_mean = saved_samples['rmse_mean'].mean(0)
    rmse_best = saved_samples['rmse_best'].mean(0)

    print(saved_samples['rmse_mean'])
    print(saved_samples['rmse_best'])

    print(f'Mean RMSE across all mean/best sample: mean {round(rmse_mean.item(),3)}, best {round(rmse_best.item(),3)}')
    print(f'This took {time_total} seconds for 1000*10 samples')

    start_time_saving = time.time()

    # Serialize dictionary with pickle
    pickled_data = pickle.dumps(saved_samples)

    # Compress pickled data
    with gzip.open(f'{sample_savepath}.pkl.gz', 'wb') as f:
        f.write(pickled_data)

    end_time_saving = time.time()
    time_saving = end_time_saving - start_time_saving
    print(f'Time to save data: {time_saving} s')

