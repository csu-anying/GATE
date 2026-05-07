import pickle
import random
import torch
import yaml
import wandb
import numpy as np

from models.Graph_text_model import ECG_Graph_Text_with_co, ECG_Graph_Text_only_time
from pretrain.pretrain_trainer import trainer_Graph_Text
from utils.dataset import ECG_Dsataset


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    print(f"Running on device {device}.")

    # Load configuration
    config = yaml.load(open("config.yaml", "r"), Loader=yaml.FullLoader)

    # Initialize wandb only on the main process (rank 0)
    run = wandb.init(
        mode="offline",  # offline online
        project="ECG_SSL_Graph",
        name=config['wandb_name'],
        config={
            "learning_rate": config['optimizer']['params']['lr'],
            "total_epochs": config['trainer']['max_epochs'],
            'weight_decay': config['optimizer']['params']['weight_decay'],
            'ecg_model': config['network']['ecg_model'],
            'text_model': config['network']['text_model'],
            'batch_size': config['trainer']['batch_size'],
            'val_zeroshot': 'all_sets',
            'prompt_type': config['zeroshot']['prompt_type'],
        }
    )

    # Set random seeds
    torch.manual_seed(42)
    random.seed(0)
    np.random.seed(0)

    # Loading data
    data_path = config['dataset']['data_path']
    dataset = ECG_Dsataset(data_path=data_path, dataset_name=config['dataset']['dataset_name'])
    train_dataset = dataset.get_dataset(train_test='train')
    val_dataset = dataset.get_dataset(train_test='val')

    # Building model
    model = ECG_Graph_Text_with_co(config['network'])
    # model = ECG_Graph_Text_only_time(config['network'])
    # Load co-occurrence embeddings and set them in the model
    with open('../extra_graph/co_occurrence_new.pkl', 'rb') as f:
        co_embeddings = pickle.load(f)
    model.co_occurrence_emb(co_embeddings)

    # Optionally freeze layers of the BERT model
    if config['network']['free_layers'] is not None:
        for layer_idx in range(int(config['network']['free_layers'])):
            for param in list(model.lm_model.encoder.layer[layer_idx].parameters()):
                param.requires_grad = False

    model = model.to(device)  # Move model to the right device (CPU/GPU)
    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model, device_ids=[0, 1, 2, 3])

    # Optimizer setup
    optimizer = torch.optim.AdamW(
        model.parameters(),
        **config['optimizer']['params'],
        betas=(0.9, 0.999)
    )
    # Create trainer instance
    trainer = trainer_Graph_Text(model=model,
                                 optimizer=optimizer,
                                 device=device,
                                 model_name=config['wandb_name'],
                                 config_optim=config['optimizer']['params'],
                                 **config['trainer'])

    # Train the model
    trainer.train_graph_text(train_dataset, val_dataset, config['zeroshot'])


if __name__ == "__main__":
    main()
