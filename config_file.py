from pathlib import Path

def get_config():
    return {
        "batch_size": 6,
        "precision": "16-mixed",
        "accelerator": "cuda",
        "progress_bar_refresh_rate": 10,
        "num_epochs": 20,
        "num_iter": 100,
        "lr": 10**-4,
        "seq_len": 350,
        "d_model": 512,
        "lang_src": "en",
        "lang_tgt": "it",
        "model_folder": "weights",
        "ckpt_path": "default",
        "model_basename": "tmodel_",
        "preload": True,
        "tokenizer_file": "tokenizer_{0}.json",
        "experiment_name": "runs/tmodel",
        "src_vocab_size": 0,
        "tgt_vocab_size": 0}

def get_weights_file_path(config, epoch: str):
    model_folder = config["model_folder"]
    model_basename = config["model_basename"]
    model_filename = f"{model_basename}{epoch}.pt"
    return str(Path('.') / model_folder / model_filename)