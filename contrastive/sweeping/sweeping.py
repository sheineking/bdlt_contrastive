import wandb
import json
import sweeping_configs as s

import sys
sys.path.append("../../contrastive")
import learning_manager as l

# ===================================================================
# Preparation
# ===================================================================
SWEEPING_CONFIGS = json.load(open(s.CONFIG_PATH))       # Get the configurations for all sweeps
SWEEPING_NAMES = ["Pairwise_SGD", "Pairwise_RMS"]       # Define which sweeps to run
MAX_NUM_RUNS = 50                                       # Define how many runs should be made for one sweep

default_hyperparameters = {
    "use_wandb": False,
    "train_mode": "pairwise",
    "model_name": "Pairwise_SGD",
    "epochs": 10,
    "batch_size": 2,
    "optimizer_name": "rmsprop",
    "lr": 0.1,
    "momentum": 0,
    "weight_decay": 0,
    "alpha": 0.99,
    "eps": 1e-08,
    "trust_coef": 0.001
}


# ===================================================================
# Sweeping
# ===================================================================
def run_single_sweep(use_wandb, train_mode, model_name, epochs, batch_size, optimizer_name, lr, momentum, weight_decay,
                     alpha, eps, trust_coef):

    # Define the learning manager
    Learning_Manager = l.LearningManager(train_mode=train_mode, model_name=model_name, use_wandb=use_wandb)

    # Conduct training with the given configuration
    Learning_Manager.conduct_training(epochs=epochs, batch_size=batch_size, optimizer_name=optimizer_name, lr=lr,
                                      momentum=momentum, weight_decay=weight_decay, alpha=alpha, eps=eps,
                                      trust_coef=trust_coef)



def sweep_agent():
    wandb.init(config=default_hyperparameters, project='big_data_lang_tech', entity='zebby')

    hyperparameters = dict(wandb.config)
    # if running inside a sweep agent, this contains the default_hyperparameters with the adjustments
    # made by the sweep controller.

    hyperparameters.update({"use_wandb": True})

    run_single_sweep(**hyperparameters)



if __name__ == "__main__":
    for name in SWEEPING_NAMES:
        # Obtain the sweeping config for the given name
        sweeping_config = SWEEPING_CONFIGS[name]["sweep_config"]

        # Get the sweep id and run the sweep (MAX_NUM_RUNS-times)
        print('\n\n')
        sweep_id = wandb.sweep(sweeping_config, project='big_data_lang_tech')

        wandb.agent(sweep_id=sweep_id,
                    function=sweep_agent,
                    count=MAX_NUM_RUNS)