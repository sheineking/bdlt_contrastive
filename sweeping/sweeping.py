import wandb
import json
import os
import sys
import argparse

import sweeping_configs as s

# ===================================================================
# Preparation
# ===================================================================
SWEEPING_CONFIGS = json.load(open(s.CONFIG_PATH))  # Get the configurations for all sweeps
MAX_NUM_RUNS = 50                                  # How many runs should be made for one sweep

def switch_directory(folder_name='../contrastive/'):
    abs_path = os.path.abspath(folder_name + "learning_manager.py")
    dir_name = os.path.dirname(abs_path)

    sys.path.append(dir_name)
    os.chdir(dir_name)


# ===================================================================
# Sweeping functions
# ===================================================================
def run_sweep_contrastive(use_wandb, train_mode, model_name, epochs, batch_size, optimizer_name, lr, momentum,
                          weight_decay, alpha, eps, trust_coef):

    # Define the learning manager
    Learning_Manager = l.LearningManager(train_mode=train_mode, model_name=model_name, use_wandb=use_wandb)

    # Conduct training with the given configuration
    Learning_Manager.conduct_training(epochs=epochs, batch_size=batch_size, optimizer_name=optimizer_name, lr=lr,
                                      momentum=momentum, weight_decay=weight_decay, alpha=alpha, eps=eps,
                                      trust_coef=trust_coef)

def run_sweep_supervised(use_wandb, model_name, epochs, batch_size, optimizer_name, lr, momentum, weight_decay,
                         alpha, eps, trust_coef):

    # Define the learning manager
    Learning_Manager = l.LearningManager(model_name=model_name, use_wandb=use_wandb)

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

    sweep_func = run_sweep_supervised if mode == "supervised" else run_sweep_contrastive
    sweep_func(**hyperparameters)



# ===================================================================
# Main functions
# ===================================================================
if __name__ == "__main__":
    global mode
    global default_hyperparameters

    # Define the mode (contrastive or supervised) for which to run the sweeps
    parser = argparse.ArgumentParser(description='Hyperparameter sweeping')
    parser.add_argument('--mode', metavar='mode', type=str, required=True,
                        help='"contrastive" or "supervised". For which type to run sweeps.')
    args = parser.parse_args()

    # Switch directory to import the correct modules
    mode = args.mode
    if mode == "supervised":
        switch_directory('../supervised/')
    elif mode == "contrastive":
        switch_directory('../contrastive/')
    else:
        print("Please provide a valid mode: 'contrastive' or 'supervised ")
        exit(1)

    # Import the correct learning manager
    import learning_manager as l

    # Get the correct dictionary based on the mode and set the global hyperparameters
    mode_configs = SWEEPING_CONFIGS[mode]
    default_hyperparameters = mode_configs.pop("default_hyperparameters")

    # Define which sweeps to run
    sweeping_names = list(mode_configs.keys())

    # Run all sweeps as specified
    for name in sweeping_names:
        # Obtain the sweeping config for the given name
        sweeping_config = mode_configs[name]["sweep_config"]

        # Get the sweep id and run the sweep (MAX_NUM_RUNS-times)
        print('\n\n')
        sweep_id = wandb.sweep(sweeping_config, project='big_data_lang_tech')

        wandb.agent(sweep_id=sweep_id,
                    function=sweep_agent,
                    count=MAX_NUM_RUNS)