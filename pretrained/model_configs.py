import json

# Note on the configurations: Not all optimizers need all parameters.
# They are included for easier usage in main.py
CONFIG_PATH = "./models/model_configs.json"
CONFIGS = { # ======================================================================
            # Configurations for classifiers on top of pre-trained contrastive models
            # ======================================================================
           # Todo: Set the three encoder attributes based on the name of the model that achieved the lowest validation
           #  loss. Same needs to be done in the sweeping_configs.py (in sweeping folder)

            # Todo: Run the sweeping_configs.py script after changing the parameters

           "Pretrained_Pairwise": {"learning_manager": {"model_name": "Pretrained_Pairwise",
                                                        "encoder": "Pairwise_RMS"},
                                   "training": {"epochs": 10, "batch_size": 32, "optimizer_name": "sgd", "lr": 0.01,
                                                "momentum": 0, "weight_decay": 0, "alpha": 0.99, "eps": 1e-08,
                                                "trust_coef": 0.001}},
           "Pretrained_Triplet": {"learning_manager": {"model_name": "Pretrained_Triplet",
                                                        "encoder": "Triplet_RMS"},
                                   "training": {"epochs": 10, "batch_size": 32, "optimizer_name": "sgd", "lr": 0.01,
                                                "momentum": 0, "weight_decay": 0, "alpha": 0.99, "eps": 1e-08,
                                                "trust_coef": 0.001}},
           "Pretrained_InfoNCE": {"learning_manager": {"model_name": "Pretrained_InfoNCE",
                                                        "encoder": "InfoNCE_RMS"},
                                   "training": {"epochs": 10, "batch_size": 32, "optimizer_name": "sgd", "lr": 0.01,
                                                "momentum": 0, "weight_decay": 0, "alpha": 0.99, "eps": 1e-08,
                                                "trust_coef": 0.001}}
           }


if __name__ == "__main__":
    with open(CONFIG_PATH, 'w') as f:
        json.dump(CONFIGS, f, indent=4)