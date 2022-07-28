import json
import learning_manager as l

# Note on the configurations: Not all optimizers need all parameters.
# They are included for easier usage in main.py
CONFIG_PATH = l.MODEL_OUT_PATH + "/model_configs.json"
CONFIGS = {"Pairwise_SGD": {"learning_manager": {"train_mode": "pairwise", "model_name": "Pairwise_SGD"},
                            "training": {"epochs": 10, "batch_size": 2, "optimizer_name": "sgd", "lr": 0.1,
                                         "momentum": 0, "weight_decay": 0, "alpha": 0.99, "eps": 1e-08,
                                         "trust_coef": 0.001}},

           "Pairwise_RMS": {"learning_manager": {"train_mode": "pairwise", "model_name": "Pairwise_RMS"},
                            "training": {"epochs": 10, "batch_size": 2, "optimizer_name": "rmsprop", "lr": 0.1,
                                         "momentum": 0, "weight_decay": 0, "alpha": 0.99, "eps": 1e-08,
                                         "trust_coef": 0.001}},

           "Pairwise_LARS": {"learning_manager": {"train_mode": "pairwise", "model_name": "Pairwise_LARS"},
                             "training": {"epochs": 10, "batch_size": 2, "optimizer_name": "lars", "lr": 0.1,
                                          "momentum": 0, "weight_decay": 0, "alpha": 0.99, "eps": 1e-08,
                                          "trust_coef": 0.001}},


           "Triplet_SGD": {"learning_manager": {"train_mode": "triplet", "model_name": "Triplet_SGD"},
                           "training": {"epochs": 10, "batch_size": 2, "optimizer_name": "sgd", "lr": 0.1,
                                        "momentum": 0, "weight_decay": 0, "alpha": 0.99, "eps": 1e-08,
                                        "trust_coef": 0.001}},

           "Triplet_RMS": {"learning_manager": {"train_mode": "triplet", "model_name": "Triplet_RMS"},
                           "training": {"epochs": 10, "batch_size": 2, "optimizer_name": "rmsprop", "lr": 0.1,
                                        "momentum": 0, "weight_decay": 0, "alpha": 0.99, "eps": 1e-08,
                                        "trust_coef": 0.001}},

           "Triplet_LARS": {"learning_manager": {"train_mode": "triplet", "model_name": "Triplet_LARS"},
                            "training": {"epochs": 10, "batch_size": 2, "optimizer_name": "lars", "lr": 0.1,
                                         "momentum": 0, "weight_decay": 0, "alpha": 0.99, "eps": 1e-08,
                                         "trust_coef": 0.001}},



           "InfoNCE_SGD": {"learning_manager": {"train_mode": "infoNCE", "model_name": "InfoNCE_SGD"},
                           "training": {"epochs": 10, "batch_size": 2, "optimizer_name": "sgd", "lr": 0.1,
                                        "momentum": 0, "weight_decay": 0, "alpha": 0.99, "eps": 1e-08,
                                        "trust_coef": 0.001}},

           "InfoNCE_RMS": {"learning_manager": {"train_mode": "infoNCE", "model_name": "InfoNCE_RMS"},
                           "training": {"epochs": 10, "batch_size": 2, "optimizer_name": "rmsprop", "lr": 0.1,
                                        "momentum": 0, "weight_decay": 0, "alpha": 0.99, "eps": 1e-08,
                                        "trust_coef": 0.001}},

           "InfoNCE_LARS": {"learning_manager": {"train_mode": "infoNCE", "model_name": "InfoNCE_LARS"},
                            "training": {"epochs": 10, "batch_size": 2, "optimizer_name": "lars", "lr": 0.1,
                                         "momentum": 0, "weight_decay": 0, "alpha": 0.99, "eps": 1e-08,
                                         "trust_coef": 0.001}},
           }


if __name__ == "__main__":
    with open(CONFIG_PATH, 'w') as f:
        json.dump(CONFIGS, f, indent=4)