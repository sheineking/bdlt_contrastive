import json
import learning_manager as l

# Note on the configurations: Not all optimizers need all parameters.
# They are included for easier usage in main.py
CONFIG_PATH = l.MODEL_OUT_PATH + "/model_configs.json"
CONFIGS = {"Supervised_SGD": {"learning_manager": {"model_name": "Supervised_SGD"},
                              "training": {"epochs": 10, "batch_size": 2, "optimizer_name": "sgd", "lr": 0.01,
                                           "momentum": 0, "weight_decay": 0, "alpha": 0.99, "eps": 1e-08,
                                           "trust_coef": 0.001}},

           "Supervised_RMS": {"learning_manager": {"model_name": "Supervised_RMS"},
                              "training": {"epochs": 10, "batch_size": 2, "optimizer_name": "rmsprop", "lr": 0.01,
                                           "momentum": 0, "weight_decay": 0, "alpha": 0.99, "eps": 1e-08,
                                           "trust_coef": 0.001}},

           "Supervised_LARS": {"learning_manager": {"model_name": "Supervised_LARS"},
                               "training": {"epochs": 10, "batch_size": 2, "optimizer_name": "lars", "lr": 0.1,
                                            "momentum": 0, "weight_decay": 0, "alpha": 0.99, "eps": 1e-08,
                                            "trust_coef": 0.001}},
           }


if __name__ == "__main__":
    with open(CONFIG_PATH, 'w') as f:
        json.dump(CONFIGS, f, indent=4)