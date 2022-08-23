import json

# Note on the configurations: Not all optimizers need all parameters.
# They are included for easier usage in main.py
CONFIG_PATH = "./models/model_configs.json"
CONFIGS = {"Supervised_SGD": {"learning_manager": {"model_name": "Supervised_SGD", "encoder": "baseline"},
                              "training": {"epochs": 10, "batch_size": 32, "optimizer_name": "sgd", "lr": 0.005,
                                           "momentum": 0.7, "weight_decay": 0, "alpha": 0.99, "eps": 1e-08,
                                           "trust_coef": 0.001}},

           "Supervised_RMS": {"learning_manager": {"model_name": "Supervised_RMS", "encoder": "baseline"},
                              "training": {"epochs": 10, "batch_size": 32, "optimizer_name": "rmsprop", "lr": 0.00005,
                                           "momentum": 0, "weight_decay": 0, "alpha": 0.99, "eps": 1e-08,
                                           "trust_coef": 0.001}},

           "Supervised_LARS": {"learning_manager": {"model_name": "Supervised_LARS", "encoder": "baseline"},
                               "training": {"epochs": 10, "batch_size": 32, "optimizer_name": "lars", "lr": 0.285,
                                            "momentum": 0.3, "weight_decay": 0, "alpha": 0.99, "eps": 1e-08,
                                            "trust_coef": 0.00125}}
           }


if __name__ == "__main__":
    with open(CONFIG_PATH, 'w') as f:
        json.dump(CONFIGS, f, indent=4)