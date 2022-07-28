import math
import json


CONFIG_PATH = "./sweeping_configs.json"

# Note: The log-uniform values were chosen based on https://github.com/wandb/client/issues/507
SWEEP_CONFIGS = {"Pairwise_SGD": {"sweep_config": {"name": "Pairwise_SGD",
                                                   "method": "bayes",
                                                   "metric": {
                                                       "name": "val_loss",
                                                       "goal": "minimize"},
                                                   "parameters": {
                                                       "train_mode":
                                                           {"values": ["pairwise"]},

                                                       "model_name":
                                                           {"values": ["Pairwise_SGD_Sweep"]},

                                                       "optimizer_name":
                                                           {"values": ["sgd"]},

                                                       "lr": {
                                                           "min": math.log(0.001), "max": math.log(0.5),
                                                           "distribution": "log_uniform"},

                                                       "momentum": {
                                                           "min": 0.0, "max": 0.95,
                                                           "distribution": "uniform"},

                                                       "weight_decay": {
                                                           "min": 0.0, "max": 0.15,
                                                           "distribution": "uniform"}
                                                   }}
                                  },

                 "Pairwise_RMS": {"sweep_config": {"name": "Pairwise_RMS",
                                                   "method": "bayes",
                                                   "metric": {
                                                       "name": "val_loss",
                                                       "goal": "minimize"},
                                                   "parameters": {
                                                       "train_mode":
                                                           {"values": ["pairwise"]},

                                                       "model_name":
                                                           {"values": ["Pairwise_RMS_Sweep"]},

                                                       "optimizer_name":
                                                           {"values": ["rmsprop"]},

                                                       "lr": {
                                                           "min": math.log(0.001), "max": math.log(0.5),
                                                           "distribution": "log_uniform"},

                                                       "momentum": {
                                                           "min": 0.0, "max": 0.95,
                                                           "distribution": "uniform"},

                                                       "weight_decay": {
                                                           "min": 0.0, "max": 0.15,
                                                           "distribution": "uniform"},

                                                       "alpha": {
                                                           "min": math.log(0.85), "max": math.log(0.99),
                                                           "distribution": "log_uniform"},

                                                       "eps": {
                                                           "min": math.log(1e-9), "max": math.log(1e-7),
                                                           "distribution": "log_uniform"}
                                                   }}
                                  },

                 "Pairwise_LARS": {"sweep_config": {"name": "Pairwise_LARS",
                                                    "method": "bayes",
                                                    "metric": {
                                                       "name": "val_loss",
                                                       "goal": "minimize"},
                                                    "parameters": {
                                                       "train_mode":
                                                           {"values": ["pairwise"]},

                                                       "model_name":
                                                           {"values": ["Pairwise_LARS_Sweep"]},

                                                       "optimizer_name":
                                                           {"values": ["lars"]},

                                                       "lr": {
                                                           "min": math.log(0.001), "max": math.log(0.5),
                                                           "distribution": "log_uniform"},

                                                       "momentum": {
                                                           "min": 0.0, "max": 0.95,
                                                           "distribution": "uniform"},

                                                       "weight_decay": {
                                                           "min": 0.0, "max": 0.15,
                                                           "distribution": "uniform"},

                                                       "eps": {
                                                           "min": math.log(1e-9), "max": math.log(1e-7),
                                                           "distribution": "log_uniform"},

                                                       "trust_coef": {
                                                           "min": math.log(0.0005), "max": math.log(0.005),
                                                           "distribution": "log_uniform"}
                                                   }}
                                   },

                 "Triplet_SGD": {"sweep_config": {"name": "Triplet_SGD",
                                                  "method": "bayes",
                                                  "metric": {
                                                       "name": "val_loss",
                                                       "goal": "minimize"},
                                                  "parameters": {
                                                      "train_mode":
                                                          {"values": ["triplet"]},

                                                      "model_name":
                                                          {"values": ["Triplet_SGD_Sweep"]},

                                                      "optimizer_name":
                                                          {"values": ["sgd"]},

                                                      "lr": {
                                                           "min": math.log(0.001), "max": math.log(0.5),
                                                           "distribution": "log_uniform"},

                                                      "momentum": {
                                                           "min": 0.0, "max": 0.95,
                                                           "distribution": "uniform"},

                                                      "weight_decay": {
                                                           "min": 0.0, "max": 0.15,
                                                           "distribution": "uniform"}
                                                   }}
                                 },

                 "Triplet_RMS": {"sweep_config": {"name": "Triplet_RMS",
                                                  "method": "bayes",
                                                  "metric": {
                                                       "name": "val_loss",
                                                       "goal": "minimize"},
                                                  "parameters": {
                                                      "train_mode":
                                                          {"values": ["triplet"]},

                                                      "model_name":
                                                          {"values": ["Triplet_RMS_Sweep"]},

                                                      "optimizer_name":
                                                           {"values": ["rmsprop"]},

                                                      "lr": {
                                                           "min": math.log(0.001), "max": math.log(0.5),
                                                           "distribution": "log_uniform"},

                                                      "momentum": {
                                                           "min": 0.0, "max": 0.95,
                                                           "distribution": "uniform"},

                                                      "weight_decay": {
                                                           "min": 0.0, "max": 0.15,
                                                           "distribution": "uniform"},

                                                      "alpha": {
                                                           "min": math.log(0.85), "max": math.log(0.99),
                                                           "distribution": "log_uniform"},

                                                      "eps": {
                                                           "min": math.log(1e-9), "max": math.log(1e-7),
                                                           "distribution": "log_uniform"}
                                                   }}
                                 },

                 "Triplet_LARS": {"sweep_config": {"name": "Triplet_LARS",
                                                   "method": "bayes",
                                                   "metric": {
                                                       "name": "val_loss",
                                                       "goal": "minimize"},
                                                   "parameters": {
                                                       "train_mode":
                                                           {"values": ["triplet"]},

                                                       "model_name":
                                                           {"values": ["Triplet_LARS_Sweep"]},

                                                       "optimizer_name":
                                                           {"values": ["lars"]},

                                                       "lr": {
                                                           "min": math.log(0.001), "max": math.log(0.5),
                                                           "distribution": "log_uniform"},

                                                       "momentum": {
                                                           "min": 0.0, "max": 0.95,
                                                           "distribution": "uniform"},

                                                       "weight_decay": {
                                                           "min": 0.0, "max": 0.15,
                                                           "distribution": "uniform"},

                                                       "eps": {
                                                           "min": math.log(1e-9), "max": math.log(1e-7),
                                                           "distribution": "log_uniform"},

                                                       "trust_coef": {
                                                           "min": math.log(0.0005), "max": math.log(0.005),
                                                           "distribution": "log_uniform"}
                                                   }}
                                  },

                 "InfoNCE_SGD": {"sweep_config": {"name": "InfoNCE_SGD",
                                                  "method": "bayes",
                                                  "metric": {
                                                       "name": "val_loss",
                                                       "goal": "minimize"},
                                                  "parameters": {
                                                       "train_mode":
                                                           {"values": ["infoNCE"]},

                                                       "model_name":
                                                           {"values": ["InfoNCE_SGD_Sweep"]},

                                                       "optimizer_name":
                                                           {"values": ["sgd"]},

                                                       "lr": {
                                                           "min": math.log(0.001), "max": math.log(0.5),
                                                           "distribution": "log_uniform"},

                                                       "momentum": {
                                                           "min": 0.0, "max": 0.95,
                                                           "distribution": "uniform"},

                                                       "weight_decay": {
                                                           "min": 0.0, "max": 0.15,
                                                           "distribution": "uniform"}
                                                   }}
                                 },

                 "InfoNCE_RMS": {"sweep_config": {"name": "InfoNCE_RMS",
                                                  "method": "bayes",
                                                  "metric": {
                                                       "name": "val_loss",
                                                       "goal": "minimize"},
                                                  "parameters": {
                                                      "train_mode":
                                                          {"values": ["infoNCE"]},

                                                      "model_name":
                                                           {"values": ["InfoNCE_RMS_Sweep"]},

                                                      "optimizer_name":
                                                           {"values": ["rmsprop"]},

                                                      "lr": {
                                                          "min": math.log(0.001), "max": math.log(0.5),
                                                          "distribution": "log_uniform"},

                                                      "momentum": {
                                                           "min": 0.0, "max": 0.95,
                                                           "distribution": "uniform"},

                                                      "weight_decay": {
                                                           "min": 0.0, "max": 0.15,
                                                           "distribution": "uniform"},

                                                      "alpha": {
                                                           "min": math.log(0.85), "max": math.log(0.99),
                                                           "distribution": "log_uniform"},

                                                      "eps": {
                                                           "min": math.log(1e-9), "max": math.log(1e-7),
                                                           "distribution": "log_uniform"}
                                                   }}
                                 },

                 "InfoNCE_LARS": {"sweep_config": {"name": "InfoNCE_LARS",
                                                   "method": "bayes",
                                                   "metric": {
                                                       "name": "val_loss",
                                                       "goal": "minimize"},
                                                   "parameters": {
                                                       "train_mode":
                                                           {"values": ["infoNCE"]},

                                                       "model_name":
                                                           {"values": ["InfoNCE_LARS_Sweep"]},

                                                       "optimizer_name":
                                                           {"values": ["lars"]},

                                                       "lr": {
                                                           "min": math.log(0.001), "max": math.log(0.5),
                                                           "distribution": "log_uniform"},

                                                       "momentum": {
                                                           "min": 0.0, "max": 0.95,
                                                           "distribution": "uniform"},

                                                       "weight_decay": {
                                                           "min": 0.0, "max": 0.15,
                                                           "distribution": "uniform"},

                                                       "eps": {
                                                           "min": math.log(1e-9), "max": math.log(1e-7),
                                                           "distribution": "log_uniform"},

                                                       "trust_coef": {
                                                           "min": math.log(0.0005), "max": math.log(0.005),
                                                           "distribution": "log_uniform"}
                                                   }}
                                  },
                 }


if __name__ == "__main__":
    with open(CONFIG_PATH, 'w') as f:
        json.dump(SWEEP_CONFIGS, f, indent=4)