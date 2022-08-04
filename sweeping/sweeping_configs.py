import math
import json

CONFIG_PATH = "./sweeping_configs.json"

# Note: The log-uniform values were chosen based on https://github.com/wandb/client/issues/507
SWEEP_CONFIGS = {"supervised":
                     {"default_hyperparameters": {"use_wandb": False,
                                                  "model_name": "Supervised_SGD",
                                                  "epochs": 15,
                                                  "batch_size": 32,
                                                  "optimizer_name": "sgd",
                                                  "lr": 0.1,
                                                  "momentum": 0,
                                                  "weight_decay": 0,
                                                  "alpha": 0.99,
                                                  "eps": 1e-08,
                                                  "trust_coef": 0.001},

                      "Supervised_SGD_finetuned": {"sweep_config": {"name": "Supervised_SGD_finetuned",
                                                                    "method": "bayes",
                                                                    "metric": {
                                                                        "name": "val_loss",
                                                                        "goal": "minimize"},
                                                                    "parameters": {
                                                                        "model_name":
                                                                            {"values": ["Supervised_SGD_Sweep"]},

                                                                        "optimizer_name":
                                                                            {"values": ["sgd"]},

                                                                        "lr": {
                                                                            "min": math.log(0.001), "max": math.log(0.05),
                                                                            "distribution": "log_uniform"},

                                                                        "momentum": {
                                                                            "min": 0.25, "max": 0.8,
                                                                            "distribution": "uniform"}
                                                                    }}
                                                   },

                      "Supervised_RMS_finetuned": {"sweep_config": {"name": "Supervised_RMS_finetuned",
                                                                    "method": "bayes",
                                                                    "metric": {
                                                                        "name": "val_loss",
                                                                        "goal": "minimize"},
                                                                    "parameters": {
                                                                        "model_name":
                                                                            {"values": ["Supervised_RMS_Sweep"]},

                                                                        "optimizer_name": {
                                                                            "values": ["rmsprop"]},

                                                                        "lr": {
                                                                            "min": math.log(0.001), "max": math.log(0.01),
                                                                            "distribution": "log_uniform"},

                                                                        "momentum": {
                                                                            "min": 0.6, "max": 0.95,
                                                                            "distribution": "uniform"},

                                                                        "weight_decay": {
                                                                            "min": math.log(0.01), "max": math.log(0.1),
                                                                            "distribution": "log_uniform"},

                                                                        "alpha": {
                                                                            "min": math.log(0.95), "max": math.log(0.99),
                                                                            "distribution": "log_uniform"},

                                                                        "eps": {
                                                                            "min": math.log(1e-9), "max": math.log(1e-8),
                                                                            "distribution": "log_uniform"}
                                                                    }}
                                                   },

                      "Supervised_LARS_finetuned": {"sweep_config": {"name": "Supervised_LARS_finetuned",
                                                                     "method": "bayes",
                                                                     "metric": {
                                                                         "name": "val_loss",
                                                                         "goal": "minimize"},

                                                                     "parameters": {
                                                                         "model_name":
                                                                             {"values": ["Supervised_LARS_Sweep"]},

                                                                         "optimizer_name":
                                                                             {"values": ["lars"]},

                                                                         "lr": {
                                                                             "min": math.log(0.01), "max": math.log(0.4),
                                                                             "distribution": "log_uniform"},

                                                                         "momentum": {
                                                                             "min": 0.3, "max": 0.95,
                                                                             "distribution": "uniform"},

                                                                         "eps": {
                                                                             "min": math.log(1e-9), "max": math.log(1e-8),
                                                                             "distribution": "log_uniform"},

                                                                         "trust_coef": {
                                                                             "min": math.log(0.001), "max": math.log(0.003),
                                                                             "distribution": "log_uniform"}
                                                                     }}
                                                    }
                      },


                 "contrastive":
                     {"default_hyperparameters": {"use_wandb": False,
                                                  "train_mode": "pairwise",
                                                  "model_name": "Pairwise_SGD",
                                                  "epochs": 15,
                                                  "batch_size": 16,
                                                  "optimizer_name": "sgd",
                                                  "lr": 0.1,
                                                  "momentum": 0,
                                                  "weight_decay": 0,
                                                  "alpha": 0.99,
                                                  "eps": 1e-08,
                                                  "trust_coef": 0.001},

                      "Pairwise_SGD": {"sweep_config": {"name": "Pairwise_SGD",
                                                        "method": "bayes",
                                                        "metric": {
                                                            "name": "val_loss",
                                                            "goal": "minimize"},
                                                        "parameters": {
                                                            "train_mode":
                                                                {"values": ["pairwise"]},

                                                            "batch_size":
                                                                {"values": [16]},

                                                            "model_name":
                                                                {"values": ["Pairwise_SGD_Sweep"]},

                                                            "optimizer_name":
                                                                {"values": ["sgd"]},

                                                            "lr": {
                                                                "min": math.log(0.0001), "max": math.log(0.02),
                                                                "distribution": "log_uniform"},

                                                            "momentum": {
                                                                "min": 0.0, "max": 0.95,
                                                                "distribution": "uniform"},

                                                            "weight_decay": {
                                                                "min": 0.0, "max": 0.1,
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

                                                            "batch_size":
                                                                {"values": [16]},

                                                            "model_name":
                                                                {"values": ["Pairwise_RMS_Sweep"]},

                                                            "optimizer_name":
                                                                {"values": ["rmsprop"]},

                                                            "lr": {
                                                                "min": math.log(0.0001), "max": math.log(0.05),
                                                                "distribution": "log_uniform"},

                                                            "momentum": {
                                                                "min": 0.0, "max": 0.95,
                                                                "distribution": "uniform"},

                                                            "alpha": {
                                                                "min": math.log(0.95), "max": math.log(0.99),
                                                                "distribution": "log_uniform"},

                                                            "eps": {
                                                                "min": math.log(1e-9), "max": math.log(1e-8),
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

                                                            "batch_size":
                                                                {"values": [16]},

                                                             "model_name":
                                                                 {"values": ["Pairwise_LARS_Sweep"]},

                                                             "optimizer_name":
                                                                 {"values": ["lars"]},

                                                             "lr": {
                                                                 "min": math.log(0.0001), "max": math.log(0.5),
                                                                 "distribution": "log_uniform"},

                                                             "momentum": {
                                                                 "min": 0.0, "max": 0.95,
                                                                 "distribution": "uniform"},

                                                             "eps": {
                                                                 "min": math.log(1e-9), "max": math.log(1e-8),
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

                                                            "batch_size":
                                                                {"values": [12]},

                                                           "model_name":
                                                               {"values": ["Triplet_SGD_Sweep"]},

                                                           "optimizer_name":
                                                               {"values": ["sgd"]},

                                                           "lr": {
                                                               "min": math.log(0.001), "max": math.log(0.5),
                                                               "distribution": "log_uniform"},

                                                           "momentum": {
                                                               "min": 0.0, "max": 0.95,
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

                                                            "batch_size":
                                                                {"values": [12]},

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

                                                            "batch_size":
                                                                {"values": [12]},

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

                                                           "batch_size":
                                                               {"values": [8]},

                                                           "model_name":
                                                               {"values": ["InfoNCE_SGD_Sweep"]},

                                                           "optimizer_name":
                                                               {"values": ["sgd"]},

                                                           "lr": {
                                                               "min": math.log(0.001), "max": math.log(0.5),
                                                               "distribution": "log_uniform"},

                                                           "momentum": {
                                                               "min": 0.0, "max": 0.95,
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

                                                           "batch_size":
                                                               {"values": [8]},

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

                                                           "batch_size":
                                                               {"values": [8]},

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

                                                            "eps": {
                                                                "min": math.log(1e-9), "max": math.log(1e-7),
                                                                "distribution": "log_uniform"},

                                                            "trust_coef": {
                                                                "min": math.log(0.0005), "max": math.log(0.005),
                                                                "distribution": "log_uniform"}
                                                        }}
                                       },
                      },

                 "pretrained":
                     {"default_hyperparameters": {"use_wandb": False,
                                                  "model_name": "Pretrained_Pairwise",
                                                  "encoder": "Pairwise_SGD",
                                                  "epochs": 15,
                                                  "batch_size": 32,
                                                  "optimizer_name": "sgd",
                                                  "lr": 0.1,
                                                  "momentum": 0,
                                                  "weight_decay": 0,
                                                  "alpha": 0.99,
                                                  "eps": 1e-08,
                                                  "trust_coef": 0.001},

                      "Pretrained_Pairwise_SGD": {"sweep_config": {"name": "Pretrained_Pairwise_SGD",
                                                                   "method": "bayes",
                                                                   "metric": {
                                                                       "name": "val_loss",
                                                                       "goal": "minimize"},

                                                                   "parameters": {
                                                                       "model_name":
                                                                           {"values": ["Pretrained_Pairwise_SGD_Sweep"]},

                                                                   "encoder":
                                                                       {"values": ["Pairwise_SGD"]},

                                                                   "optimizer_name":
                                                                       {"values": ["sgd"]},

                                                                   "lr": {
                                                                       "min": math.log(0.001),
                                                                       "max": math.log(0.05),
                                                                       "distribution": "log_uniform"},

                                                                   "momentum": {
                                                                       "min": 0.25, "max": 0.8,
                                                                       "distribution": "uniform"}
                                                                    }}
                                                   },

                      "Pretrained_Pairwise_LARS": {"sweep_config": {"name": "Pretrained_Pairwise_LARS",
                                                                    "method": "bayes",
                                                                    "metric": {
                                                                        "name": "val_loss",
                                                                        "goal": "minimize"},

                                                                    "parameters": {
                                                                        "model_name":
                                                                            {"values": ["Pretrained_Pairwise_LARS_Sweep"]},

                                                                        "encoder":
                                                                            {"values": ["Pairwise_SGD"]},

                                                                        "optimizer_name":
                                                                            {"values": ["lars"]},

                                                                        "lr": {
                                                                            "min": math.log(0.01), "max": math.log(0.4),
                                                                            "distribution": "log_uniform"},

                                                                        "momentum": {
                                                                            "min": 0.3, "max": 0.95,
                                                                            "distribution": "uniform"},

                                                                        "eps": {
                                                                            "min": math.log(1e-9), "max": math.log(1e-8),
                                                                            "distribution": "log_uniform"},

                                                                        "trust_coef": {
                                                                            "min": math.log(0.001), "max": math.log(0.003),
                                                                            "distribution": "log_uniform"}
                                                                   }}
                                                  },

                      "Pretrained_Triplet_SGD": {"sweep_config": {"name": "Pretrained_Triplet_SGD",
                                                                   "method": "bayes",
                                                                   "metric": {
                                                                       "name": "val_loss",
                                                                       "goal": "minimize"},

                                                                   "parameters": {
                                                                       "model_name":
                                                                           {"values": [
                                                                               "Pretrained_Triplet_SGD_Sweep"]},

                                                                       "encoder":
                                                                           {"values": ["Triplet_SGD"]},

                                                                       "optimizer_name":
                                                                           {"values": ["sgd"]},

                                                                       "lr": {
                                                                           "min": math.log(0.001),
                                                                           "max": math.log(0.05),
                                                                           "distribution": "log_uniform"},

                                                                       "momentum": {
                                                                           "min": 0.25, "max": 0.8,
                                                                           "distribution": "uniform"}
                                                                   }}
                                                  },

                      "Pretrained_Triplet_LARS": {"sweep_config": {"name": "Pretrained_Triplet_LARS",
                                                                   "method": "bayes",
                                                                   "metric": {
                                                                       "name": "val_loss",
                                                                       "goal": "minimize"},

                                                                   "parameters": {
                                                                       "model_name":
                                                                           {"values": [
                                                                               "Pretrained_Triplet_LARS_Sweep"]},

                                                                       "encoder":
                                                                           {"values": ["Triplet_SGD"]},

                                                                       "optimizer_name":
                                                                           {"values": ["lars"]},

                                                                       "lr": {
                                                                           "min": math.log(0.01), "max": math.log(0.4),
                                                                           "distribution": "log_uniform"},

                                                                       "momentum": {
                                                                           "min": 0.3, "max": 0.95,
                                                                           "distribution": "uniform"},

                                                                       "eps": {
                                                                           "min": math.log(1e-9),
                                                                           "max": math.log(1e-8),
                                                                           "distribution": "log_uniform"},

                                                                       "trust_coef": {
                                                                           "min": math.log(0.001),
                                                                           "max": math.log(0.003),
                                                                           "distribution": "log_uniform"}
                                                                    }}
                                                   },

                      "Pretrained_InfoNCE_SGD": {"sweep_config": {"name": "Pretrained_InfoNCE_SGD",
                                                                  "method": "bayes",
                                                                  "metric": {
                                                                      "name": "val_loss",
                                                                      "goal": "minimize"},

                                                                  "parameters": {
                                                                      "model_name":
                                                                          {"values": [
                                                                              "Pretrained_InfoNCE_SGD_Sweep"]},

                                                                      "encoder":
                                                                          {"values": ["InfoNCE_SGD"]},

                                                                      "optimizer_name":
                                                                          {"values": ["sgd"]},

                                                                      "lr": {
                                                                          "min": math.log(0.001),
                                                                          "max": math.log(0.05),
                                                                          "distribution": "log_uniform"},

                                                                      "momentum": {
                                                                          "min": 0.25, "max": 0.8,
                                                                          "distribution": "uniform"}
                                                                  }}
                                                 },

                      "Pretrained_InfoNCE_LARS": {"sweep_config": {"name": "Pretrained_InfoNCE_LARS",
                                                                   "method": "bayes",
                                                                   "metric": {
                                                                       "name": "val_loss",
                                                                       "goal": "minimize"},

                                                                   "parameters": {
                                                                       "model_name":
                                                                           {"values": [
                                                                               "Pretrained_InfoNCE_LARS_Sweep"]},

                                                                       "encoder":
                                                                           {"values": ["InfoNCE_SGD"]},

                                                                       "optimizer_name":
                                                                           {"values": ["lars"]},

                                                                       "lr": {
                                                                           "min": math.log(0.01), "max": math.log(0.4),
                                                                           "distribution": "log_uniform"},

                                                                       "momentum": {
                                                                           "min": 0.3, "max": 0.95,
                                                                           "distribution": "uniform"},

                                                                       "eps": {
                                                                           "min": math.log(1e-9),
                                                                           "max": math.log(1e-8),
                                                                           "distribution": "log_uniform"},

                                                                       "trust_coef": {
                                                                           "min": math.log(0.001),
                                                                           "max": math.log(0.003),
                                                                           "distribution": "log_uniform"}
                                                                   }}
                                                  }
                      }
                 }


if __name__ == "__main__":
    with open(CONFIG_PATH, 'w') as f:
        json.dump(SWEEP_CONFIGS, f, indent=4)