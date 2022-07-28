import torch as T
import torch.nn.functional as F
from transformers import AutoModel
from torchlars import LARS

# Available models: https://www.sbert.net/docs/pretrained_models.html
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
HIDDEN_SIZE = 384

# ================================================================
# Models
# ================================================================
class SupervisedModel(T.nn.Module):
    def __init__(self):
        super(SupervisedModel, self).__init__()

        # Encoder
        self.encoder = AutoModel.from_pretrained(MODEL_NAME)

        # Output layer
        self.linear = T.nn.Linear(HIDDEN_SIZE, 1)



    def feed(self, x):
        """
        Calculates the embeddings based on tokens, attention_masks and token_type_ids
        :param x:   Input dictionary
        :return:    Normalized embedding
        """

        # Encode the input
        output = self.encoder(**x)
        embedding = mean_pooling(output, x['attention_mask'])

        # Return the normalized embedding
        return F.normalize(embedding, p=2, dim=1)


    def forward(self, batch: dict):
        """
        Function that takes in a dictionary for the current batch that contains tokenized sentence pairs of the form

        {"input_ids": tensor([[...]]), "attention_mask": tensor([[...]]), "token_type_ids": tensor([[..]])}

        and returns a prediction for whether they are paraphrases

        :param batch            The batch of tokenized sentence pairs produced by a dataloader.
        :return:                The predictions for the sentence pairs
        """

        # Drop the labels
        batch.pop("labels", None)

        # Calculate the embeddings for the sentence pairs in the batch
        embeddings = self.feed(batch)

        # Return the results of the sigmoid
        logits = self.linear(embeddings)
        return logits





# Helper functions
# ------------------------------------------------------------------------
# From https://huggingface.co/sentence-transformers/all-mpnet-base-v2
def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0] #First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return T.sum(token_embeddings * input_mask_expanded, 1) / T.clamp(input_mask_expanded.sum(1), min=1e-9)





# ================================================================
# Optimizer
# ================================================================
def get_optimizer(params, optimizer_name="sgd", lr=0.1, momentum=0, weight_decay=0, alpha=0.99, eps=1e-08,
                  trust_coef=0.001):
    """
    Function to prepare an optimizer as specified by the parameters.
    The selection was made based on the optimizers chosen by Khosla et al (https://arxiv.org/abs/2004.11362)

    :param params:              Parameters of the model
    :param optimizer_name:      Used to identify the optimizer to be used
    :param lr:                  Learning rate
    :param momentum:            Momentum factor for SGD, RMSProp, and LARS
    :param weight_decay:        Weight Decay for SGD, RMSProp, and LARS
    :param alpha:               Alpha for RMSProp
    :param eps:                 Epsilon for RMSProp or LARS
    :param trust_coef:          Trust coefficient for LARS
    :return:
    """
    optimizer_name = optimizer_name.lower()

    if optimizer_name == "rmsprop":
        # https://pytorch.org/docs/stable/generated/torch.optim.RMSprop.html#torch.optim.RMSprop
        optimizer = T.optim.RMSprop(params=params, lr=lr, alpha=alpha, eps=eps, weight_decay=weight_decay,
                                    momentum=momentum)

    elif optimizer_name == "lars":
        # https://pypi.org/project/torchlars/
        base_optimizer = T.optim.SGD(params=params, lr=lr, momentum=momentum, weight_decay=weight_decay)
        optimizer = LARS(optimizer=base_optimizer, eps=eps, trust_coef=trust_coef)

    else:
        # https://pytorch.org/docs/stable/generated/torch.optim.SGD.html#torch.optim.SGD
        optimizer = T.optim.SGD(params=params, lr=lr, momentum=momentum, weight_decay=weight_decay)

    return optimizer


