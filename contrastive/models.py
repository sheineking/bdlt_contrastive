import torch as T
import torch.nn.functional as F
from transformers import AutoModel

# Available models: https://www.sbert.net/docs/pretrained_models.html
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
model = AutoModel.from_pretrained(MODEL_NAME)


class ContrastiveModel(T.nn.Module):
    def __init__(self):
        super(ContrastiveModel, self).__init__()
        self.encoder = model

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


    def forward(self, batch: dict, num_sentences: int):
        """
        Function that takes in a dictionary for the current batch that contains
        dictionaries of tokenized sentences of the form

        {"input_ids": tensor([[...]]), "attention_mask": tensor([[...]]), "token_type_ids": tensor([[..]])}

        and returns the corresponding embeddings.

        :param batch            The batch produced by a dataloader.
                                It has to contain at least dictionaries with the keys "input1", "input2", ...
        :param num_sentences    How many sentences are in the dataset (Varies based on loss)
        :return:                The embeddings in a dictionary
        """

        # Produce embeddings for each of the inputs
        embeddings = {}
        for num in range(1, num_sentences + 1):
            embedding = self.feed(batch["input" + str(num)])
            embeddings['emb' + str(num)] = embedding

        return embeddings




# ================================================================
# Helper functions
# ================================================================
# From https://huggingface.co/sentence-transformers/all-mpnet-base-v2
def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0] #First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return T.sum(token_embeddings * input_mask_expanded, 1) / T.clamp(input_mask_expanded.sum(1), min=1e-9)