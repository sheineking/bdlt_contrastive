import torch as T
import torch.nn.functional as F
from transformers import AutoModel

# Available models: https://www.sbert.net/docs/pretrained_models.html
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
model = AutoModel.from_pretrained(MODEL_NAME)



# Todo: Clarify if no grad needs to be set here for non-training

class PairwiseContrastive(T.nn.Module):
    def __init__(self):
        super(PairwiseContrastive, self).__init__()
        self.encoder = model

    def feed(self, x):
        # Encode the input
        output = self.encoder(**x)
        embedding = mean_pooling(output, x['attention_mask'])

        # Return the normalized embedding
        return F.normalize(embedding, p=2, dim=1)

    def forward(self, x1, x2):
        """
        Function that takes in two dictionaries of tokenized sentences of the form

        {"input_ids": tensor([[...]]), "attention_mask": tensor([[...]])}

        and returns the corresponding embeddings.

        :param x1:      Dictionary corresponding to the first tokenized sentence
        :param x2:      Dictionary corresponding to the second tokenized sentence
        :return:        Two embeddings
        """
        embedding1 = self.feed(x1)
        embedding2 = self.feed(x2)

        return embedding1, embedding2




# ================================================================
# Helper functions
# ================================================================
# From https://huggingface.co/sentence-transformers/all-mpnet-base-v2
def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0] #First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return T.sum(token_embeddings * input_mask_expanded, 1) / T.clamp(input_mask_expanded.sum(1), min=1e-9)