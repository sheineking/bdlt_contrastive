import torch as T
from info_nce import InfoNCE

# =======================================================
# Epsilon values
# =======================================================
# 0.5 was chosen based on experiments on the GLUE dataset. (see ./distance_csv/)
#  - Note: Pairwise_SGD with batch_size=16; Otherwise the parameters as set in model_configs.json
#  - Anything >= 0.7 leads to overfitting (the downward spikes in the distance values are on the validation set)
PAIRWISE_EPS = 0.5
TRIPLET_EPS = 0.5

class PairwiseLoss(T.nn.Module):
    def __init__(self, eps=PAIRWISE_EPS):
        super(PairwiseLoss, self).__init__()
        self.eps = eps
        self.pdist = T.nn.PairwiseDistance(p=2)

    def forward(self, embeddings: dict, label, num_sentences=None):
        """
        Calculates the pairwise loss based on two embeddings and a label
        :param embeddings:      The embeddings of one batch as a dictionary.
                                It has to contain two dictionaries with the keys "emb1" and "emb2"
        :param label:           1=The two sentences are paraphrases; 0=They are not
        :param num_sentences:   Parameter is not required; Only for compatibility with InfoNCE
        :return:                Loss for the given embeddings
        """

        # Calculate the distance
        dist = self.pdist(embeddings["emb1"], embeddings["emb2"])

        # Calculate the loss (Based on Chopra et al. 2005)
        # - For all instances of label=1, the distance is the loss
        # - For all instances of label=0, it is the max of 0 and the additional distance beyond a hyperparameter epsilon
        positive_cases = T.square(dist)
        negative_cases = T.square(T.maximum(input=(PAIRWISE_EPS - dist), other=T.zeros_like(input=dist)))

        loss = T.mean((label) * positive_cases + (1-label) * negative_cases)
        return loss

    def write_csv(self, dist, label):
        batch_size = dist.shape[0]
        for elem in range(batch_size):
            l = label[elem].item()
            d = dist[elem].item()

            with open(self.csv_path, "a") as f:
                f.write(str(l) + "," + str(d) + "\n")




class TripleLoss(T.nn.Module):
    def __init__(self, eps=TRIPLET_EPS):
        super(TripleLoss, self).__init__()
        self.triplet_loss = T.nn.TripletMarginLoss(margin=eps, p=2)

    def forward(self, embeddings: dict, label=None, num_sentences=None):
        """
        Calculates the triplet loss based on three embeddings
        :param embeddings:      The embeddings of one batch as a dictionary.
                                It has to contain three dictionaries with the keys "emb1", "emb2", and "emb3"
        :param label:           Parameter is not required; Only for compatibility with pairwise loss
        :param num_sentences:   Parameter is not required; Only for compatibility with InfoNCE
        :return:                Loss for the given embeddings
        """

        anchor = embeddings['emb1']
        positive = embeddings['emb2']
        negative = embeddings['emb3']

        return self.triplet_loss(anchor, positive, negative)



class InfoNCE_Loss(T.nn.Module):
    def __init__(self):
        super(InfoNCE_Loss, self).__init__()
        self.info_nce_loss = InfoNCE(negative_mode='paired')

    def forward(self, embeddings: dict, label=None, num_sentences=None):
        """
        Calculates the paired InfoNCE loss based on the embeddings
        :param embeddings:      The embeddings of one batch as a dictionary.
                                It has to contain at least three dictionaries with the keys "emb1", "emb2", ...
        :param label:           Parameter is not required; Only for compatibility with pairwise loss
        :param num_sentences:   Number of sentences in the batch
        :return:                Loss for the given embeddings
        """

        anchor = embeddings['emb1']
        positive = embeddings['emb2']

        # Stack the negative tensors into one tensor with the dimensions [batch_size, num_negative, embedding_size]
        neg_list = []
        for i in range(3, num_sentences+1):
            neg_list.append(embeddings['emb' + str(i)])
        negatives = T.stack(tensors=neg_list, dim=1)

        # Return the loss
        return self.info_nce_loss(anchor, positive, negatives)


