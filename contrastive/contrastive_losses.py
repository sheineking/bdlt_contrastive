# - pairwise distance:
# - triplet loss: https://pytorch.org/docs/stable/generated/torch.nn.TripletMarginLoss.html
# - InfoNCE: https://pypi.org/project/info-nce-pytorch/
import torch as T

# Todo: Find good value for epsilon
PAIRWISE_EPS = 2.0

class PairwiseLoss(T.nn.Module):
    def __init__(self, eps=PAIRWISE_EPS):
        super(PairwiseLoss, self).__init__()
        self.eps = eps
        self.pdist = T.nn.PairwiseDistance(p=2)

    def forward(self, y1, y2, label):
        """
        Calculates the loss based on two embeddings and a label
        :param y1:      First embedding
        :param y2:      Second embedding
        :param label:   1=The two sentences are paraphrases; 0=They are not
        :return:        Loss for the given combination
        """

        # Calculate the distance
        dist = self.pdist(y1, y2)

        # Calculate the loss (Based on Chopra et al. 2005)
        # - For all instances of label=1, the distance is the loss
        # - For all instances of label=0, it is the max of 0 and the additional distance beyond a hyperparameter epsilon
        positive_cases = T.square(dist)
        negative_cases = T.square(T.maximum(input=dist - PAIRWISE_EPS, other=T.zeros_like(input=dist)))

        loss = T.mean((label) * positive_cases + (1-label) * negative_cases)
        return loss