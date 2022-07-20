import datasets

def build_triplet_ds(dataset: datasets.Dataset):
    """
    :param dataset: Dataset with two sentences and a label
    :return:        Dataset that only contains the pairs that have a label of 1.
                    Additionally, a third sentence was added
    """

    # Filter only positive samples
    filtered_ds = dataset.filter(lambda x: x["label"] == 1)

    # Add a third sentence
    filtered_ds = filtered_ds.map(lambda x: {'sentence3': "This is a random sentence."})

    return filtered_ds



def build_infoNCE_ds(dataset: datasets.Dataset):
    """
    :param dataset: Dataset with two sentences and a label
    :return:        Dataset that only contains the pairs that have a label of 1.
                    Additionally, a three more sentences were added
    """

    # Filter only positive samples
    filtered_ds = dataset.filter(lambda x: x["label"] == 1)

    # Add a third sentence
    filtered_ds = filtered_ds.map(lambda x: {'sentence3': "This is a random sentence.",
                                             'sentence4': "Another dummy example.",
                                             'sentence5': "This is the last negative sample."})

    return filtered_ds