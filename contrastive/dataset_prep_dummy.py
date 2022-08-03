import datasets
import pandas as pd
import csv
import numpy as np
from random import shuffle

# =============================================================
# Hard negative mining
# =============================================================
class HardNegativePreparer():
    """
    This class is used by the contrastive learning_manager to get negative samples for specified sentences.
    It takes the sentences from the csv "Negative_Sentences.csv" created by the HardNegativeFinder
    """
    def __init__(self, csv_path="Matched_Sentences.csv"):
        self.df = pd.read_csv(csv_path)

    def build_dataset_with_negatives(self, dataset: datasets.Dataset, n=1):
        """
        :param dataset: Dataset with two sentences and a label
        :param n:       Number of negatives to add
        :return:        Dataset that only contains the pairs that have a label of 1.
                        Additionally, n more sentences were added
        """
        # Filter only positive samples
        filtered_ds = dataset.filter(lambda x: x["label"] == 1)

        # Add the hard negatives to each ds separately
        train_ds = filtered_ds["train"].map(lambda x: self.get_hard_negatives(sen_id=x["idx"], n=n, dataset="train"))
        val_ds = filtered_ds["validation"].map(lambda x: self.get_hard_negatives(sen_id=x["idx"], n=n, dataset="val"))
        test_ds = filtered_ds["test"].map(lambda x: self.get_hard_negatives(sen_id=x["idx"], n=n, dataset="test"))

        # Combine the three datasets into one DatasetDict and return it
        return datasets.DatasetDict({"train": train_ds, "validation": val_ds, "test": test_ds})

    def get_hard_negatives(self, sen_id: int, n=1, dataset="train"):
        """
        Takes in a sentence i and returns n hard negatives as obtained by the HardNegativeFinder
        :param sen_id:  The id for which to return hard negatives
        :param n:       The number of hard negatives to return
        :param dataset: The name of the dataset the sentence belongs to
        :return:        A dictionary of hard negative sentences (starting with sentence3, sentence4, ...)
        """
        # Define the columns of self.df based on n
        columns = [str("sen_" + str(idx)) for idx in range(n)]

        # Get the correct row by matching the sen_id on anchor_idx and the dataset name
        row = self.df.loc[((self.df["anchor_idx"] == "sen1_" + str(sen_id)) & (self.df["dataset"]==dataset))]

        # Get the sentences as a list and turn them into a dictionary starting with key sentence3
        sentences = row[columns].values.tolist()[0]
        result_dict = dict(zip(["sentence" + str(idx + 3) for idx in range(n)], sentences))
        return result_dict











def construct_q_gram_set(string: str, q=3, q_padding=False):
    s = set()

    # Apply padding if specified
    if q_padding:
        string = "#"*(q-1) + string + "#"*(q-1)

    string = string.lower()

    # If the string is shorter than Q, apply padding at the end
    if len(string) < q:
        string= string + str("#"*(q-len(string)))

    for i,_ in enumerate(string):
        q_gram = string[i:min(i+q, len(string))]
        if len(q_gram)==q:
            s.add(q_gram)

    return s


class HardNegativeFinder():
    """
    This class is used to find hard negative sentences for sentences in the provided dataset.
    These are written to Negative_Sentences.csv.
    """
    def __init__(self, ds: datasets.Dataset):
        self.train_ds = ds["train"]
        self.val_ds = ds["validation"]
        self.test_ds = ds["test"]

    def create_qgrams(self):
        """
        For all positive pairs, only sentence 1 is turned into a Q-gram, because this is the anchor sentence.
        Then, both sentences of all negative pairs are also turned into Q-grams as matching candidates for the anchors.

        In addition, all q-grams are saved to build the vocabulary used by LSH
        """

        result_dict = {}
        loop_list = [{"name": "train", "ds": self.train_ds}, {"name": "val", "ds": self.val_ds},
                     {"name": "test", "ds": self.test_ds}]

        for pair in loop_list:
            vocab = set()

            # Get all positive pairs
            pos_sentences = pair["ds"].filter(lambda x: x["label"] == 1)
            neg_sentences = pair["ds"].filter(lambda x: x["label"] == 0)

            # Get the Q-gram sets for all anchor sentences
            anchor_sen = []
            anchor_idx = []
            for elem in pos_sentences:
                sen1 = elem["sentence1"]
                q_gram_set = construct_q_gram_set(sen1)

                # Add the set to the anchor and update the vocabulary
                anchor_sen.append(q_gram_set)
                vocab.update(q_gram_set)

                # Add the idx to identify sentences
                anchor_idx.append("sen1_" + str(elem["idx"]))


            # Do the same for both negative sentences
            candidate_sen = []
            candidate_idx = []
            for elem in neg_sentences:
                sen1 = elem["sentence1"]
                sen2 = elem["sentence2"]

                q_gram_set1 = construct_q_gram_set(sen1)
                q_gram_set2 = construct_q_gram_set(sen2)

                # Add the sets to the candidates and the vocabulary
                candidate_sen.append(q_gram_set1)
                candidate_sen.append(q_gram_set2)
                vocab.update(q_gram_set1)
                vocab.update(q_gram_set2)

                # Add the idx to identify sentences
                candidate_idx.append("sen1_" + str(elem["idx"]))
                candidate_idx.append("sen2_" + str(elem["idx"]))


            # Create the Hasher for the current dataset based on the vocabulary and save the Q-grams
            Hasher = Locality_Sensitive_Hasher(vocab=vocab, num_signatures=300)

            # Add both lists and the hasher to the result_dict
            result_dict[pair["name"]] = {"anchors": {"idx": anchor_idx,
                                                     "sen": anchor_sen},
                                         "candidates": {"idx": candidate_idx,
                                                        "sen": candidate_sen},
                                         "hasher": Hasher}

        self.q_gram_dict = result_dict


    def find_negatives(self):
        if not hasattr(self, "q_gram_dict"):
            self.create_qgrams()

        # Get the signatures for the anchors and candidates for all three datasets
        self.get_signatures()

        # Initialize a dictionary
        negatives_dict = {}


        for ds_name, ds_dict in self.q_gram_dict.items():
            print("\n\n" + "=" * 50)
            print(f"Forming matches for {ds_name}")
            print("=" * 50)

            anchor_sig = ds_dict["anchors"]["sig"]
            candidate_sig = ds_dict["candidates"]["sig"]
            anchor_idx = ds_dict["anchors"]["idx"]
            candidate_idx = ds_dict["candidates"]["idx"]

            # Initialize a sub dictionary for the current ds_name
            ds_match_dict = {}

            # For each of the anchor signatures, find matches in the candidate signatures
            num_anchors = len(anchor_sig)
            for i, sig_vector in enumerate(anchor_sig):
                print(f"- Anchor {i + 1}/{num_anchors}")

                # If an element in a row of candidate sig matches the element in sig_vector, its value is set to true
                # sum(axis=1) counts the number of True values per row
                match_vector = (candidate_sig==sig_vector).sum(axis=1)

                # Combine the number of matches in match_vector with the candidate idx into a dictionary
                # Save the dict in the ds_match_dict (key: corresponding anchor idx)
                ds_match_dict[anchor_idx[i]] = dict(zip(candidate_idx, list(match_vector)))

            # Save the ds_match_dict in the negative_dict
            negatives_dict[ds_name] = ds_match_dict

        # Set the negatives_dict attribute
        self.negatives_dict = negatives_dict


    def get_signatures(self):
        """
        Uses the Hasher-instance in each sub-dictionary of self.q_gram_dict to create signature matrices for
        anchors and candidates.
        One row of the matrix corresponds to one sentence
        """

        for ds_name, ds_dict in self.q_gram_dict.items():
            print("\n\n" + "="*50)
            print(f"Getting signatures for {ds_name}")
            print("=" * 50)

            # Get the signature matrix for both the anchors and the candidates
            Hasher = ds_dict["hasher"]

            print("Anchor sentences:")
            anchor_sig = Hasher.create_signature_matrix(ds_dict["anchors"]["sen"])

            print("\nCandidate sentences:")
            candidate_sig = Hasher.create_signature_matrix(ds_dict["candidates"]["sen"])

            # Add signatures to the dictionary
            self.q_gram_dict[ds_name]["anchors"]["sig"] = anchor_sig
            self.q_gram_dict[ds_name]["candidates"]["sig"] = candidate_sig


    def write_negatives(self, n=50, out_path="Negative_Sentences.csv"):
        """
        Function that uses the negatives identified in find_negatives and writes the sentences out to a csv-File
        Process per anchor sentence
        1. Sort the dictionary by value (number of matching signatures)
        2. Take the n sentences with the highest number of matches
        3. Identify the sentences in the dataset by their index and sentence number
        4. Write a new row into the CSV ([ds_type], [anchor_idx], [sentence1], [sentence2], ..., [sentence_N])

        :param n:           Number of sentences to be stored per anchor
        :param out_path:    Path to the csv that stores the matches
        """

        if not hasattr(self, "negatives_dict"):
            self.find_negatives()

        with open(out_path, "w") as file:
            sentence_columns = [("sen_" + str(num)) for num in range(n)]
            header = "dataset,anchor_idx," + ",".join(sentence_columns) + "\n"
            file.write(header)


        for ds_name, ds_dict in self.negatives_dict.items():
            for anchor_idx, num_match_dict in ds_dict.items():
                # 1. Get the candidate_idx as a list sorted by number of matches
                sorted_idx = list(dict(sorted(num_match_dict.items(), key=lambda x:x[1], reverse=True)).keys())

                # 2. and 3. Get the first n sentences for the sorted_idx list
                top_n_sentences = self.identify_sentences(sorted_idx=sorted_idx, n=n, ds_name=ds_name)

                # 4. Define the csv row and append it to the file
                csv_row = [ds_name, anchor_idx, *top_n_sentences]

                with open(out_path, "a") as file:
                    writer = csv.writer(file)
                    writer.writerow(item for item in csv_row)


    def identify_sentences(self, sorted_idx: list, n, ds_name):
        """
        Takes in a list of indices and a number of sentences.
        Identifies the sentences in the dataset and returns them as a list
        :param sorted_idx:      List of candidate indices; Sorted by number of matches with an anchor sentence
        :param n:               How many sentences to return
        :param ds_name:         Name of the dataset in which to look for the sentences
        :return:                List of the sentences with the highest number of matches
        """

        # Keep a maximum of n indices
        num_sentences = min(len(sorted_idx), n)
        sorted_idx = sorted_idx[:num_sentences]

        # Turn the sentence indices into the correct format for the dataset
        # - The number at the end is the idx in the ds
        # - sen1 or sen2 identifies the sentence in the pair
        index_nums = [int(full_idx.split("_")[1]) for full_idx in sorted_idx]
        sen_keys = ["sentence1" if full_idx.split("_")[0] == "sen1" else "sentence2" for full_idx in sorted_idx]

        # Filter the correct dataset to only contain sentences in the sorted_idx
        ds = getattr(self, ds_name + "_ds").filter(lambda x: x["idx"] in index_nums)

        # Get the position of each index in index_nums in the idx column of the dataset
        ds_idx = ds["idx"]
        positional_index = [ds_idx.index(elem) for elem in index_nums]

        # Get the sentences based on their position in the ds and whether it is sentence1 or sentence2
        result_list = []
        for i, position in enumerate(positional_index):
            sentence = ds[position][sen_keys[i]]
            result_list.append(sentence)

        return result_list








def create_vocab_dict(vocab):
    # Create a dictionary to map vocab elements to indices
    vocab_dict = {}
    for i, q_gram in enumerate(vocab):
        vocab_dict[q_gram] = i

    return vocab_dict

# ==============================================================
# LSH to speed up the comparison
# ==============================================================
class Locality_Sensitive_Hasher():
    def __init__(self, vocab, num_signatures=100):
        """
        :param vocab:           Set of all q-grams in the sets to be encoded
        :param num_signatures:  How many signatures should be used to represent one q-gram set
        """

        self.vocab = vocab
        print('Setting vocabulary...')
        self.vocab_dict = create_vocab_dict(self.vocab)
        self.num_signatures = num_signatures

        print('Creating hash functions...')
        self.update_lsh_hash_funcs()

    def update_vocab(self, vocab):
        self.vocab = vocab
        self.vocab_dict = create_vocab_dict(self.vocab)
        print('Vocabulary updated')

        print('Updating hash functions with new vocabulary...')
        self.update_lsh_hash_funcs()


    def update_lsh_hash_funcs(self):
        self.hash_list = []
        for i in range(self.num_signatures):
            # Create a randomized list of numbers from one to the length of the vocabulary
            hash_func = list(range(1, len(self.vocab) + 1))
            shuffle(hash_func)
            self.hash_list.append(hash_func)


    def create_signature_matrix(self, q_grams: list):
        """
        Takes in a list of Q-grams and returns a matrix of signatures
        :param q_grams:     List of Q-grams
        :return:            Numpy array of signatures
        """

        num_elem = len(q_grams)
        sig_matrix = np.zeros((num_elem, self.num_signatures))

        for i, q_gram in enumerate(q_grams):
            print(f"- Sentence {i+1}/{num_elem}")
            sparse_vector = self.create_sparse_vector(q_gram)
            signature = self.create_dense_vector(sparse_vector)

            sig_matrix[i:i+1, :] = signature

        return sig_matrix


    def create_sparse_vector(self, q_gram_set):
        '''
        Takes in a string
        :param q_gram_set:      Set of q-grams to be turned into a sparse one-vector using self.vocab_dict
        :return:                One-hot vector corresponding to the provided value as numpy array
        '''

        val_one_hot = np.zeros(len(self.vocab))
        # Identify the index of each q_gram in the dictionary and set the corresponding element to 1
        for q_gram in q_gram_set:
            ind = self.vocab_dict[q_gram]
            val_one_hot[ind:(ind + 1)] = 1

        return val_one_hot


    def create_dense_vector(self, one_hot_vector):
        '''
        Creates a sparse signature vector using self.hash_list
        :param one_hot_vector:      Sparse one-hot vector to be turned into a dense signature vector
        :return:                    Dense signature vector as numpy array
        '''
        sig_vec = np.zeros(self.num_signatures, dtype=int)
        for sig_post, func in enumerate(self.hash_list):
            for i in range(1, len(self.vocab)+1):
                # Obtain the index of i in the hash-function (first iteration looks for the position of 1, then 2, ...)
                idx = func.index(i)
                vec_value = one_hot_vector[idx]

                # If the value at that position in the vector is 1, append the signature value
                # and proceed with the next hash
                if vec_value==1:
                    sig_vec[sig_post:(sig_post+1)] = i
                    break

        return sig_vec


if __name__ == "__main__":
    ds = datasets.load_dataset(path="glue", name="mrpc")
    Finder = HardNegativeFinder(ds=ds)

    Finder.write_negatives()