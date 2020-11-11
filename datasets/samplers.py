# Author: Jacek Komorowski
# Warsaw University of Technology

import random
import copy

from torch.utils.data import DataLoader, Sampler

from datasets.oxford import OxfordDataset

VERBOSE = False


class BatchSampler(Sampler):
    # Sampler returning list of indices to form a mini-batch
    # Samples elements in groups consisting of k=2 similar elements (positives)
    # Batch has the following structure: item1_1, ..., item1_k, item2_1, ... item2_k, itemn_1, ..., itemn_k
    def __init__(self, dataset: OxfordDataset, batch_size: int, batch_size_limit: int = None,
                 batch_expansion_rate: float = None):
        if batch_expansion_rate is not None:
            assert batch_expansion_rate > 1., 'batch_expansion_rate must be greater than 1'
            assert batch_size <= batch_size_limit, 'batch_size_limit must be greater or equal to batch_size'

        self.batch_size = batch_size
        self.batch_size_limit = batch_size_limit
        self.batch_expansion_rate = batch_expansion_rate
        self.dataset = dataset
        self.k = 2  # Number of positive examples per group must be 2
        if self.batch_size < 2 * self.k:
            self.batch_size = 2 * self.k
            print('WARNING: Batch too small. Batch size increased to {}.'.format(self.batch_size))

        self.batch_idx = []     # Index of elements in each batch (re-generated every epoch)

        self.elems_ndx = {}    # Dictionary of point cloud indexes
        for ndx in self.dataset.queries:
            self.elems_ndx[ndx] = True

    def __iter__(self):
        # Re-generate batches every epoch
        self.generate_batches()
        for batch in self.batch_idx:
            yield batch

    def __len(self):
        return len(self.batch_idx)

    def expand_batch(self):
        if self.batch_expansion_rate is None:
            print('WARNING: batch_expansion_rate is None')
            return

        if self.batch_size >= self.batch_size_limit:
            return

        old_batch_size = self.batch_size
        self.batch_size = int(self.batch_size * self.batch_expansion_rate)
        self.batch_size = min(self.batch_size, self.batch_size_limit)
        print('=> Batch size increased from: {} to {}'.format(old_batch_size, self.batch_size))

    def generate_batches(self):
        # Generate training/evaluation batches.
        # batch_idx holds indexes of elements in each batch as a list of lists
        self.batch_idx = []

        unused_elements_ndx = copy.deepcopy(self.elems_ndx)
        current_batch = []

        assert self.k == 2, 'sampler can sample only k=2 elements from the same class'

        while True:
            if len(current_batch) >= self.batch_size or len(unused_elements_ndx) == 0:
                # Flush out a new batch and reinitialize a list of available location
                # Flush out batch, when it has a desired size, or a smaller batch, when there's no more
                # elements to process
                if len(current_batch) >= 2*self.k:
                    # Ensure there're at least two groups of similar elements, otherwise, it would not be possible
                    # to find negative examples in the batch
                    assert len(current_batch) % self.k == 0, 'Incorrect bach size: {}'.format(len(current_batch))
                    self.batch_idx.append(current_batch)
                    current_batch = []
                if len(unused_elements_ndx) == 0:
                    break

            # Add k=2 similar elements to the batch
            selected_element = random.choice(list(unused_elements_ndx))
            unused_elements_ndx.pop(selected_element)
            positives = self.dataset.get_positives_ndx(selected_element)
            if len(positives) == 0:
                # Broken dataset element without any positives
                continue

            unused_positives = [e for e in positives if e in unused_elements_ndx]
            # If there're unused elements similar to selected_element, sample from them
            # otherwise sample from all similar elements
            if len(unused_positives) > 0:
                second_positive = random.choice(unused_positives)
                unused_elements_ndx.pop(second_positive)
            else:
                second_positive = random.choice(positives)

            current_batch += [selected_element, second_positive]

        for batch in self.batch_idx:
            assert len(batch) % self.k == 0, 'Incorrect bach size: {}'.format(len(batch))


if __name__ == '__main__':
    dataset_path = '/media/sf_Datasets/PointNetVLAD'
    query_filename = 'test_queries_baseline.pickle'

    ds = OxfordDataset(dataset_path, query_filename)
    sampler = BatchSampler(ds, batch_size=16)
    dataloader = DataLoader(ds, batch_sampler=sampler)
    e = ds[0]
    res = next(iter(dataloader))
    print(res)

