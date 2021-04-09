# Author: Jacek Komorowski
# Warsaw University of Technology

# Dataset wrapper for Oxford laser scans dataset from PointNetVLAD project
# For information on dataset see: https://github.com/mikacuy/pointnetvlad

import os
import pickle
import numpy as np
import math
from scipy.linalg import expm, norm
import random
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import psutil
from bitarray import bitarray
import tqdm

DEBUG = False


class OxfordDataset(Dataset):
    """
    Dataset wrapper for Oxford laser scans dataset from PointNetVLAD project.
    """
    def __init__(self, dataset_path, query_filename, transform=None, set_transform=None, max_elems=None):
        # transform: transform applied to each element
        # set transform: transform applied to the entire set (anchor+positives+negatives); the same transform is applied
        if DEBUG:
            print('Initializing dataset: {}'.format(dataset_path))
            print(psutil.virtual_memory())
        assert os.path.exists(dataset_path), 'Cannot access dataset path: {}'.format(dataset_path)
        self.dataset_path = dataset_path
        self.query_filepath = os.path.join(dataset_path, query_filename)
        assert os.path.exists(self.query_filepath), 'Cannot access query file: {}'.format(self.query_filepath)
        self.transform = transform
        self.set_transform = set_transform
        self.max_elems = max_elems
        self.n_points = 4096    # pointclouds in the dataset are downsampled to 4096 points

        cached_query_filepath = os.path.splitext(self.query_filepath)[0] + '_cached.pickle'
        if not os.path.exists(cached_query_filepath):
            # Pre-process query file
            self.queries = self.preprocess_queries(self.query_filepath, cached_query_filepath)
        else:
            print('Loading preprocessed query file: {}...'.format(cached_query_filepath))
            with open(cached_query_filepath, 'rb') as handle:
                # key:{'query':file,'positives':[files],'negatives:[files], 'neighbors':[keys]}
                self.queries = pickle.load(handle)

        if max_elems is not None:
            filtered_queries = {}
            for ndx in self.queries:
                if ndx >= self.max_elems:
                    break
                filtered_queries[ndx] = {'query': self.queries[ndx]['query'],
                                         'positives': self.queries[ndx]['positives'][0:max_elems],
                                         'negatives': self.queries[ndx]['negatives'][0:max_elems]}
            self.queries = filtered_queries

        print('{} queries in the dataset'.format(len(self)))

    def preprocess_queries(self, query_filepath, cached_query_filepath):
        print('Loading query file: {}...'.format(query_filepath))
        with open(query_filepath, 'rb') as handle:
            # key:{'query':file,'positives':[files],'negatives:[files], 'neighbors':[keys]}
            queries = pickle.load(handle)

        # Convert to bitarray
        for ndx in tqdm.tqdm(queries):
            queries[ndx]['positives'] = set(queries[ndx]['positives'])
            queries[ndx]['negatives'] = set(queries[ndx]['negatives'])
            pos_mask = [e_ndx in queries[ndx]['positives'] for e_ndx in range(len(queries))]
            neg_mask = [e_ndx in queries[ndx]['negatives'] for e_ndx in range(len(queries))]
            queries[ndx]['positives'] = bitarray(pos_mask)
            queries[ndx]['negatives'] = bitarray(neg_mask)

        with open(cached_query_filepath, 'wb') as handle:
            pickle.dump(queries, handle)

        return queries

    def __len__(self):
        return len(self.queries)

    def __getitem__(self, ndx):
        # Load point cloud and apply transform
        filename = self.queries[ndx]['query']
        query_pc = self.load_pc(filename)
        if self.transform is not None:
            query_pc = self.transform(query_pc)
        return query_pc, ndx

    def get_item_by_filename(self, filename):
        # Load point cloud and apply transform
        query_pc = self.load_pc(filename)
        if self.transform is not None:
            query_pc = self.transform(query_pc)
        return query_pc

    def get_items(self, ndx_l):
        # Load multiple point clouds and stack into (batch_size, n_points, 3) tensor
        clouds = [self[ndx][0] for ndx in ndx_l]
        clouds = torch.stack(clouds, dim=0)
        return clouds

    def get_positives_ndx(self, ndx):
        # Get list of indexes of similar clouds
        return self.queries[ndx]['positives'].search(bitarray([True]))

    def get_negatives_ndx(self, ndx):
        # Get list of indexes of dissimilar clouds
        return self.queries[ndx]['negatives'].search(bitarray([True]))

    def load_pc(self, filename):
        # Load point cloud, does not apply any transform
        # Returns Nx3 matrix
        file_path = os.path.join(self.dataset_path, filename)
        pc = np.fromfile(file_path, dtype=np.float64)
        # coords are within -1..1 range in each dimension
        assert pc.shape[0] == self.n_points * 3, "Error in point cloud shape: {}".format(file_name)
        pc = np.reshape(pc, (pc.shape[0] // 3, 3))
        pc = torch.tensor(pc, dtype=torch.float)
        return pc


class TrainTransform:
    def __init__(self, aug_mode):
        # 1 is default mode, no transform
        self.aug_mode = aug_mode
        if self.aug_mode == 1:
            t = [JitterPoints(sigma=0.001, clip=0.002), RemoveRandomPoints(r=(0.0, 0.1)),
                 RandomTranslation(max_delta=0.01), RemoveRandomBlock(p=0.4)]
        else:
            raise NotImplementedError('Unknown aug_mode: {}'.format(self.aug_mode))
        self.transform = transforms.Compose(t)

    def __call__(self, e):
        if self.transform is not None:
            e = self.transform(e)
        return e


class TrainSetTransform:
    def __init__(self, aug_mode):
        # 1 is default mode, no transform
        self.aug_mode = aug_mode
        self.transform = None
        t = [RandomRotation(max_theta=5, max_theta2=0, axis=np.array([0, 0, 1])),
             RandomFlip([0.25, 0.25, 0.])]
        self.transform = transforms.Compose(t)

    def __call__(self, e):
        if self.transform is not None:
            e = self.transform(e)
        return e


class RandomFlip:
    def __init__(self, p):
        # p = [p_x, p_y, p_z] probability of flipping each axis
        assert len(p) == 3
        assert 0 < sum(p) <= 1, 'sum(p) must be in (0, 1] range, is: {}'.format(sum(p))
        self.p = p
        self.p_cum_sum = np.cumsum(p)

    def __call__(self, coords):
        r = random.random()
        if r <= self.p_cum_sum[0]:
            # Flip the first axis
            coords[..., 0] = -coords[..., 0]
        elif r <= self.p_cum_sum[1]:
            # Flip the second axis
            coords[..., 1] = -coords[..., 1]
        elif r <= self.p_cum_sum[2]:
            # Flip the third axis
            coords[..., 2] = -coords[..., 2]

        return coords


class RandomRotation:
    def __init__(self, axis=None, max_theta=180, max_theta2=15):
        self.axis = axis
        self.max_theta = max_theta      # Rotation around axis
        self.max_theta2 = max_theta2    # Smaller rotation in random direction

    def _M(self, axis, theta):
        return expm(np.cross(np.eye(3), axis / norm(axis) * theta)).astype(np.float32)

    def __call__(self, coords):
        if self.axis is not None:
            axis = self.axis
        else:
            axis = np.random.rand(3) - 0.5
        R = self._M(axis, (np.pi * self.max_theta / 180) * 2 * (np.random.rand(1) - 0.5))
        if self.max_theta2 is None:
            coords = coords @ R
        else:
            R_n = self._M(np.random.rand(3) - 0.5, (np.pi * self.max_theta2 / 180) * 2 * (np.random.rand(1) - 0.5))
            coords = coords @ R @ R_n

        return coords


class RandomTranslation:
    def __init__(self, max_delta=0.05):
        self.max_delta = max_delta

    def __call__(self, coords):
        trans = self.max_delta * np.random.randn(1, 3)
        return coords + trans.astype(np.float32)


class RandomScale:
    def __init__(self, min, max):
        self.scale = max - min
        self.bias = min

    def __call__(self, coords):
        s = self.scale * np.random.rand(1) + self.bias
        return coords * s.astype(np.float32)


class RandomShear:
    def __init__(self, delta=0.1):
        self.delta = delta

    def __call__(self, coords):
        T = np.eye(3) + self.delta * np.random.randn(3, 3)
        return coords @ T.astype(np.float32)


class JitterPoints:
    def __init__(self, sigma=0.01, clip=None, p=1.):
        assert 0 < p <= 1.
        assert sigma > 0.

        self.sigma = sigma
        self.clip = clip
        self.p = p

    def __call__(self, e):
        """ Randomly jitter points. jittering is per point.
            Input:
              BxNx3 array, original batch of point clouds
            Return:
              BxNx3 array, jittered batch of point clouds
        """

        sample_shape = (e.shape[0],)
        if self.p < 1.:
            # Create a mask for points to jitter
            m = torch.distributions.categorical.Categorical(probs=torch.tensor([1 - self.p, self.p]))
            mask = m.sample(sample_shape=sample_shape)
        else:
            mask = torch.ones(sample_shape, dtype=torch.int64 )

        mask = mask == 1
        jitter = self.sigma * torch.randn_like(e[mask])

        if self.clip is not None:
            jitter = torch.clamp(jitter, min=-self.clip, max=self.clip)

        e[mask] = e[mask] + jitter
        return e


class RemoveRandomPoints:
    def __init__(self, r):
        if type(r) is list or type(r) is tuple:
            assert len(r) == 2
            assert 0 <= r[0] <= 1
            assert 0 <= r[1] <= 1
            self.r_min = float(r[0])
            self.r_max = float(r[1])
        else:
            assert 0 <= r <= 1
            self.r_min = None
            self.r_max = float(r)

    def __call__(self, e):
        n = len(e)
        if self.r_min is None:
            r = self.r_max
        else:
            # Randomly select removal ratio
            r = random.uniform(self.r_min, self.r_max)

        mask = np.random.choice(range(n), size=int(n*r), replace=False)   # select elements to remove
        e[mask] = torch.zeros_like(e[mask])
        return e


class RemoveRandomBlock:
    """
    Randomly remove part of the point cloud. Similar to PyTorch RandomErasing but operating on 3D point clouds.
    Erases fronto-parallel cuboid.
    Instead of erasing we set coords of removed points to (0, 0, 0) to retain the same number of points
    """
    def __init__(self, p=0.5, scale=(0.02, 0.33), ratio=(0.3, 3.3)):
        self.p = p
        self.scale = scale
        self.ratio = ratio

    def get_params(self, coords):
        # Find point cloud 3D bounding box
        flattened_coords = coords.view(-1, 3)
        min_coords, _ = torch.min(flattened_coords, dim=0)
        max_coords, _ = torch.max(flattened_coords, dim=0)
        span = max_coords - min_coords
        area = span[0] * span[1]
        erase_area = random.uniform(self.scale[0], self.scale[1]) * area
        aspect_ratio = random.uniform(self.ratio[0], self.ratio[1])

        h = math.sqrt(erase_area * aspect_ratio)
        w = math.sqrt(erase_area / aspect_ratio)

        x = min_coords[0] + random.uniform(0, 1) * (span[0] - w)
        y = min_coords[1] + random.uniform(0, 1) * (span[1] - h)

        return x, y, w, h

    def __call__(self, coords):
        if random.random() < self.p:
            x, y, w, h = self.get_params(coords)     # Fronto-parallel cuboid to remove
            mask = (x < coords[..., 0]) & (coords[..., 0] < x+w) & (y < coords[..., 1]) & (coords[..., 1] < y+h)
            coords[mask] = torch.zeros_like(coords[mask])
        return coords


if __name__ == '__main__':
    dataset_path = '/media/sf_Datasets/PointNetVLAD'
    query_filename = 'test_queries_baseline.pickle'

    my_dataset = OxfordDataset()

    e = my_dataset[10]

