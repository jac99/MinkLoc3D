# Author: Jacek Komorowski
# Warsaw University of Technology

import os
import configparser
import time
import numpy as np


class ModelParams:
    def __init__(self, model_params_path):
        config = configparser.ConfigParser()
        config.read(model_params_path)
        params = config['MODEL']

        self.model_params_path = model_params_path
        self.model = params.get('model')
        self.output_dim = params.getint('output_dim', 256)      # Size of the final descriptor

        # Add gating as the last step
        if 'vlad' in self.model.lower():
            self.cluster_size = params.getint('cluster_size', 64)   # Size of NetVLAD cluster
            self.gating = params.getboolean('gating', True)         # Use gating after the NetVlad

        #######################################################################
        # Model dependent
        #######################################################################

        if 'MinkFPN' in self.model:
            # Models using MinkowskiEngine
            self.mink_quantization_size = params.getfloat('mink_quantization_size')
            # Size of the local features from backbone network (only for MinkNet based models)
            # For PointNet-based models we always use 1024 intermediary features
            self.feature_size = params.getint('feature_size', 256)
            if 'planes' in params:
                self.planes = [int(e) for e in params['planes'].split(',')]
            else:
                self.planes = [32, 64, 64]

            if 'layers' in params:
                self.layers = [int(e) for e in params['layers'].split(',')]
            else:
                self.layers = [1, 1, 1]

            self.num_top_down = params.getint('num_top_down', 1)
            self.conv0_kernel_size = params.getint('conv0_kernel_size', 5)

    def print(self):
        print('Model parameters:')
        param_dict = vars(self)
        for e in param_dict:
            print('{}: {}'.format(e, param_dict[e]))

        print('')


def get_datetime():
    return time.strftime("%Y%m%d_%H%M")


def xyz_from_depth(depth_image, depth_intrinsic, depth_scale=1000.):
    # Return X, Y, Z coordinates from a depth map.
    # This mimics OpenCV cv2.rgbd.depthTo3d() function
    fx = depth_intrinsic[0, 0]
    fy = depth_intrinsic[1, 1]
    cx = depth_intrinsic[0, 2]
    cy = depth_intrinsic[1, 2]
    # Construct (y, x) array with pixel coordinates
    y, x = np.meshgrid(range(depth_image.shape[0]), range(depth_image.shape[1]), sparse=False, indexing='ij')

    X = (x - cx) * depth_image / (fx * depth_scale)
    Y = (y - cy) * depth_image / (fy * depth_scale)
    xyz = np.stack([X, Y, depth_image / depth_scale], axis=2)
    xyz[depth_image == 0] = np.nan
    return xyz


class MinkLocParams:
    """
    Params for training MinkLoc models on Oxford dataset
    """
    def __init__(self, params_path, model_params_path):
        """
        Configuration files
        :param path: General configuration file
        :param model_params: Model-specific configuration
        """

        assert os.path.exists(params_path), 'Cannot find configuration file: {}'.format(params_path)
        assert os.path.exists(model_params_path), 'Cannot find model-specific configuration file: {}'.format(model_params_path)
        self.params_path = params_path
        self.model_params_path = model_params_path
        self.model_params_path = model_params_path

        config = configparser.ConfigParser()

        config.read(self.params_path)
        params = config['DEFAULT']
        self.num_points = params.getint('num_points', 4096)
        self.dataset_folder = params.get('dataset_folder')

        params = config['TRAIN']
        self.num_workers = params.getint('num_workers', 0)
        self.batch_size = params.getint('batch_size', 128)

        # Set batch_expansion_th to turn on dynamic batch sizing
        # When number of non-zero triplets falls below batch_expansion_th, expand batch size
        self.batch_expansion_th = params.getfloat('batch_expansion_th', None)
        if self.batch_expansion_th is not None:
            assert 0. < self.batch_expansion_th < 1., 'batch_expansion_th must be between 0 and 1'
            self.batch_size_limit = params.getint('batch_size_limit', 256)
            # Batch size expansion rate
            self.batch_expansion_rate = params.getfloat('batch_expansion_rate', 1.5)
            assert self.batch_expansion_rate > 1., 'batch_expansion_rate must be greater than 1'
        else:
            self.batch_size_limit = self.batch_size
            self.batch_expansion_rate = None

        self.lr = params.getfloat('lr', 1e-3)

        self.scheduler = params.get('scheduler', 'MultiStepLR')
        if self.scheduler is not None:
            if self.scheduler == 'CosineAnnealingLR':
                self.min_lr = params.getfloat('min_lr')
            elif self.scheduler == 'MultiStepLR':
                scheduler_milestones = params.get('scheduler_milestones')
                self.scheduler_milestones = [int(e) for e in scheduler_milestones.split(',')]
            else:
                raise NotImplementedError('Unsupported LR scheduler: {}'.format(self.scheduler))

        self.epochs = params.getint('epochs', 20)
        self.weight_decay = params.getfloat('weight_decay', None)
        self.normalize_embeddings = params.getboolean('normalize_embeddings', True)    # Normalize embeddings during training and evaluation
        self.loss = params.get('loss')

        if 'Contrastive' in self.loss:
            self.pos_margin = params.getfloat('pos_margin', 0.2)
            self.neg_margin = params.getfloat('neg_margin', 0.65)
        elif 'Triplet' in self.loss:
            self.margin = params.getfloat('margin', 0.4)    # Margin used in loss function
        else:
            raise 'Unsupported loss function: {}'.format(self.loss)

        self.aug_mode = params.getint('aug_mode', 1)    # Augmentation mode (1 is default)

        self.train_file = params.get('train_file')
        self.val_file = params.get('val_file', None)

        self.eval_database_files = ['oxford_evaluation_database.pickle', 'business_evaluation_database.pickle',
                                    'residential_evaluation_database.pickle', 'university_evaluation_database.pickle']

        self.eval_query_files = ['oxford_evaluation_query.pickle', 'business_evaluation_query.pickle',
                                 'residential_evaluation_query.pickle', 'university_evaluation_query.pickle']

        assert len(self.eval_database_files) == len(self.eval_query_files)

        # Read model parameters
        self.model_params = ModelParams(self.model_params_path)

        self._check_params()

    def _check_params(self):
        assert os.path.exists(self.dataset_folder), 'Cannot access dataset: {}'.format(self.dataset_folder)

    def print(self):
        print('Parameters:')
        param_dict = vars(self)
        for e in param_dict:
            if e != 'model_params':
                print('{}: {}'.format(e, param_dict[e]))

        self.model_params.print()
        print('')

