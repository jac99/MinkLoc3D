# Author: Jacek Komorowski
# Warsaw University of Technology

import argparse
import torch

from training.trainer import do_train
from misc.utils import MinkLocParams
from datasets.dataset_utils import make_dataloaders


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train Minkowski Net embeddings using BatchHard negative mining')
    parser.add_argument('--config', type=str, required=True, help='Path to configuration file')
    parser.add_argument('--model_config', type=str, required=True, help='Path to the model-specific configuration file')
    parser.add_argument('--debug', dest='debug', action='store_true')
    parser.set_defaults(debug=False)
    parser.add_argument('--visualize', dest='visualize', action='store_true')
    parser.set_defaults(visualize=False)

    args = parser.parse_args()
    print('Training config path: {}'.format(args.config))
    print('Model config path: {}'.format(args.model_config))
    print('Debug mode: {}'.format(args.debug))
    print('Visualize: {}'.format(args.visualize))

    params = MinkLocParams(args.config, args.model_config)
    params.print()

    if args.debug:
        torch.autograd.set_detect_anomaly(True)

    dataloaders = make_dataloaders(params, debug=args.debug)
    do_train(dataloaders, params, debug=args.debug, visualize=args.visualize)
