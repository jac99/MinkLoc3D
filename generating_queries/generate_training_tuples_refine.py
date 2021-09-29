# PointNetVLAD datasets: based on Oxford RobotCar and Inhouse
# Code adapted from PointNetVLAD repo: https://github.com/mikacuy/pointnetvlad

import os
import pandas as pd
import argparse
import tqdm

# Import test set boundaries
from generating_queries.generate_test_sets import P1, P2, P3, P4, P5, P6, P7, P8, P9, P10, check_in_test_set
from generating_queries.generate_training_tuples_baseline import construct_query_dict

# Test set boundaries
P = [P1, P2, P3, P4, P5, P6, P7, P8, P9, P10]


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate Refined training dataset')
    parser.add_argument('--dataset_root', type=str, required=True, help='Dataset root folder')
    args = parser.parse_args()
    print('Dataset root: {}'.format(args.dataset_root))

    assert os.path.exists(args.dataset_root), f"Cannot access dataset root folder: {args.dataset_root}"
    base_path = args.dataset_root

    runs_folder = "inhouse_datasets/"
    filename = "pointcloud_centroids_10.csv"
    pointcloud_fols = "/pointcloud_25m_10/"

    all_folders = sorted(os.listdir(os.path.join(base_path, runs_folder)))

    folders = []
    index_list = range(5, 15)
    for index in index_list:
        folders.append(all_folders[index])

    print(folders)

    ####Initialize pandas DataFrame
    df_train = pd.DataFrame(columns=['file', 'northing', 'easting'])

    for folder in tqdm.tqdm(folders):
        df_locations = pd.read_csv(os.path.join(base_path, runs_folder, folder, filename), sep=',')
        df_locations['timestamp'] = runs_folder + folder + pointcloud_fols + df_locations['timestamp'].astype(str) + '.bin'
        df_locations = df_locations.rename(columns={'timestamp': 'file'})
        for index, row in df_locations.iterrows():
            if check_in_test_set(row['northing'], row['easting'], P):
                continue
            else:
                df_train = df_train.append(row, ignore_index=True)

    print(len(df_train['file']))

    ##Combine with Oxford data
    runs_folder = "oxford/"
    filename = "pointcloud_locations_20m_10overlap.csv"
    pointcloud_fols = "/pointcloud_20m_10overlap/"

    all_folders = sorted(os.listdir(os.path.join(base_path, runs_folder)))

    folders = []
    index_list = range(len(all_folders) - 1)
    for index in index_list:
        folders.append(all_folders[index])

    print(folders)

    for folder in folders:
        df_locations = pd.read_csv(os.path.join(base_path, runs_folder, folder, filename), sep=',')
        df_locations['timestamp'] = runs_folder + folder + pointcloud_fols + df_locations['timestamp'].astype(str) + '.bin'
        df_locations = df_locations.rename(columns={'timestamp': 'file'})
        for index, row in df_locations.iterrows():
            if check_in_test_set(row['northing'], row['easting'], P):
                continue
            else:
                df_train = df_train.append(row, ignore_index=True)

    print("Number of training submaps: " + str(len(df_train['file'])))
    # ind_nn_r is a threshold for positive elements - 12.5 is in original PointNetVLAD code for refined dataset
    construct_query_dict(df_train, base_path, "training_queries_refine.pickle", ind_nn_r=12.5)
