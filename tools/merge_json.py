"""
Split json outputs from MegaDetector's batch process can be merged as per specified folders
Author: Shivam Shrotriya
Date: 03-08-2022
"""
# %% Constants and imports

import argparse
import json
import os

from tqdm import tqdm


# %% Main function

def get_info_(j_folder):
    file_list_ = []

    for root, dirs, files in os.walk(j_folder):
        for file in files:
            if file.endswith(".json"):
                file_list_.append(os.path.join(root, file))
    #print(file_list_)

    return file_list_


def merge_json_(file_list_):
    print('Processing {} json files'.format(len(file_list_)))
    detection_results = []

    for file_ in tqdm(file_list_):
        with open(file_, 'r') as f:
            input_data = json.load(f)

        # %% Build internal mappings
        images = input_data['images']

        for image in images:
            detection_results.append(image)

    return detection_results


def write_results_to_file(file_list_, detection_results, output_file):
    """Writes results of combined json to specified json file
    """
    info_file_ = file_list_[0]
    with open(info_file_, 'r') as f:
        input_data = json.load(f)

    final_output = {
        'info': input_data['info'],
        'detection_categories': input_data['detection_categories'],
        'classification_categories': input_data['classification_categories'],
        'images': detection_results
    }
    with open(output_file, 'w') as f:
        json.dump(final_output, f, indent=1)
    print('Output file saved at {}'.format(output_file))


# %% Command-line driver


def main():
    parser = argparse.ArgumentParser(
        description='Module to merge multiple jason files into one')
    parser.add_argument(
        'j_folder',
        help='Path to folder containing json files')
    parser.add_argument(
        'output_file',
        help='Path to output JSON results file, should end with a .json extension')
    args = parser.parse_args()

    file_list_ = get_info_(j_folder=args.j_folder)

    detection_results = merge_json_(file_list_)

    write_results_to_file(file_list_, detection_results, output_file=args.output_file)


if __name__ == '__main__':
    main()
