"""
Reads SpSeg output json or Timelapse csv and sorts images in folder path
Author: Shivam Shrotriya
Date: 30-07-2022
"""
import argparse
import os
import json
import shutil
import pandas as pd

from tqdm import tqdm


# %% Main function
def sort_images(image_dir, output_file, output_dir, species=None, exclude=None):

    def create_dir(out_dir, folder, file_n):
        path = os.path.join(out_dir, folder)
        if not os.path.exists(path):
            os.mkdir(path)
        path = os.path.join(path, file_n)

        return path

    if output_file.endswith('.json'):
        # %% Read input
        print('Loading SpSeg output')

        with open(output_file, 'r') as f:
            input_data = json.load(f)

        # %% Build internal mappings
        class_keys = input_data['classification_categories']
        images = input_data['images']
        print('Processing input data on {} images'.format(len(images)))

        for image in tqdm(images):
            # print('file:',image['file'])
            file = image['file']
            im_file = os.path.join(image_dir, file)
            file_name = os.path.basename(file)
            file_path = os.path.dirname(file)
            out_path = os.path.join(output_dir, file_path)
            if not os.path.exists(out_path):
                os.makedirs(out_path)

            try:
                if len(image['detections']) == 0:
                    # print('{} is empty'.format(file))
                    copy_dir = create_dir(out_path, 'blank', file_name)
                    shutil.copyfile(im_file, copy_dir)
                else:
                    for detect in image['detections']:
                        if detect['category'] == "1":
                            # print('{} has animal'.format(file))
                            if detect['conf'] >= 0.85:
                                cls = detect['classifications'][0]
                                # print('{} has animal: {}'.format(file,class_keys[cls[0]]))
                                copy_dir = create_dir(out_path, class_keys[cls[0]], file_name)
                                shutil.copyfile(im_file, copy_dir)
                            else:
                                # print('{} has unknown animal'.format(file))
                                copy_dir = create_dir(out_path, 'possibly_animal', file_name)
                                shutil.copyfile(im_file, copy_dir)
                        elif detect['category'] == "2":
                            # print('{} has person'.format(file))
                            if detect['conf'] >= 0.85:
                                copy_dir = create_dir(out_path, 'person', file_name)
                                shutil.copyfile(im_file, copy_dir)
                            else:
                                copy_dir = create_dir(out_path, 'possibly_person', file_name)
                                shutil.copyfile(im_file, copy_dir)
                        elif detect['category'] == "3":
                            # print('{} has vehicle'.format(file))
                            if detect['conf'] >= 0.85:
                                copy_dir = create_dir(out_path, 'vehicle', file_name)
                                shutil.copyfile(im_file, copy_dir)
                            else:
                                copy_dir = create_dir(out_path, 'possibly_vehicle', file_name)
                                shutil.copyfile(im_file, copy_dir)
            except:
                print('{} failed to load'.format(file))
                continue

    elif output_file.endswith('.csv'):
        print('Loading Timelapse corrected CSV')
        df = pd.read_csv(output_file)
        print('Processing input data on {} entries'.format(len(df.index)))
        # print(df.columns)

        if species:
            df = df[df['Species'].isin(species)]
            print('Sorting {} images of selected species'.format(len(df.index)))
            for ind in tqdm(df.index):
                file = os.path.join(df['RelativePath'][ind], df['File'][ind])
                im_file = os.path.join(image_dir, file)
                file_name = df['File'][ind]
                file_path = df['RelativePath'][ind]
                out_path = os.path.join(output_dir, file_path)
                if not os.path.exists(out_path):
                    os.makedirs(out_path)

                copy_dir = create_dir(out_path, df['Species'][ind], file_name)
                shutil.copyfile(im_file, copy_dir)
        elif exclude:
            if any(i in ['empty', 'blank'] for i in exclude):
                df = df[~df['Empty']]
            if 'person' in exclude:
                df = df[~df['Person']]
            if 'vehicle' in exclude:
                df = df[~df['Vehicle']]
            # reducing species for exclusion
            df = df[~df['Species'].isin(exclude)]
            print('Sorting {} images after exclusion'.format(len(df.index)))

            for ind in tqdm(df.index):
                file = os.path.join(df['RelativePath'][ind], df['File'][ind])
                im_file = os.path.join(image_dir, file)
                file_name = df['File'][ind]
                file_path = df['RelativePath'][ind]
                out_path = os.path.join(output_dir, file_path)
                if not os.path.exists(out_path):
                    os.makedirs(out_path)

                if df['Empty'][ind]:
                    copy_dir = create_dir(out_path, 'blank', file_name)
                    shutil.copyfile(im_file, copy_dir)
                else:
                    if df['Person'][ind]:
                        copy_dir = create_dir(out_path, 'person', file_name)
                        shutil.copyfile(im_file, copy_dir)
                    if df['Vehicle'][ind]:
                        copy_dir = create_dir(out_path, 'vehicle', file_name)
                        shutil.copyfile(im_file, copy_dir)
                    if df['Animal'][ind]:
                        if pd.isnull(df['Species'][ind]):
                            copy_dir = create_dir(out_path, 'Other', file_name)
                            shutil.copyfile(im_file, copy_dir)
                        else:
                            copy_dir = create_dir(out_path, df['Species'][ind], file_name)
                            shutil.copyfile(im_file, copy_dir)
        else:
            for ind in tqdm(df.index):
                file = os.path.join(df['RelativePath'][ind], df['File'][ind])
                im_file = os.path.join(image_dir, file)
                file_name = df['File'][ind]
                file_path = df['RelativePath'][ind]
                out_path = os.path.join(output_dir, file_path)
                if not os.path.exists(out_path):
                    os.makedirs(out_path)

                if df['Empty'][ind]:
                    copy_dir = create_dir(out_path, 'blank', file_name)
                    shutil.copyfile(im_file, copy_dir)
                else:
                    if df['Person'][ind]:
                        copy_dir = create_dir(out_path, 'person', file_name)
                        shutil.copyfile(im_file, copy_dir)
                    if df['Vehicle'][ind]:
                        copy_dir = create_dir(out_path, 'vehicle', file_name)
                        shutil.copyfile(im_file, copy_dir)
                    if df['Animal'][ind]:
                        if pd.isnull(df['Species'][ind]):
                            copy_dir = create_dir(out_path, 'Other', file_name)
                            shutil.copyfile(im_file, copy_dir)
                        else:
                            copy_dir = create_dir(out_path, df['Species'][ind], file_name)
                            shutil.copyfile(im_file, copy_dir)
    else:
        raise ValueError('output_file specified is not a json list or a csv file')


# %% Command-line driver


def main():
    parser = argparse.ArgumentParser(
        description='Module to sort images in folders from SpSeg output')
    parser.add_argument(
        'image_dir',
        help='Path to the image directory')
    parser.add_argument(
        '--recursive',
        action='store_true',
        help='Recurse into directories, only meaningful if using --image_dir')
    parser.add_argument(
        'output_file',
        help='Path to output JSON or CSV results file, should end with a .json or .csv extension')
    parser.add_argument(
        '--output_dir',
        help='Directory for output images')
    group = parser.add_mutually_exclusive_group(required=False)  # provide either species of exclusion list
    group.add_argument(
        '--species',
        nargs="*",
        default=None,
        help='Specify species for which you need to export the data. Must be in the format of timelapse or SpSeg json output')
    group.add_argument(
        '--exclude',
        nargs="+",
        default=None,
        help='Specify folders for which you do not need to export the data. Example- empty/blank, person, vehicle, '
             'species name in the format of timelapse or SpSeg json output')

    args = parser.parse_args()

    if not os.path.isdir(args.image_dir):
        raise ValueError('{} is not a directory'.format(args.image_dir))

    if args.output_file.endswith('.json'):
        if args.species or args.exclude:
            raise ValueError('{Folder specific sorting currently works only on timelapse output in CSV format')

    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)

    sort_images(image_dir=args.image_dir,
                output_file=args.output_file,
                output_dir=args.output_dir,
                species=args.species,
                exclude=args.exclude)


if __name__ == '__main__':
    main()
