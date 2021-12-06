"""
Script for under sampling CNN training images:
Example commad: python random_sampling.py --input_csv ../paths/species_data_v2.csv --output_dir ../paths/ --filename
 ../paths/species_data_v2_usample.csv --sample_size 5000
"""
import pandas as pd
import argparse


def under_sampling(args):
    csv_file = args.input_csv
    data = pd.read_csv(csv_file)
    columns = ['path', 'class']
    data.columns = columns
    classes = data['class'].unique()
    print("Input data shape: ", data.shape)
    print("Id, Images")

    # empty dataframe
    final_data = pd.DataFrame(columns=columns)
    for cls in classes:
        cls_sample = data[data['class'] == cls]
        cls_size = len(cls_sample)
        print(cls, cls_size)
        sample_size = args.sample_size
        if cls_size > sample_size:
            final_sample = cls_sample.sample(sample_size)
        else:
            final_sample = cls_sample
        final_data = pd.concat([final_data, final_sample])

    print("Output data shape: ", final_data.shape)

    # save dataframe as csv
    if args.output_csv:
        out_csv_file = args.output_csv
    else:
        out_csv_file = csv_file

    final_data.to_csv(out_csv_file, mode='w', index=False, header=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_csv", type=str, help="Input csv file")
    parser.add_argument("--output_csv", type=str, help="Output directory where sampled paths/images save"
                        , default="../paths")
    parser.add_argument("--sample_size", type=int, help="Size of sample", default=0.15)
    args = parser.parse_args()
    under_sampling(args)
