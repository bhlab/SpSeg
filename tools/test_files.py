"""
Script for cleaning paths in csv file
Example: python test_files.py --input_csv ..\paths\species_data_test.csv --output_csv ..\paths\species_data_test2.csv
"""

import numpy as np
import argparse


def clean_csv(args):
    csv_file = args.input_csv
    data = np.loadtxt(csv_file, delimiter=',', dtype='str')
    print("Input data shape: ", data.shape)
    X = data[:, 0]

    indexes = []
    for j, i in enumerate(X):
        if i[-3:] != "jpg":
            indexes.append(j)

    print(len(indexes), " paths are not having extension at the end")
    data2 = np.delete(data, indexes, 0)
    print("Output data shape: ", data2.shape)

    if args.output_csv:
        out_csv_file = args.output_csv
    else:
        out_csv_file = csv_file

    np.savetxt(out_csv_file, data2, delimiter=",", fmt='%s')
    print("done")


# def clean_rasters(args):
    # _image = gdal.Open("D:/DIgiKam_test/species_crops\WILD_DOG\TATR_19_BL3_597_A_I__00172.JPG___crop00_mdv4.0.jpg")
    # _image = np.array(_image.ReadAsArray())
    # print(_image.shape)
    # _image
    #
    # file_name = "../paths/uc_cnn_valid.csv"
    # with open(file_name, 'r', newline='\n') as csvfile:
    #     plots = csv.reader(csvfile, delimiter=',')
    #     all_rows = []
    #     count = 0
    #     for row in plots:
    #         _image = gdal.Open(row[0])
    #         _image = np.array(_image.ReadAsArray())
    #
    #         if _image.shape[0] == 3 and _image.shape[1] == 256 and _image.shape[2] == 256:
    #             all_rows.append([row[0], row[1]])
    #             count += 1
    #     print(count)
    #
    # filename = "../paths/uc_cnn_valid2.csv"
    # with open(filename, 'w', newline="\n") as csvfile:
    #         csvwriter = csv.writer(csvfile)
    #         csvwriter.writerows(all_rows)
    # print("CSV file created")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_csv", type=str, help="Input csv file")
    parser.add_argument("--output_csv", type=str, help="Output directory where train, test and validation"
                                                       " datasets save")
    args = parser.parse_args()
    clean_csv(args)



