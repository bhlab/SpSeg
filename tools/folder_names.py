import os
import shutil
import glob
import sys

dir = "\\\\192.168.205.116\\Melghat_2020\\MELGHAT 2020 SORTED DATA\\"
out_dir = "D:\\DIgiKam_test\\species_jaydeep"

# sorted_names = []
ranges = next(os.walk(dir))[1]
# print(ranges)
for range in ranges:
    grids = next(os.walk(os.path.join(dir, range)))[1]
    # print(grids)
    grid_id = 0
    for grid in grids:
        cams = next(os.walk(os.path.join(dir, range, grid)))[1]
        # print(cams)
        for cam in cams:
            if cam[0:1] == "A" or cam[0:1] == "B":
                if os.path.exists(os.path.join(dir, range, grid, cam)):
                    species = next(os.walk(os.path.join(dir, range, grid, cam)))[1]
                    # print(species)
                    for j in species:
                        if j not in ["Human", "Blank"]:
                            files = os.listdir(os.path.join(dir,range, grid, cam, j))
                            for file in files:
                                source_file = os.path.join(dir, range, grid, cam, j, file)
                                target_path = os.path.join(out_dir, j)
                                if not os.path.exists(target_path):
                                    os.mkdir(target_path)

                                if os.path.isfile(source_file):
                                    target_file = os.path.join(target_path, file)
                                    shutil.copyfile(source_file, target_file)
                                else:
                                    files2 = os.listdir(os.path.join(dir, range, grid, cam, j, file))
                                    for file2 in files2:
                                        source_file2 = os.path.join(dir, range, grid, cam, j, file, file2)
                                        target_file2 = os.path.join(target_path, file2)
                                        shutil.copyfile(source_file2, target_file2)
                                    print("Inside folder exists at ", source_file, " and copied at ", target_path)

        print("Grid number ", grid, " Copied")
    print(range, " range", " Copied")

# print(sorted_names)
# for i in sorted_names:
#     print(i)


        # if os.path.exists(os.path.join(dir, grid, "A")):
        #     if os.path.exists(os.path.join(dir, grid, "B")):
        #         my_list3 = next(os.walk(os.path.join(dir, grid, "B")))[1]
        #         for j in my_list3:
        #             if j not in sorted_names:
        #                 sorted_names.append(j)






sys.exit()
# sorted_names = []
grids = next(os.walk(dir))[1]
grid_id = 0
for grid in grids:
    if os.path.exists(os.path.join(dir, grid, "A")):
        # my_list2 = os.listdir(os.path.join(dir, grid, "A"))
        my_list2 = next(os.walk(os.path.join(dir, grid, "A")))[1]
        for i in my_list2:
            if i not in ["camel", "vehicle", "human", "blank"]:
                # print(os.path.join(dir, grid, "A"))
                files = os.listdir(os.path.join(dir, grid, "A", i))
                # print(files)
                if len(files) > 0:
                    for image in files:
                        source_path = os.path.join(dir, grid, "A", i, image)
                        target_path = os.path.join(out_dir, i)
                        if not os.path.exists(target_path):
                            os.mkdir(target_path)
                        target_path = os.path.join(target_path, image)
                        # print(source_path, target_path)
                        shutil.copyfile(source_path, target_path)

    if os.path.exists(os.path.join(dir, grid, "B")):
        # my_list2 = os.listdir(os.path.join(dir, grid, "A"))
        my_list3 = next(os.walk(os.path.join(dir, grid, "B")))[1]
        for i in my_list3:
            if i not in ["camel", "vehicle", "human", "blank"]:
                # print(os.path.join(dir, grid, "A"))
                files = os.listdir(os.path.join(dir, grid, "B", i))
                # print(files)
                if len(files) > 0:
                    for image in files:
                        source_path = os.path.join(dir, grid, "B", i, image)
                        target_path = os.path.join(out_dir, i)
                        if not os.path.exists(target_path):
                            os.mkdir(target_path)
                        target_path = os.path.join(target_path, image)
                        # print(source_path, target_path)
                        shutil.copyfile(source_path, target_path)
    grid_id += 1
    print(str(grid_id) + " Grids completed..")

#         if os.path.exists(os.path.join(dir, grid, "B")):
#             my_list3 = next(os.walk(os.path.join(dir, grid, "B")))[1]
#             for j in my_list3:
#                 if j not in sorted_names:
#                     sorted_names.append(j)
#
# print(len(sorted_names)-4)
# for i in sorted_names:
#     print(i)
# #
# for i in sorted_names.sort():
#     print(i)