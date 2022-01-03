# SpSeg
'Species Segregator' or SpSeg is a Machine-learning tool for species-level segregation of camera-trap images originating from wildlife census and studies. SpSeg is currently trained for Central Indian Landscape specifically. The model is build as second-step to Microsoft's [MegaDetector](https://github.com/microsoft/CameraTraps/blob/master/megadetector.md "MegaDetector"), which identifies animals, person and vehical in these images. SpSeg reads the results of MegaDetector and classifies the animal images into species (or a defined biological taxonomic level).
 
## Training data
Training dataset includes 36 species commonly encountered in camera-trap surveys in Eastern Vidarbha Landscape, Maharashtra, India:
|Species|Scientific name|Image set|Species|Scientific name|Image set|
|:------|:--------------|:-------:|:------|:--------------|:-------:|
|00_barking_deer|_Muntiacus muntjak_|7920|18_langur|_Semnopithecus entellus_|12913
|01_birds|Excluding fowls|2005|19_leopard|_Panthera pardus_|7449
|02_buffalo|_Bubalus bubalis_|7265|20_rhesus_macaque|_Macaca mulatta_|5086
|03_spotted_deer|_Axis axis_|45790|21_nilgai|_Boselaphus tragocamelus_|6864
|04_four_horned_antelope|_Tetracerus quadricornis_|6383|22_palm_squirrel|_Funambulus palmarum_ & _Funambulus pennantii_|1854
|05_common_palm_civet|_Paradoxurus hermaphroditus_|8571|23_indian_peafowl|_Pavo cristatus_|10534
|06_cow|_Bos taurus_|7031|24_ratel|_Mellivora capensis_|5751
|07_dog|_Canis lupus familiaris_|4150|25_rodents|Several mouse, rat, gerbil and vole species|4992
|08_gaur|_Bos gaurus_|14646|26_mongooses|_Urva edwardsii_ & _Urva smithii_|5716
|09_goat|_Capra hircus_|3959|27_rusty_spotted_cat|_Prionailurus rubiginosus_|1649
|10_golden_jackal|_Canis aureus_|2189|28_sambar|_Rusa unicolor_|28040
|11_hare|_Lepus nigricollis_|8403|29_domestic_sheep|_Ovis aries_|2891
|12_striped_hyena|_Hyaena hyaena_|2303|30_sloth_bear|_Melursus ursinus_|6348
|13_indian_fox|_Vulpes bengalensis_|379|31_small_indian_civet|_Viverricula indica_|4187
|14_indian_pangolin|_Manis crassicaudata_|1442|32_tiger|_Panthera tigris_|9111
|15_indian_porcupine|_Hystrix indica_|5090|33_wild_boar|_Sus scrofa_|18871
|16_jungle_cat|_Felis chaus_|4376|34_wild_dog|_Cuon alpinus_|7743
|17_jungle_fowls|Includes _Gallus gallus_, _Gallus sonneratii_ & _Galloperdix spadicea_|4760|35_indian_wolf|_Canis lupus pallipes_|553

## Training pipeline
SpSeg repository contains all the required tools to trainand test the model. Run the codes from Tools directory in the repository

**Step 1:** Run [MegaDetector](https://github.com/microsoft/CameraTraps/blob/master/megadetector.md "MegaDetector") model on Images to separate animal images. Latest model V4.1 can be downloaded from [here](https://lilablobssc.blob.core.windows.net/models/camera_traps/megadetector/md_v4.1.0/md_v4.1.0.pb "here").

`python run_tf_detector_batch.py path_to_model/md_v4.1.0.pb image_directory image_directory/output_file.json`

**Step 2:** Crop the the animals in sqaure images.

`python crop_detections.py image_directory/output_file.json path_to_crops --images-dir image_directory --detector-version "4.0" --threshold 0.8  --logdir "." --threads 25 --square-crops`

**Step 3:** Create CSV file with paths to each image in the directory alongwith a numrical identifier of the species.

`python csv_paths.py --image_folder path_to_crops --image_format jpg --output_csv ../paths/species_data.csv --net_type cnn`

**Step 4:** In windows systems, sometimes file paths do not have required extension at the end (‘file_01.’ instead of file_01.jpg). This steps removes these paths from the data (should be used cautiously).

`python test_files.py --input_csv ../paths/species_data_test.csv --output_csv ../paths/species_data_cleaned.csv`

**Step 5:** Since the number of images varry in each species class, we restricted sample size at 5000 images max for each class by randomly undersampling.

`python random_sampling.py --input_csv ../paths/species_data_test.csv --output_csv ../paths/species_data_usample.csv --sample_size 5000`

**Step 6:** Split the dataset into train, testing and validation sets. Validation data size is set to be equal to testing data size.

`python split_dataset.py --input_csv ../paths/species_data_usample.csv --output_dir ../paths/ --file_name species_data --test_per 0.15`

**Step 7:** Training CNN models from keras https://keras.io/api/applications/

`python train_cnn.py --model model_name --train_csv ../paths/species_data_train.csv --valid_csv ../paths/species_data_valid.csv --batch_size 10 --num_classes 37 --epochs 100 --input_shape 224 224 3`

**Step 8:** Calculate accuracy of CNN models

`python accuracy_cnn.py --model model_name --input_shape 224 224 3 --csv_paths ../paths/species_data_test.csv --weights ../trained_models/model_file.hdf5`

- - - -
_Dr Bilal Habib's lab at Wildlife Institute India, Dehradun, India is a [partner](https://github.com/microsoft/CameraTraps#who-is-using-the-ai-for-earth-camera-trap-tools "partner") in the development, evaluation and use of MegaDetector model. Development of SpSeg was supported by [Microsoft AI for Earth](https://www.microsoft.com/en-us/ai/ai-for-earth "Microsoft AI for Earth") (Grant ID:00138001338)_
