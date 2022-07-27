# SpSeg
'Species Segregator' or SpSeg is a Machine-learning tool for species-level segregation of camera-trap images originating from wildlife census and studies. SpSeg is currently trained for the Central Indian landscape specifically. The model is build as second-step to Microsoft's [MegaDetector](https://github.com/microsoft/CameraTraps/blob/master/megadetector.md "MegaDetector"), which identifies animals, person and vehicle in these images. SpSeg reads the results of MegaDetector and classifies the animal images into species (or a defined biological taxonomic level).

### *Kindly note:*
The tool is currently under developement and the instruction for installation and use on a new data are not shared yet. One can use the current 'environment_multimodel.yml' to setup an Anaconda environment and find the models from a publicly shared [SpSeg_Models](https://drive.google.com/drive/folders/1u4wLhY8N_ovPzN8nZp4cqUxEVRAGYg6v?usp=sharing "SpSeg_Models") Google Drive folder to run on a new dataset at their own risk. There is no need to setup a separate MegaDetector environment, which is incorporated in the codes here. However, [MegaDetector model v4.1.0](https://lilablobssc.blob.core.windows.net/models/camera_traps/megadetector/md_v4.1.0/md_v4.1.0.pb "MegaDetector model v4.1.0") is required to obtain images with 'Animal' tags and the bounding boxes. 

We further plan to 1) train a couple of EfficientNet models with PyTorch, 2) finalize model based on the top performing models in a multi-model approach, and 3) share the tools to use in camera-trap studies in practical ways.

## Results of initial trained models
The models in different architectures were trained for 100 ephocs each with the same training and test dataset. So far we have achieved the highest test accuracy for **ResNet152v2** and **InceptionResNetv2** at 89.2%.
|Architecture|avg top-1 acc|Architecture|avg top-1 acc
|:-----------|:------------|:-----------|:------------|
|Xception|88.9%|  |  |
|VGG16|3.4%|VGG19|3.3%|
|ResNet50|88.5%|ResNet50v2|87.5%|
|ResNet101|88.8%|ResNet101v2|89.1%|
|ResNet152|82.0%|ResNet152v2|89.2%|
|InceptionResNetv2|89.2%|  |  |

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

- - - -
_Dr Bilal Habib's lab at Wildlife Institute India, Dehradun, India is a [partner](https://github.com/microsoft/CameraTraps#who-is-using-the-ai-for-earth-camera-trap-tools "partner") in the development, evaluation and use of MegaDetector model. Development of SpSeg was supported by [Microsoft AI for Earth](https://www.microsoft.com/en-us/ai/ai-for-earth "Microsoft AI for Earth") (Grant ID:00138001338)_
