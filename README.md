# On this page
[What is SpSeg?](https://github.com/bhlab/SpSeg#what-is-spseg)  
&emsp;  &emsp;  [SpSeg models](https://github.com/bhlab/SpSeg#spseg-models)  
&emsp;  &emsp;  [Training data](https://github.com/bhlab/SpSeg#training-data)  
[How to Use SpSeg?](https://github.com/bhlab/SpSeg#how-to-use-spseg)  
&emsp;  &emsp;  [Setting up the tools](https://github.com/bhlab/SpSeg#setting-up-the-tools)  
&emsp;  &emsp;  &emsp;  [Camera-Trap image data organization](https://github.com/bhlab/SpSeg#camera-trap-image-data-organization)  
&emsp;  &emsp;  [Running SpSeg](https://github.com/bhlab/SpSeg#running-spseg)  
[How to use the output to process the images?](https://github.com/bhlab/SpSeg#how-to-use-the-output-to-process-the-images)  
[Further development](https://github.com/bhlab/SpSeg#further-development)  
[Citing SpSeg](https://github.com/bhlab/SpSeg#citation)  and [License information](https://github.com/bhlab/SpSeg#license)

# What is SpSeg?
'SpSeg' (short for 'Species Segregator) is a Machine-learning tool for species-level segregation of camera-trap images originating from wildlife census and
studies. SpSeg is currently trained specifically for the Central Indian landscape . 

>SpSeg is part of Microsoft's [MegaDetector](https://github.com/microsoft/CameraTraps/blob/master/megadetector.md "MegaDetector") ecosystem and works as a second-step to species level segregation of camera-trap images. The approach in MegaDetector pipeline is to first classify the images into **Animal**, **Person**, **Vehicle** and **Blank**, followed by a fine level classification of **Animal** images into useful taxonomic classes.

📝  Check out this detailed overview of the approach in a [Medium article](https://medium.com/microsoftazure/accelerating-biodiversity-surveys-with-azure-machine-learning-9be53f41e674) and an [explainer](http://dmorris.net/misc/cameratraps/ai4e_camera_traps_overview) by [Dan Morris](https://github.com/agentmorris "Dan Morris") on why this two-step process. 

<img src="https://www.dropbox.com/s/0pxb8nt9h65de6b/Indian_fox_cr.jpg?raw=1" height="400"> <br>
*SpSeg model identifies an Indian fox within the bounding box.*  

## SpSeg models
We initially trained different CNN architecture models from [keras](https://keras.io/api/applications/) (see the [results](https://github.com/bhlab/SpSeg#results-of-initial-trained-models) below). We found two models performing well - **`SpSeg_x.hdf5`** and **`SpSeg_r152v2.hdf5`** - and have tested those on various field datasets. **`SpSeg_x.hdf5`** is an Xception model that achieved 88.9% test accuracy, and **`SpSeg_r152v2.hdf5`** is a ResNet152v2 model that achieved 89.2% test accuracy. Find the models from a publicly shared [SpSeg_Models](https://forms.gle/6Ah5uPbLdSdLuEwS8 "SpSeg_Models") Google Drive folder.
### *Results of initial trained models*
The models in different architectures were trained for 100 ephocs each with the same training and test dataset.
|Architecture|avg top-1 acc|Architecture|avg top-1 acc
|:-----------|:------------|:-----------|:------------|
|ResNet152v2|89.2%|ResNet101|88.8%|
|InceptionResNetv2|89.2%| ResNet50|88.5%|
|ResNet101v2|89.1%|ResNet50v2|87.5%|
|Xception|88.9%| ResNet152|82.0%|

## Training data
SpSeg models can currently segregate 36 commonly encountered species (or taxonomic class) in camera-trap surveys in the Eastern Vidarbha Landscape, Maharashtra, India. A maximum of 5000 images were randomly selected for each species from our [image dataset](#table1) to train the models. The models were trained with 70% of the selected data, while 15% images were used for validation and remaining 15% were used for testing. Therefore, species with more than 5000 images in our image dataset (table below) have better accuracy, except sexually dimorphic species like Nilgai.
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

---
# How to Use SpSeg?
**The best place to start is the** [**User Guide of SpSeg**](https://github.com/bhlab/SpSeg/raw/master/SpSeg_Manual_1.0.pdf), which covers the process in details. 

SpSeg model (either **`SpSeg_x.hdf5`** or **`SpSeg_r152v2.hdf5`**) reads the results of MegaDetector and classifies the animal images into species (or a defined biological taxonomic level). Therefore, both the models (detector and classifier) are required to run simultaneously on the images.

📝 You can find more about how you can run MegaDetector on your images from this official [MegaDetecor GitHub page](https://github.com/microsoft/CameraTraps/blob/main/megadetector.md).

⚠️ Keep in mind that the SpSeg is developed with MDv4. Although MDv5 has been launched, at this moment <ins>we recommend to stick with MDv4.</ins> We shall integrate SpSeg tools with MDv5 in future updates.

## Setting up the tools
These instructions are quite similar to the [instruction for MegaDetector installation](https://github.com/microsoft/CameraTraps/blob/main/megadetector.md#using-the-model), where more details could be found. For technical details check the official websites of the mentioned software.

#### Step 1. Installation
Download and install [Anaconda](https://www.anaconda.com/products/individual) (or [Miniconda](https://docs.conda.io/en/latest/miniconda.html)). Installing [Git](https://git-scm.com/downloads) is optional, but recommended to keep your repo updated. The latest [NVIDIA drivers](https://www.nvidia.com/download/index.aspx) need to be checked if you are using a GPU.

#### Step 2. Access the detector and classifier models
Download [MegaDetector model v4.1.0](https://lilawildlife.blob.core.windows.net/lila-wildlife/models/camera_traps/megadetector/md_v4.1.0/md_v4.1.0.pb) and [SpSeg models](https://drive.google.com/drive/folders/1u4wLhY8N_ovPzN8nZp4cqUxEVRAGYg6v?usp=sharing) and keep at an accessible location. These instructions assume that the models are downloaded to a folder called `c:\spseg_models`.

#### Step 3. Clone git repo and set Python environment
There is no need to setup a separate MegaDetector environment, which is incorporated in the codes here. We are giving the instructions for Windows machines only, which is expected to work on Linux machines in a similar way. The environment is not tested on Mac.

Open Anaconda Prompt and make sure you have not opened an Anaconda Powershell Prompt or Windows Command Prompt. It should look something like this with the name of environment in the parenthesis:  
<img src="https://www.dropbox.com/s/rw3vrextya6ysvd/anaconda1.png?raw=1" height="50">  
Run the following commands one by one to download SpSeg repo from GitHub:
```batch
mkdir c:\git
cd c:\git
git clone https://github.com/bhlab/SpSeg
```
The commnads above are suggestions. `git` command will work at any directory and you can download GitHub repo wherever you find it convinient. Alternatively, you can download the SpSeg repo as zip folder and unzip it to a desired location.

⚠️ If your computer does not have a CUDA-supported GPU--> Go to the local SpSeg repo folder and open `environment_SpSeg.yml` file in notepad, change the dependency `tensorflow-gpu` to `tensorflow` and save the file.

Next, you need to enter where SpSeg repo is located using `cd` command:
```batch
cd c:\git\spseg
conda env create --file environment_SpSeg.yml
```
You are all set to run SpSeg! 🎉

### Camera-Trap image data organization
We assume that your camera-trap data was organized in either of the following directory structures:  
&emsp;  &emsp;  dataDir/Station   
&emsp;  &emsp;  dataDir/Station/Camera  
The first structure arrives when a single camera was placed at the station, while the second structure arrives when A and B cameras were placed facing each other. The directory before Station ID (dataDir) could include site ID, range ID and/or block ID.

## Running SpSeg
Next time when you open Anaconda Prompt, first you need to activate SpSeg environment:
```batch
conda activate SpSeg
```
and then locate the SpSeg tools folder.
```batch
cd c:\git\spseg\tools
```
Which should look something like this:  
<img src="https://www.dropbox.com/s/e0p3btrav0mt0ct/SpSeg_env.PNG?raw=1" height="50">  

The main tool to use SpSeg models is **`run_spseg.py`**  

Once you are in the right setup (environment and tools folder), the following command calls detector and classifier models and runs over all the images:
```python
python run_spseg.py "C:\spseg_models\md_v4.1.0.pb" "D:\dataDir" --recursive --spseg_model " C:\spseg_models\SpSeg_r152v2.hdf5" "D:\dataDir\data_output.json"
```

**"D:\dataDir"** is the location (local or on network server) where your organized camera-trap images are stored. You can set the path and name of the output file at **"D:\dataDir\data_output.json"**.

The above command returns a *.json* directory for each image with detector tags for animal, person or vehicle, and species tags for animals (both the tags with a confident value). 

💡 The image paths are retuned as absolute, which creates mismatches during further processing of the data specially if processing is carried out on a different system. Therefore, <ins>the best practice is to get the image paths as relative and keep the output file within image folder</ins>.

Relative paths in the output can be returned by setting the flag for it:
```python
python run_spseg.py "C:\spseg_models\md_v4.1.0.pb" "C:\dataDir" --recursive --spseg_model " C:\spseg_models\SpSeg_ r152v2.hdf5" "C:\dataDir\data_output.json" -- output_relative_filenames
```

---
# How to use the output to process the images?
AI-tools are not 100% accurate. Therefore, a workflow is required to process the output and correct for the errors. Currently, there are two approaches to process the output and finalize the identification/segregation of the images for further analyses-
1. Using [Timelapse](http://saul.cpsc.ucalgary.ca/timelapse) to review the images <ins>(only for Microsoft Windows systems)</ins>, 
2. Using an Image file manager system like Windows File Explorer, [ExifPro](https://github.com/mikekov/ExifPro), [IrfanView](https://www.irfanview.com/) or [FastStone](https://www.faststone.org/).

Detailed instructions on these workflows are provided in the [**User Guide of SpSeg**](https://github.com/bhlab/SpSeg/raw/master/SpSeg_Manual_1.0.pdf).

---
## Further development
We have in pipeline- 1. integration with MegaDetector version 5. 2. developement of DigiKam based workflow, 3. Further refinement of the SpSeg models (particularlly for dimorphic ungulate species and species with small dataset), and 4. Addition of more species, e.g. we missed out on the elephant in the current models.

So stay tuned!

Also, if you have any suggestion for additional species or would like to add to training dataset, reach out to us (bh@wii.gov.in) to Participate!

---
### Citation
Cite the use of SpSeg models as:
>Shrotriya, Shivam; Guthula, Venkanna Babu; Mondal, Indranil; and Habib, Bilal (2022). SpSeg User's Guide, version 1.0. TR No. 2022/29. Wildlife Institute of India, Dehradun, India.

And, don't forget to cite MegaDetector (without which SpSeg won't even exist):
>Beery, Sara; Morris, Dan; and Yang, Siyu (2019). Efficient Pipeline for Camera Trap Image Review. arXiv preprint arXiv:1907.06772

### License
SpSeg models and tools are licensed under the [Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International license](https://creativecommons.org/licenses/by-nc-sa/4.0/). Find details on the terms of the license [here](https://github.com/bhlab/SpSeg/edit/master/License.md).

- - - -
_Dr Bilal Habib's lab ([BHlab](https://bhlab.in/)) at [Wildlife Institute India](www.wii.gov.in), Dehradun, India is a [collaborator](https://github.com/microsoft/CameraTraps#who-is-using-the-ai-for-earth-camera-trap-tools "collaborator") in the development, evaluation and use of MegaDetector model. Development of SpSeg was supported by [Microsoft AI for Earth](https://www.microsoft.com/en-us/ai/ai-for-earth "Microsoft AI for Earth") (Grant ID:00138001338). We thank Mahrastra Forest Department for funding long term research in the state of Maharashtra. Kathan Bandopadhyay helped in preparing training dataset. The researchers and volunteers are thanked for their time for reviewing and testing the tools in real workfow._
