# nnDetection Tutorial

In this tutorial, we are going through the basic steps to run an nnDetection experiment.

## 1. Data folder preparation
Your dataset folder should be formatted in such a way that the nnDetection scripts can successfully detect all the dataset subjects and the corresponding NIFTI files.

NOTE 1: nnDetection requires all the volumes and segmentation masks to be in the NIFTI format. Use the script `Hive_convert_DICOM_dataset_to_NIFTI_dataset` to convert a DICOM dataset into a NIFTI format.

NOTE 2: nnDetection requires the annotation mask to be in the form of an instance segmentation (each object to be detected is labelled with a unique, incremental value).

The standard dataset folder structure is the following:

    [Dataset_folder]
        [Subject_0]
            - Subject_0_image0.nii.gz    # Subject_0 modality 0
            - Subject_0_image1.nii.gz    # Subject_0 modality 1
            - Subject_0_mask.nii.gz      # Subject_0 instance segmentation mask
        [Subject_1]
            - Subject_1_image0.nii.gz    # Subject_1 modality 0
            - Subject_1_image1.nii.gz    # Subject_1 modality 1
            - Subject_1_mask.nii.gz      # Subject_1 instance segmentation mask
        ...

## 2. Create Pipeline File

After organizing the Dataset folder according to the standard format, you are ready to generate a *Pipeline file*, that will be later used to run all the experiment steps (Data Preparation, Preprocessing and Training).
In order to do so, you first need to prepare a JSON configuration file, specifying all the parameters and attributes for the experiment, as described in the section `Configs -> nnDetection config`. Some default config files are made available with the **Hive** package, in the **configs** section.
In addition, *root_experiment_folder* should be set as an environment variable (this will be the base folder for all the experiments that will be created). Check it with:
```
echo $root_experiment_folder
```
and, if not present, run:
```
export root_experiment_folder=/YOUR/PATH/TO/Experiments
```
To generate the *Pipeline file*, run:
```
Hive_create_pipeline --input-data-folder /PATH/TO/Dataset_folder --config-file /YOUR/CONFIG_FILE.json --task-ID 000
```
with `task_id` representing an unique identifier number for the experiment. 

Optionally, you can set the split ratio (in %, set a value between 0-100) between train and test data:
```
Hive_create_pipeline --input-data-folder /PATH/TO/Dataset_folder --config-file /YOUR/CONFIG_FILE.json --task-ID 000 --test-split 20
```
By default, 80% of the available data will be dedicated for cross-fold validation, while 20% will be reserved as testing set.

The *Pipeline file* will be available, as a *txt* file, in `root_experiment_folder/experiment_folder`, with *experiment_folder* as indicated in the config file with the  `"Experiment Name"` attribute. 
## 3. Run Pipeline
Once the *Pipeline file* is created, you are ready to run your nnDetection experiment, either with the available script, or by just copying/pasting the single commands from the *txt* file into your shell.
To run the full pipeline with the **Hive** script:
```
Hive_run_pipeline_from_file --file /YOUR/PIPELINE_FILE.txt
```

## 4. Extract Predictions
After completing the 5-fold Cross Validation, the NIFTI predictions (Segmentation and Bounding Boxes), can be extracted and saved with the command:
```bash
Hive_extract_experiment_predictions --config-file /CONFIG_FILE.json --output-experiment-folder /PATH/TO/OUTPUT/FOLDER
```

## 5. Compute COCO, FROC and Dice Evaluation Metrics
The object detection and segmentation evaluation metrics can be computed and with the following command:
```bash
nndet_compute_metric_results --config-file /PATH/TO/CONFIG_FILE.json --output-dir /OUTPUT/PATH
```
After the successful execution, the evaluation metrics will be written in an Excel Spreadsheet, together with the FROC PNG plots.