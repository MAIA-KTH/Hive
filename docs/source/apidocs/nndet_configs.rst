nnDetection config
========================================================


.. jsonschema:: configs/nnDet_config_template.json
.. code-block:: json

    {
        "Experiment Name": "Retina UNet",
        "Seed": 12345,
        "label_suffix": "_mask.nii.gz",
        "Modalities": {
            "_CT.nii.gz": "CT"
        },
        "label_dict": {
            "0": "ABD_Lymph_Node"
        },
        "n_folds": 5,
        "FileExtension": ".nii.gz"
    }
