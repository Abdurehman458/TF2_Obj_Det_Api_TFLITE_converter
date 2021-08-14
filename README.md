# TF2 OBJect Detection Api –-> TFLITE Conversion:

# Note: 
TF 2.3 has some bugs while doing tflite conversion. TF2.4 has fixes. So make sure that TF version is >= 2.4.
Also install tensorflow2 object detection api as script imports some libraries from it. 
Here is a comprehensive guide for installing [TF2 Object Detection Api](https://tensorflow-object-detection-api-tutorial.readthedocs.io/en/latest/install.html)

# Steps: 
1. First create a SAVED MODEL by using this CMD. 
Converting Model (SAVEDMODEL) created by TF OBJ API directly wont work. 
TF has provided additional script to convert Object Detection Api 
SAVEDMODEL to TFLITE compatible SAVEDMODEL.

```
python3 export_tflite_graph_tf2.py \
    --pipeline_config_path trained_models/ssd_v2/pipeline.config \
    --trained_checkpoint_dir trained_models/ssd_v2/checkpoint \
    --output_directory tflite_models/saved_models
```
*export_tflite_graph_tf2.py is available in TF OBJ API so TF OBJ API installation is required.

2. Now run tflite_converter.py (setup path of saved model according to requirment)

'''
python3 tflite_converter.py
'''