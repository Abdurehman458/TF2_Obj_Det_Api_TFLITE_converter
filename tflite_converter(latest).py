##########instructions######################
""" use virtual env to convert to tflite models
! source env/bin/activate 
! python3 tflite_converter.py """

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'    # Suppress TensorFlow logging

import tensorflow as tf
tf.get_logger().setLevel('ERROR')           # Suppress TensorFlow logging (2)

print(tf.__version__)

from object_detection.utils import label_map_util
from object_detection.utils import config_util
from object_detection.utils import visualization_utils as vis_util

import numpy as np
import tqdm
tf_models = os.path.join(os.getcwd(), 'tflite_models')
# tr_models = os.path.join(os.getcwd(), 'trained_models')
 
# MODELS_DIR = os.path.join(tr_models, 'ssd_v2')

# PATH_TO_SAVED_MODEL = os.path.join(tf_models, 'blaze_saved_model/saved_model')
PATH_TO_SAVED_MODEL = os.path.join(tf_models, 'ssd_v2/saved_model')

# LABEL_FILENAME = 'face_label_map.pbtxt'
# PATH_TO_LABELS = os.path.join(MODELS_DIR, LABEL_FILENAME)

# Dynamic Range Quantization
# converter = tf.lite.TFLiteConverter.from_saved_model(PATH_TO_SAVED_MODEL,signature_keys=['serving_default'])
# converter.optimizations = [tf.lite.Optimize.DEFAULT]
# # converter.allow_custom_ops = True
# converter.experimental_new_converter = True
# converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS,
#                                        tf.lite.OpsSet.SELECT_TF_OPS]
# tflite_model = converter.convert()
# with open('no_quant.tflite', 'wb') as f:
#   f.write(tflite_model)


from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.mobilenet import preprocess_input
from tensorflow.keras.preprocessing import image as imagge

img_dir = os.path.join(os.getcwd(), 'img')
images = []

def rep_data_gen():
    for img in tqdm.tqdm(os.listdir(img_dir)):
        img = os.path.join(img_dir, img)
        img = imagge.load_img(img, target_size=(300, 300))
        img = imagge.img_to_array(img)
        img = tf.keras.applications.mobilenet.preprocess_input(img)
        # print(type(img),img.shape,img)
        # exit()

        # img = img/255
        # img = img.astype(np.float32)
        images.append(img)

    # print(images.shape) # a is np array of 160 3D images
    for input_value in tf.data.Dataset.from_tensor_slices(images).batch(1).take(100):
        # print(input_value)
        yield [input_value]

##### Uint8 Quantization #####

# converter = tf.lite.TFLiteConverter.from_saved_model(PATH_TO_SAVED_MODEL)
# converter.optimizations = [tf.lite.Optimize.DEFAULT]
# converter.allow_custom_ops = True
# converter.representative_dataset = rep_data_gen
# # Ensure that if any ops can't be quantized, the converter throws an error
# converter.experimental_new_converter = True
# converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8,
#             tf.lite.OpsSet.TFLITE_BUILTINS, tf.lite.OpsSet.SELECT_TF_OPS]
# # Set the input and output tensors to uint8 (APIs added in r2.3)
# converter.inference_input_type = tf.uint8
# # converter.inference_output_type = tf.uint8
# tflite_model_quant = converter.convert()
# with open('ssd_v2_quant.tflite', 'wb') as f:
#   f.write(tflite_model_quant)



def apply_nms(output_dict, iou_thresh=0.5, score_thresh=0.6):
    q = 1 # no of classes
    num = int(output_dict['num_detections'])
    boxes = np.zeros([1, num, q, 4])
    scores = np.zeros([1, num, q])
    # val = [0]*q
    for i in range(num):
        # indices = np.where(classes == output_dict['detection_classes'][i])[0][0]
        boxes[0, i, output_dict['detection_classes'][i], :] = output_dict['detection_boxes'][i]
        scores[0, i, output_dict['detection_classes'][i]] = output_dict['detection_scores'][i]
    nmsd = tf.image.combined_non_max_suppression(boxes=boxes,
                                                 scores=scores,
                                                 max_output_size_per_class=num,
                                                 max_total_size=num,
                                                 iou_threshold=iou_thresh,
                                                 score_threshold=score_thresh,
                                                 pad_per_class=False,
                                                 clip_boxes=False)
    valid = nmsd.valid_detections[0].numpy()
    output_dict = {
                   'detection_boxes' : nmsd.nmsed_boxes[0].numpy()[:valid],
                   'detection_classes' : nmsd.nmsed_classes[0].numpy().astype(np.int64)[:valid],
                   'detection_scores' : nmsd.nmsed_scores[0].numpy()[:valid],
                   }
    return output_dict
def make_and_show_inference(img, interpreter, input_details, output_details, category_index, nms=False, score_thresh=0.6, iou_thresh=0.5):
    
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_rgb = cv2.resize(img_rgb, (128, 128), cv2.INTER_AREA)
    img_rgb = img_rgb.reshape([1, 128, 128, 3])
    # img_rgb = img_rgb.astype(np.float32)
    # img_rgb = img_rgb/255
    interpreter.set_tensor(input_details[0]['index'], img_rgb)
    interpreter.invoke()
    x1=interpreter.get_tensor(output_details[0]['index'])[0]
    x2=interpreter.get_tensor(output_details[1]['index'])[0]
    # print(x1.shape, x2.shapes)
    output_dict = get_output_dict(img_rgb, interpreter, output_details, nms, iou_thresh, score_thresh)
    # Visualization of the results of a detection.
    vis_util.visualize_boxes_and_labels_on_image_array(
    img,
    output_dict['detection_boxes'],
    output_dict['detection_classes'],
    output_dict['detection_scores'],
    category_index,
    use_normalized_coordinates=True,
    min_score_thresh=score_thresh,
    line_thickness=3)

def get_output_dict(image, interpreter, output_details, nms=False, iou_thresh=0.5, score_thresh=0.6):
    output_dict = {
                   'detection_boxes' : interpreter.get_tensor(output_details[0]['index'])[0],
                   'detection_classes' : interpreter.get_tensor(output_details[1]['index'])[0],
                   'detection_scores' : interpreter.get_tensor(output_details[2]['index'])[0],
                   'num_detections' : interpreter.get_tensor(output_details[3]['index'])[0]
                   }

    output_dict['detection_classes'] = output_dict['detection_classes'].astype(np.int64)
    if nms:
        output_dict = apply_nms(output_dict, iou_thresh, score_thresh)
    return output_dict

def create_category_index(label_path='coco_labels.txt'):
    f = open(label_path)
    category_index = {}
    for i, val in enumerate(f):
        if i != 0:
            val = val[:-1]
            if val != '???':
                category_index.update({(i-1): {'id': (i-1), 'name': val}})
            
    f.close()
    return category_index

# interpreter = tf.lite.Interpreter(model_path="ssd_v2_quant.tflite")
interpreter = tf.lite.Interpreter(model_path="blazeface_quant.tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
print(input_details)
# exit()

category_index = create_category_index()

import cv2
cap = cv2.VideoCapture("facetest.mp4")

while(True):
    ret, img = cap.read()
    if ret:
        make_and_show_inference(img, interpreter, input_details, output_details, category_index)
        cv2.imshow("image", img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break
    
cap.release()
cv2.destroyAllWindows()

# print(output_details)