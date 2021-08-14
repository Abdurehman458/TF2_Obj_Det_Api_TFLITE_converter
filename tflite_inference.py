import tensorflow as tf
import numpy as np
print(tf.__version__)
import cv2
import pathlib

# interpreter = tf.lite.Interpreter(model_path="blazeface.tflite")
interpreter = tf.lite.Interpreter(model_path="no_quant.tflite")
# interpreter = tf.lite.Interpreter(model_path="blazeface_obj_api.tflite")

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

#print(input_details)
#print(output_details)

interpreter.allocate_tensors()

def draw_rect(image, box):
    y_min = int(max(1, (box[0] * fheight)))
    x_min = int(max(1, (box[1] * fwidth)))
    y_max = int(min(fheight, (box[2] * fheight)))
    x_max = int(min(fwidth, (box[3] * fwidth)))
    
    # draw a rectangle on the image
    cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (255, 255, 255), 2)

# for file in pathlib.Path('img').iterdir():

#     if file.suffix != '.jpg' and file.suffix != '.png':
#         continue

cap = cv2.VideoCapture("/home/arm/Videos/vlc-record-2020-08-12-13h14m54s-August-04-2020-7 am-alec2-up.mkv-.mp4")
(grabbed, frame) = cap.read()
fshape = frame.shape
fheight = fshape[0]
fwidth = fshape[1]
while(True):
    ret, img = cap.read()
    if ret:
        # img = cv2.imread(r"{}".format(file.resolve()))
        orig_img=img
        img=img.astype(np.float32) #for float model
        img = img/255.0 
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # new_img = cv2.resize(img, (300, 300)) 
        new_img = cv2.resize(img, (128, 128))
        
        interpreter.set_tensor(input_details[0]['index'], [new_img])

        interpreter.invoke()
        rects = interpreter.get_tensor(
            output_details[0]['index'])

        # scores = interpreter.get_tensor(        #for balazeface
        #     output_details[1]['index'])
        scores = interpreter.get_tensor(
            output_details[2]['index'])
        # print(scores[0])
        # exit()
        for index, score in enumerate(scores[0]):
            if score > 0.5:
                draw_rect(orig_img,rects[0][index])
                # print(score)
            
        cv2.imshow("image", orig_img)
        # cv2.imshow("small",new_img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break
cap.release()
cv2.destroyAllWindows()