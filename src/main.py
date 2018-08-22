import sys
import cv2
import os
from sys import platform
import math

try:
    from openpose import *
except:
    raise Exception('Dependencies error, check your environment installation.')

params = dict()
params["logging_level"] = 3
params["output_resolution"] = "-1x-1"
params["net_resolution"] = "-1x368"
params["model_pose"] = "BODY_25"
params["alpha_pose"] = 0.6
params["scale_gap"] = 0.3
params["scale_number"] = 1
params["render_threshold"] = 0.05
# If GPU version is built, and multiple GPUs are available, set the ID here
params["num_gpu_start"] = 0
params["disable_blending"] = False
# Ensure you point to the correct path where models are located
params["default_model_folder"] = dir_path + "/../libraries/openpose/models/"
# Construct OpenPose object allocates GPU memory
openpose = OpenPose(params)

cap = cv2.VideoCapture(0)
muscle_boom = cv2.imread("muscle_boom.png")
muscle_boom_mask = cv2.imread("muscle_boom_mask.png")
muscle_forearm = cv2.imread("muscle_forearm.png")
muscle_forearm_mask = cv2.imread("muscle_forearm_mask.png")

r = 10
t = 3

while 1:

    ret,frame = cap.read()
    keypoints,labeled_image = openpose.forward(frame, True)
    output_image = frame.copy()
    #print(keypoints,'\n')

    try:
        threshold = 0.35

        person = keypoints[0]
        if person[2][2] > threshold and person[3][2] > threshold and person[4][2] > threshold:
            RShoulder = person[2][0], person[2][1]
            RElbow = person[3][0], person[3][1]
            RWrist = person[4][0], person[4][1]
            cv2.circle(labeled_image, RShoulder, r, (255, 0, 0), thickness=t)
            cv2.circle(labeled_image, RElbow, r, (0, 255, 0), thickness=t)
            cv2.circle(labeled_image, RWrist, r, (0, 0, 255), thickness=t)

            boom_length = int(math.sqrt((RShoulder[1] - RElbow[1]) ** 2 + (RShoulder[0] - RElbow[0]) ** 2) * 1.6)
            boom_size = (boom_length, int(boom_length / 2.29))
            muscle_boom_resize = muscle_boom.copy()
            muscle_boom_mask_resize = muscle_boom_mask.copy()

            muscle_boom_resize = cv2.resize(muscle_boom_resize, boom_size)
            muscle_boom_mask_resize = cv2.resize(muscle_boom_mask_resize, boom_size)

            muscle_boom_rotate_ratio = (RElbow[1] - RShoulder[1]) / (RShoulder[0] - RElbow[0])
            muscle_boom_rotate_degree = math.atan(muscle_boom_rotate_ratio) * 180 / math.pi
            muscle_boom_x_offset = 90
            muscle_boom_rotate_M = cv2.getRotationMatrix2D(
                (int(boom_size[0] / 2 + muscle_boom_x_offset), boom_size[1] / 2), muscle_boom_rotate_degree, 1)
            muscle_boom_resize = cv2.warpAffine(muscle_boom_resize, muscle_boom_rotate_M, boom_size)
            muscle_boom_mask_resize = cv2.warpAffine(muscle_boom_mask_resize, muscle_boom_rotate_M, boom_size)

            muscle_boom_position = (int(RShoulder[0] - muscle_boom_x_offset), int(RShoulder[1]))
            output_image = cv2.seamlessClone(muscle_boom_resize, output_image, muscle_boom_mask_resize,
                                             muscle_boom_position, cv2.NORMAL_CLONE)

            # print("boom", muscle_boom_rotate_degree)

            # ------------------------------------------------------------------------------------------------------

            forearm_length = int(math.sqrt((RElbow[1] - RWrist[1]) ** 2 + (RElbow[0] - RWrist[0]) ** 2) * 0.9)
            forearm_size = (forearm_length, int(forearm_length / 0.58))
            muscle_forearm_resize = muscle_forearm.copy()
            muscle_forearm_mask_resize = muscle_forearm_mask.copy()

            muscle_forearm_rotate_ratio = (RWrist[0] - RElbow[0]) / (RWrist[1] - RElbow[1])
            muscle_forearm_rotate_degree = math.atan(muscle_forearm_rotate_ratio) * 180 / math.pi + 20
            muscle_forearm_y_offset = 85
            muscle_forearm_rotate_M = cv2.getRotationMatrix2D(
                (int(forearm_size[0] / 2), forearm_size[1] / 2 + muscle_forearm_y_offset),
                muscle_forearm_rotate_degree,
                1)

            # print("forearm", muscle_forearm_rotate_degree)
            muscle_forearm_resize = cv2.warpAffine(muscle_forearm_resize, muscle_forearm_rotate_M, forearm_size)
            muscle_forearm_mask_resize = cv2.warpAffine(muscle_forearm_mask_resize, muscle_forearm_rotate_M,
                                                        forearm_size)

            muscle_forearm_resize = cv2.resize(muscle_forearm_resize, forearm_size)
            muscle_forearm_mask_resize = cv2.resize(muscle_forearm_mask_resize, forearm_size)

            muscle_forearm_position = (int(RElbow[0]), int(RElbow[1] - muscle_forearm_y_offset))
            output_image = cv2.seamlessClone(muscle_forearm_resize, output_image, muscle_forearm_mask_resize,
                                             muscle_forearm_position, cv2.NORMAL_CLONE)

        if person[5][2] > threshold and person[6][2] > threshold and person[7][2] > threshold:
            LShoulder = person[5][0], person[5][1]
            LElbow = person[6][0], person[6][1]
            LWrist = person[7][0], person[7][1]

            cv2.circle(output_image, LShoulder, r, (255, 0, 0), thickness=t)
            cv2.circle(output_image, LElbow, r, (0, 255, 0), thickness=t)
            cv2.circle(output_image, LWrist, r, (0, 0, 255), thickness=t)

    except Exception:
        pass

    # Display the image
    cv2.imshow("output", output_image)
    cv2.imshow("label", labeled_image)
    cv2.waitKey(15)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cap.destroyAllWindows()
