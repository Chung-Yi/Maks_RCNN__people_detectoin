import os
import sys
import random
import numpy as np
import skimage.io
import cv2
import time
from datetime import datetime
from mrcnn.config import Config
from mrcnn import utils
import mrcnn.model as modellib
from mrcnn import visualize
from samples.coco.coco import CocoConfig
# from samples.coco import coco

ROOT_DIR = os.getcwd()
sys.path.append(ROOT_DIR)
sys.path.append(os.path.join(ROOT_DIR, "samples/coco/"))
from samples.coco import coco

MODEL_DIR = os.path.join(ROOT_DIR, 'logs')
COCO_MODEL_PATH = os.path.join(MODEL_DIR, 'shapes20190319T1514',
                               'mask_rcnn_shapes_0020.h5')

# download coco_model
if not os.path.exists(COCO_MODEL_PATH):
    utils.download_trained_weights(COCO_MODEL_PATH)

IMAGE_DIR = os.path.join(ROOT_DIR, 'test_images')


def random_colors(N):
    np.random.seed(1)
    colors = [tuple(255 * np.random.rand(3)) for _ in range(N)]
    return colors


def apply_mask(image, mask, color, alpha=0.5):
    """ apply mask to image """
    for c in range(3):
        image[:, :, c] = np.where(
            mask == 1, image[:, :, c] * (1 - alpha) + alpha * color[c] * 255,
            image[:, :, c])
    return image


def get_people_boxes(boxes, class_ids):
    people_boxes = []
    print(class_ids[0])
    for i, box in enumerate(boxes):
        if class_ids[i] in [1]:
            people_boxes.append(box)

    return np.array(people_boxes)


def display_instances(image, boxes, masks, ids, names, scores):
    n_instances = boxes.shape[0]
    if not n_instances:
        print('No instances to display')
    else:
        assert boxes.shape[0] == masks.shape[-1] == ids.shape[0]

    colors = random_colors(n_instances)
    height, width = image.shape[:2]

    for i, color in enumerate(colors):
        if not np.any(boxes[i]):
            continue

        y1, x1, y2, x2 = boxes[i]
        mask = masks[:, :, i]
        image = apply_mask(image, mask, color)
        image = cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)

        label = names[ids[i]]
        score = scores[i] if scores is not None else None
        caption = '{} {:.2f}'.format(label, score) if score else label
        image = cv2.putText(image, caption, (x1, y1), cv2.FONT_HERSHEY_COMPLEX,
                            0.7, color, 2)

    return image


class ShapesConfig(Config):
    NAME = 'shapes'

    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

    NUM_CLASSES = 1 + 1  # backgriund + 1 class

    IMAGE_MIN_DIM = 720
    IMAGE_MAX_DIM = 1280

    RPN_ANCHOR_SCALES = (8 * 6, 16 * 6, 32 * 6, 64 * 6, 128 * 6)

    TRAIN_ROIS_PER_IMAGE = 100

    STEPS_PER_EPOCH = 100

    VALIDATION_STEPS = 50


class InferenceConfig(ShapesConfig):
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1


config = InferenceConfig()
model = modellib.MaskRCNN(mode='inference', model_dir=MODEL_DIR, config=config)
model.load_weights(
    COCO_MODEL_PATH,
    by_name=True,
    exclude=["mrcnn_bbox_fc", "mrcnn_class_logits", "mrcnn_mask"])

class_names = ['BG', 'person']
file_names = next(os.walk(IMAGE_DIR))[2]

# test with image
# image = skimage.io.imread(os.path.join(IMAGE_DIR, random.choice(file_names)))

# a = datetime.now()
# results = model.detect([image], verbose=1)
# b = datetime.now()
# # print((b-a).second)
# r = results[0]
# visualize.display_instances(image, r['rois'], r['masks'], r['class_ids'],
#                             class_names, r['scores'])

img = cv2.imread("images/p5.jpg")
results = model.detect([img], verbose=0)
r = results[0]
people_boxes = get_people_boxes(r['rois'], r['class_ids'])

for box in people_boxes:
    print('person: ', box)
    y1, x1, y2, x2 = box
    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 1)
    cv2.imshow('img', img)
cv2.waitKey(0)
cv2.destroyAllWindows()

# test with webcam
# cap = cv2.VideoCapture(0)
# cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
# cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

# while True:
#     ret, frame = cap.read()
#     results = model.detect([frame], verbose=0)
#     r = results[0]

#     frame = display_instances(frame, r['rois'], r['masks'], r['class_ids'],
#                               class_names, r['scores'])

#     cv2.imshow('frame', frame)
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# cap.release()
# cv2.destroyAllWindows()