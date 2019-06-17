import cv2
import numpy as np
from mrcnn.config import Config


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


def main():
    import os
    import sys
    import random
    import math
    import time
    from samples.coco.coco import CocoConfig
    from mrcnn import utils
    from mrcnn import model as modellib
    ROOT_DIR = os.getcwd()
    MODEL_DIR = os.path.join(ROOT_DIR, 'logs')
    # COCO_MODEL_PATH = os.path.join(ROOT_DIR, 'mask_rcnn_coco.h5')
    COCO_MODEL_PATH = os.path.join(
        MODEL_DIR, 'shapes20190319T1514/mask_rcnn_shapes_0020.h5')
    if not os.path.exists(COCO_MODEL_PATH):
        utils.download_trained_weights(COCO_MODEL_PATH)

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
    config.display()

    model = modellib.MaskRCNN(
        mode='inference', model_dir=MODEL_DIR, config=config)
    # model.load_weights(COCO_MODEL_PATH, by_name=True)
    model.load_weights(
        COCO_MODEL_PATH,
        by_name=True,
        exclude=["mrcnn_bbox_fc", "mrcnn_class_logits", "mrcnn_mask"])
    class_names = ['BG', 'person']
    # class_names = ['BG', 'person', 'bicycle', 'car', 'motorcycle', 'airplane',
    #                 'bus', 'train', 'truck', 'boat', 'traffic light',
    #                 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird',
    #                 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear',
    #                 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie',
    #                 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    #                 'kite', 'baseball bat', 'baseball glove', 'skateboard',
    #                 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
    #                 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
    #                 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
    #                 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed',
    #                 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
    #                 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster',
    #                 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',
    #                 'teddy bear', 'hair drier', 'toothbrush']

    img = cv2.imread("images/p7.jpg")
    results = model.detect([img], verbose=0)
    r = results[0]
    frame = display_instances(img, r['rois'], r['masks'], r['class_ids'],
                              class_names, r['scores'])

    cv2.imshow('frame', frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    # cap = cv2.VideoCapture(0)
    # cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    # cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    # while True:
    #     ret, frame = cap.read()
    #     results = model.detect([frame], verbose=0)
    #     r = results[0]

    #     frame = display_instances(
    #         frame, r['rois'], r['masks'], r['class_ids'], class_names, r['scores']
    #     )

    #     cv2.imshow('frame', frame)
    #     if cv2.waitKey(1) & 0xFF == ord('q'):
    #         break

    # cap.release()
    # cv2.destroyAllWindows()


if __name__ == '__main__':
    main()