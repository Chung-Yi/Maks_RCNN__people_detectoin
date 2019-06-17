import os
import cv2
import numpy as np
import mrcnn.config
import mrcnn.utils
from mrcnn.model import MaskRCNN
from pathlib import Path
from argparse import ArgumentParser
import matplotlib.pyplot as plt


class MaskRCNNConfig(mrcnn.config.Config):
    NAME = "coco_pretrained_model_config"
    IMAGES_PER_GPU = 1
    GPU_COUNT = 1
    NUM_CLASSES = 1 + 1  # COCO dataset has 80 classes + one background class
    DETECTION_MIN_CONFIDENCE = 0.8


# def _parse_arguments():
#     parser = ArgumentParser()
#     parser.add_argument(
#         "--did",
#         default="./pedestrian.mp4",
#         help="device ID, ex: 0AA1EA9A5A04B78D4581DD6D17742627.")
#     parser.add_argument(
#         "--mode",
#         default="image",
#     )
#     return parser.parse_args()


def apply_mask(image, mask, color, alpha=0.5):
    """ apply mask to image """
    for c in range(3):
        image[:, :, c] = np.where(
            mask == 1, image[:, :, c] * (1 - alpha) + alpha * color[c] * 255,
            image[:, :, c])
    return image


def random_colors(N):
    np.random.seed(1)
    colors = [tuple(255 * np.random.rand(3)) for _ in range(N)]
    return colors


# def get_people_boxes(image, boxes, class_ids, masks):
#     n_instances = boxes.shape[0]
#     if not n_instances:
#         print('No instances to display')
#     else:
#         assert boxes.shape[0] == masks.shape[-1] == class_ids.shape[0]

#     colors = random_colors(n_instances)

#     for i, color in enumerate(colors):
#         if not np.any(boxes[i]):
#             continue
#         if class_ids[i] == 1:
#             y1, x1, y2, x2 = boxes[i]
#             mask = masks[:, :, i]
#             image = apply_mask(image, mask, color)
#             image = cv2.rectangle(image, (x1, y1), (x2, y2), color, 1)

#     return image


def get_people_boxes(frame, boxes, class_ids):
    people_boxes = []

    for i, box in enumerate(boxes):
        # If the detected object isn't a car / truck, skip it
        if class_ids[i] in [1]:
            y1, x1, y2, x2 = boxes[i]
            image = cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 1)

    return image


def main():
    # args = _parse_arguments()
    ROOT_DIR = Path(".")
    MODEL_DIR = os.path.join(ROOT_DIR, "logs")
    COCO_MODEL_PATH = os.path.join(MODEL_DIR, "shapes20190319T1514",
                                   "mask_rcnn_coco.h5")

    # VIDEO_SOURCE = args.did

    if not os.path.exists(COCO_MODEL_PATH):
        mrcnn.utils.download_trained_weights(COCO_MODEL_PATH)

    IMAGE_DIR = os.path.join(ROOT_DIR, "images")

    # Load pretrained model
    model = MaskRCNN(
        mode="inference", model_dir=MODEL_DIR, config=MaskRCNNConfig())
    model.load_weights(
        COCO_MODEL_PATH,
        by_name=True,
        exclude=["mrcnn_bbox_fc", "mrcnn_class_logits", "mrcnn_mask"])

    img = cv2.imread("images/p7.jpg")
    image = img[:, :, ::-1]
    results = model.detect([image], verbose=0)
    r = results[0]
    frame = get_people_boxes(img, r['rois'], r['class_ids'])
    cv2.imshow('frame', frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # if args.mode == "image":
    #     img = cv2.imread("images/p5.jpg")
    #     results = model.detect([img], verbose=0)
    #     r = results[0]
    #     people_boxes = get_people_boxes(r['rois'], r['class_ids'])

    #     for box in people_boxes:
    #         print('person: ', box)
    #         y1, x1, y2, x2 = box
    #         cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 1)
    #         cv2.imshow('img', img)
    #     cv2.waitKey(0)
    #     cv2.destroyAllWindows()

    # elif args.mode == "video":
    #     cap = cv2.VideoCapture("./walk.mp4")
    #     while cap.isOpened():
    #         success, frame = cap.read()
    #         cv2.imshow('frame', frame)
    #         if not success:
    #             break
    #         frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    #         results = model.detect([frame], verbose=0)
    #         print(results)
    #         r = results[0]
    #         people_boxes = get_people_boxes(r['rois'], r['class_ids'])

    #         for box in people_boxes:
    #             print('person: ', box)
    #             y1, x1, y2, x2 = box
    #             cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 1)
    #             cv2.imshow('frame', frame)

    #         if cv2.waitKey(1) & 0xFF == ord('q'):
    #             break

    #     video_capture.release()
    #     cv2.destroyAllWindows()

    # cap = cv2.VideoCapture("./pedestrian.mp4")
    # while cap.isOpened():
    #     success, frame = cap.read()
    #     if not success:
    #         break

    #     frame = frame[:, :, ::-1]

    # results = model.detect([frame], verbose=0)
    # r = results[0]
    # frame = get_people_boxes(frame, r['rois'], r['class_ids'], r['masks'])
    # cv2.imshow('frame', frame)

    # cap.release()
    # cv2.destroyAllWindows()


if __name__ == "__main__":
    main()