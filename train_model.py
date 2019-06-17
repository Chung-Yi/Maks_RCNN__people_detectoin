import os
import random
import sys
import numpy as np
import cv2
import glob
import matplotlib
import matplotlib.pyplot as plt
from mrcnn import utils
from mrcnn import model as modellib
from mrcnn.config import Config
import yaml
from mrcnn.model import log
from PIL import Image

ROOT_DIR = os.getcwd()
MODEL_DIR = os.path.join(ROOT_DIR, 'logs')
iter_num = 0
COCO_MODEL_PATH = os.path.join(ROOT_DIR, 'mask_rcnn_coco.h5')

# download coco_model
if not os.path.exists(COCO_MODEL_PATH):
    utils.download_trained_weights(COCO_MODEL_PATH)


class ShapesConfig(Config):
    NAME = 'shapes'

    GPU_COUNT = 1
    IMAGES_PER_GPU = 2

    NUM_CLASSES = 1 + 1  # backgriund + 1 class

    IMAGE_MIN_DIM = 720
    IMAGE_MAX_DIM = 1280

    RPN_ANCHOR_SCALES = (8 * 6, 16 * 6, 32 * 6, 64 * 6, 128 * 6)

    TRAIN_ROIS_PER_IMAGE = 100

    STEPS_PER_EPOCH = 100

    VALIDATION_STEPS = 50


class DrugDataset(utils.Dataset):
    def get_obj_index(self, image):
        n = np.max(image)
        return n

    def from_yaml_get_class(self, image_id):
        info = self.image_info[image_id]
        with open(info['yaml_path']) as f:
            temp = yaml.load(f.read())
            labels = temp['label_names']
            del labels[0]
        return labels

    def draw_mask(self, num_obj, mask, image, image_id):
        #print("draw_mask-->",image_id)
        #print("self.image_info",self.image_info)
        info = self.image_info[image_id]
        #print("info-->",info)
        #print("info[width]----->",info['width'],"-info[height]--->",info['height'])
        for index in range(num_obj):
            for i in range(info['width']):
                for j in range(info['height']):
                    #print("image_id-->",image_id,"-i--->",i,"-j--->",j)
                    #print("info[width]----->",info['width'],"-info[height]--->",info['height'])
                    at_pixel = image.getpixel((i, j))
                    if at_pixel == index + 1:
                        mask[j, i, index] = 1
        return mask

    def load_shapes(self, count, image_folder, mask_folder, image_list,
                    dataset_root_path):
        self.add_class('shapes', 1, 'person')
        for i in range(count):
            # get image info
            image_name = image_list[i].split('.')[0]
            mask_path = os.path.join(mask_folder, image_list[i])
            label_img = os.path.join(dataset_root_path, 'labelme_json',
                                     image_name + '_json', 'img.png')
            yaml_path = os.path.join(dataset_root_path, 'labelme_json',
                                     image_name + '_json', 'info.yaml')
            img = cv2.imread(label_img)
            # filestr = image_list[i].split(".")[0]
            # mask_path = mask_folder + "/" + filestr + ".png"
            # yaml_path = dataset_root_path + "labelme_json/" + filestr + "_json/info.yaml"
            # img = cv2.imread(dataset_root_path + "labelme_json/" + filestr + "_json/img.png")
            self.add_image(
                'shapes',
                image_id=i,
                path=os.path.join(image_folder, image_list[i]),
                width=img.shape[1],
                height=img.shape[0],
                mask_path=mask_path,
                yaml_path=yaml_path)

    def load_mask(self, image_id):
        """Generate instance masks for shapes of the given image ID.
        """
        global iter_num
        print("image_id", image_id)
        info = self.image_info[image_id]
        count = 1  # number of object
        img = Image.open(info['mask_path'])
        num_obj = self.get_obj_index(img)
        mask = np.zeros([info['height'], info['width'], num_obj],
                        dtype=np.uint8)
        mask = self.draw_mask(num_obj, mask, img, image_id)
        occlusion = np.logical_not(mask[:, :, -1]).astype(np.uint8)
        for i in range(count - 2, -1, -1):
            mask[:, :, i] = mask[:, :, i] * occlusion

            occlusion = np.logical_and(occlusion,
                                       np.logical_not(mask[:, :, i]))
        labels = []
        labels = self.from_yaml_get_class(image_id)
        labels_form = []
        for i in range(len(labels)):
            if labels[i].find("person") != -1:
                # print "box"
                labels_form.append("person")
            # elif labels[i].find("triangle")!=-1:
            #     #print "column"
            #     labels_form.append("triangle")
            # elif labels[i].find("white")!=-1:
            #     #print "package"
            #     labels_form.append("white")
        class_ids = np.array([self.class_names.index(s) for s in labels_form])
        return mask, class_ids.astype(np.int32)

    def get_ax(rows=1, cols=1, size=8):
        """Return a Matplotlib Axes array to be used in
        all visualizations in the notebook. Provide a
        central point to control graph sizes.

        Change the default size attribute to control the size
        of rendered images
        """
        _, ax = plt.subplots(rows, cols, figsize=(size * cols, size * rows))
        return ax


config = ShapesConfig()
config.display()

dataset_root_path = 'dataset'
image_folder = os.path.join(dataset_root_path, 'images')
mask_folder = os.path.join(dataset_root_path, 'mask')
# dataset_root_path="dataset/"
# image_folder = dataset_root_path + "images"
# mask_folder = dataset_root_path + "mask"
image_list = os.listdir(image_folder)
count = (len(image_list))

# train data
train_data = DrugDataset()
train_data.load_shapes(count, image_folder, mask_folder, image_list,
                       dataset_root_path)
train_data.prepare()

# validate data
val_data = DrugDataset()
val_data.load_shapes(8, image_folder, mask_folder, image_list,
                     dataset_root_path)
val_data.prepare()

# create model
model = modellib.MaskRCNN(mode='training', config=config, model_dir=MODEL_DIR)

init_with = 'coco'
if init_with == 'imagenet':
    model.load_weights(model.get_imagenet_weights(), by_name=True)
elif init_with == 'coco':
    model.load_weights(
        COCO_MODEL_PATH,
        by_name=True,
        exclude=[
            "mrcnn_class_logits", "mrcnn_bbox_fc", "mrcnn_bbox", "mrcnn_mask"
        ])
elif init_with == 'last':
    model.load_weights(model.find_last()[1], by_name=True)

model.train(
    train_data,
    val_data,
    learning_rate=config.LEARNING_RATE,
    epochs=20,
    layers='heads')
model.train(
    train_data,
    val_data,
    learning_rate=config.LEARNING_RATE / 10,
    epochs=40,
    layers='all')
