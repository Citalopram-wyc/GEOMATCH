import json
import os
import math
import random
from random import random as rand
import torchvision.transforms as transforms
import torch

import re
from torchvision.transforms.functional import hflip, resize

# from dataset.utils import pre_caption

from torch.utils.data import Dataset

from PIL import Image
from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = None


def pre_caption(caption, max_words):
    # print()
    caption = re.sub(
        r"([,.'!?\"()*#:;~])",
        '',
        caption.lower(),
    ).replace('-', ' ').replace('/', ' ').replace('<person>', 'person')

    caption = re.sub(
        r"\s{2,}",
        ' ',
        caption,
    )
    caption = caption.rstrip('\n')
    caption = caption.strip(' ')

    # truncate caption
    caption_words = caption.split(' ')
    if len(caption_words) > max_words:
        caption = ' '.join(caption_words[:max_words])

    if not len(caption):
        raise ValueError("pre_caption yields invalid text")

    return caption


class text2match_dataset(Dataset):
    def __init__(self, ann_file, transform, image_root, max_words=30, mode='train', config=None):
        self.image_res = 384

        self.ann = []

        for f in ann_file:
            self.ann += json.load(open(f, 'r',encoding='utf-8'))
        self.transform = transform
        self.image_root = image_root
        self.max_words = max_words
        self.mode = mode
        self.img_ids = {}

        n = 0
        for ann in self.ann:
            img_id = ann['image_id']
            if img_id not in self.img_ids.keys():
                self.img_ids[img_id] = n
                n += 1

    def __len__(self):
        return len(self.ann)

    def __getitem__(self, index):
        # print('Note: This part is in the dataset building process')

        ann = self.ann[index]
        caption = pre_caption(ann['caption'], 30)
        # print("Here is the caption",caption)
        image_path = os.path.join(self.image_root, ann['image'])

        image = Image.open(image_path).convert('RGB')
        # print("Here is the original image", image)
        W, H = image.size
        # print(self.annfile)
        # random crop
        target_bboxes = []
        sens = []
        matched_bboxes = []
        for sen in ann["sentences"]:
            if sen is None:
                sen = 'NONE'
            else:
                sen = pre_caption(sen, 30)
            sens.append(sen)

        no_bbox_value = -100
        no_bbox_tensor = [no_bbox_value, no_bbox_value, no_bbox_value, no_bbox_value]

        for box in ann["bboxes"]:
            if box is None:
                target_bboxes.append(no_bbox_tensor)
            else:

                target_bboxes.append(box)
        # max_boxes = 3
        # if len(target_bboxes)<max_boxes:
        #     target_bboxes+=[no_bbox_tensor]*(max_boxes-len(target_bboxes
        #                                                    ))
        # else:
        #     target_bboxes = target_bboxes[:max_boxes]
        if len(ann["matched_boxes"]) == 1:
            for i in range(len(ann["bboxes"])):
                matched_bboxes.append(no_bbox_tensor)
        else:
            for matched_boxe in ann["matched_boxes"]:
                if matched_boxe is None:
                    matched_bboxes.append(no_bbox_tensor)
                else:
                    matched_bboxes.append(matched_boxe)
        # if len(matched_bboxes)<max_boxes:
        #     matched_bboxes+=[no_bbox_tensor]*(max_boxes-len(matched_bboxes
        #                                                    ))
        # else:
        #     matched_bboxes = matched_bboxes[:max_boxes]
        image = resize(image, [self.image_res, self.image_res], interpolation=Image.BICUBIC)
        image = self.transform(image)
        for matched_images in ann["matched_images"]:
            if matched_images is not None:
                matched_images = 'train/' + matched_images
                matched_image_path = os.path.join(self.image_root, matched_images)
                matched_image = Image.open(matched_image_path).convert('RGB')
                matched_image = resize(matched_image, [self.image_res, self.image_res], interpolation=Image.BICUBIC)
                matched_image = self.transform(matched_image)
                break  # 找到第一个有效元素后退出循环
            else:
                matched_image = image

        target_bboxes = torch.tensor(target_bboxes, dtype=torch.float32)
        matched_bboxes = torch.tensor(matched_bboxes, dtype=torch.float32)
        # print(target_bboxes.shape)
        # print(matched_bboxes.shape)
        # print(matched_image.shape)
        # print(image.shape)
        # print(len(sens))
        # print(len(caption))
        # print(self.img_ids[ann['image_id']])
        # # caption = caption[:100]
        #
        # print(f"Caption length: {len(caption.split())}")
        # print(ann['image_id'])
        return image, caption, self.img_ids[ann['image_id']], sens, target_bboxes, matched_image, matched_bboxes


class text2match_val_dataset1(Dataset):
    def __init__(self, ann_file, transform, image_root, max_words=30, mode='train', config=None):
        self.image_res = 384

        self.ann = []

        for f in ann_file:
            self.ann += json.load(open(f, 'r'))
        self.transform = transform
        self.image_root = image_root
        self.max_words = max_words
        self.mode = mode
        self.img_ids = {}

        n = 0
        for ann in self.ann:
            img_id = ann['image_id']
            if img_id not in self.img_ids.keys():
                self.img_ids[img_id] = n
                n += 1

    def __len__(self):
        return len(self.ann)

    def __getitem__(self, index):
        # print('Note: This part is in the dataset building process')

        ann = self.ann[index]
        caption = pre_caption(ann['caption'], 30)
        # print("Here is the caption",caption)
        image_path = os.path.join(self.image_root, ann['image'])

        image = Image.open(image_path).convert('RGB')
        # print("Here is the original image", image)
        W, H = image.size
        # print(self.annfile)
        # random crop
        target_bboxes = []
        sens = []
        matched_bboxes = []
        for sen in ann["sentences"]:
            if sen is None:
                sen = 'NONE'
            else:
                sen = pre_caption(sen, 30)
            sens.append(sen)

        no_bbox_value = -100
        no_bbox_tensor = [no_bbox_value, no_bbox_value, no_bbox_value, no_bbox_value]

        for box in ann["bboxes"]:
            if box is None:
                target_bboxes.append(no_bbox_tensor)
            else:

                target_bboxes.append(box)
        # max_boxes = 3
        # if len(target_bboxes)<max_boxes:
        #     target_bboxes+=[no_bbox_tensor]*(max_boxes-len(target_bboxes
        #                                                    ))
        # else:
        #     target_bboxes = target_bboxes[:max_boxes]
        if len(ann["matched_boxes"]) == 1:
            for i in range(len(ann["bboxes"])):
                matched_bboxes.append(no_bbox_tensor)
        else:
            for matched_boxe in ann["matched_boxes"]:
                if matched_boxe is None:
                    matched_bboxes.append(no_bbox_tensor)
                else:
                    matched_bboxes.append(matched_boxe)
        # if len(matched_bboxes)<max_boxes:
        #     matched_bboxes+=[no_bbox_tensor]*(max_boxes-len(matched_bboxes
        #                                                    ))
        # else:
        #     matched_bboxes = matched_bboxes[:max_boxes]
        image = resize(image, [self.image_res, self.image_res], interpolation=Image.BICUBIC)
        image = self.transform(image)
        for matched_images in ann["matched_images"]:
            if matched_images is not None:
                matched_images = 'train/' + matched_images
                matched_image_path = os.path.join(self.image_root, matched_images)
                matched_image = Image.open(matched_image_path).convert('RGB')
                matched_image = resize(matched_image, [self.image_res, self.image_res], interpolation=Image.BICUBIC)
                matched_image = self.transform(matched_image)
                break  # 找到第一个有效元素后退出循环
            else:
                matched_image = image

        target_bboxes = torch.tensor(target_bboxes, dtype=torch.float32)
        matched_bboxes = torch.tensor(matched_bboxes, dtype=torch.float32)
        # print(target_bboxes.shape)
        # print(matched_bboxes.shape)
        # print(matched_image.shape)
        # print(image.shape)
        # print(len(sens))
        # print(len(caption))
        # print(self.img_ids[ann['image_id']])
        # # caption = caption[:100]
        #
        # print(f"Caption length: {len(caption.split())}")
        # print(ann['image_id'])
        return image, caption, self.img_ids[ann['image_id']], sens, target_bboxes, matched_image, matched_bboxes, ann['image_id']


class text2match_val_dataset(Dataset):
    def __init__(self, ann_file, transform, image_root, max_words=50):
        self.ann = json.load(open(ann_file, 'r', encoding='utf-8'))
        self.transform = transform
        self.image_root = image_root
        self.max_words = max_words

        self.text = []
        self.sens = []
        self.image = []
        self.txt2img = {}
        self.img2txt = {}
        self.sens2img = {}
        self.img2sens = {}
        # self.sens2bboxes = {}
        # self.box2match = {}
        self.img2building = {}

        txt_id = 0
        sens_id = 0
        building_id = 0
        ann_building = 0
        for img_id, ann in enumerate(self.ann):
            ann["building_id"] = ann["image_id"][:4]  # [:4]是文件夹名称也就代表了图片属于那一组或者说是哪一场景
            if ann_building == 0:
                ann_building = ann["building_id"]
            self.image.append(ann['image'])
            self.img2txt[img_id] = []
            self.img2sens[img_id] = []
            self.img2building[img_id] = building_id
            if ann_building != ann["building_id"]:
                ann_building = ann["building_id"]
                building_id += 1

            for i, caption in enumerate(ann['caption']):
                # print(caption)
                self.text.append(pre_caption(caption, self.max_words))
                self.img2txt[img_id].append(txt_id)
                self.txt2img[txt_id] = img_id
                txt_id += 1

            for i, sens in enumerate(ann['sentences']):
                # print(sens)
                if sens is None:
                    sens = 'NONE'
                self.sens.append(pre_caption(sens, self.max_words))
                self.img2sens[img_id].append(sens_id)
                self.sens2img[sens_id] = img_id
                sens_id += 1

    def __len__(self):
        return len(self.image)

    def __getitem__(self, index):

        image_path = os.path.join(self.image_root, self.ann[index]['image'])
        image = Image.open(image_path).convert('RGB')
        image = self.transform(image)
        # 获取当前图片的名称
        image_name = self.image[index]  # 直接从 image 列表中获取图片的文件名
        print(image_name)
        # print(self.text, self.sens, self.txt2img,self.img2txt, self.sens2img, self.img2sens)
        return image, index



class match_val_dataset(Dataset):
    def __init__(self, transform, image_root, max_words=256):

        #1127
        #1164
        caption = "this is an aerial view of a cluster of high-rise buildings in an urban environment, with a tall white building in the center, a main road behind it, and green space and residential buildings around it."
        print(len(caption))
        sentence = "A low-rise, flat-roofed outbuildings attached to the main building."
        org_image_path = r"X:\_paper4\GeoText-1652-main\GeoText-1652-main\GeoText1652_Dataset\images\train\0846\0846.jpg"
        self.org_image_path = org_image_path
        self.transform = transform
        self.image_root = image_root
        self.max_words = max_words


        self.caption = pre_caption(caption, self.max_words)  # 单个caption句子
        self.sentence = pre_caption(sentence, self.max_words)  # 单个sentence句子

        org_image = Image.open(self.org_image_path).convert('RGB')
        self.org_image = self.transform(org_image)
        # 获取所有图片文件名
        # self.image = [f for f in os.listdir(image_root) if f.endswith(('.jpg', '.JPG','.jpeg', '.png'))]  # 根据需要增加图片格式

        # List to store image file paths
        self.image = []

        # Walk through the root directory and all subdirectories
        for root, dirs, files in os.walk(image_root):
            for file in files:
                if file.endswith(('.jpg', '.JPG', '.jpeg', '.png')):  # Filter for image files
                    self.image.append(os.path.join(root, file))  # Append the full path of each image

    def __len__(self):
        return len(self.image)

    def __getitem__(self, index):

        image_name = self.image[index]
        image_path = os.path.join(self.image_root, image_name)
        image = Image.open(image_path).convert('RGB')
        image = self.transform(image)
        # print(self.text, self.sens, self.txt2img,self.img2txt, self.sens2img, self.img2sens)
        return image, self.caption, self.sentence, index, image_name, self.org_image


# image_root = r"X:\_paper4\GeoText-1652-main\GeoText-1652-main\GeoText1652_Dataset\images\test\wx"
# normalize = transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
# test_transform = transforms.Compose([
#         transforms.Resize((384, 384), interpolation=Image.BICUBIC),
#         transforms.ToTensor(),
#         normalize,
#     ])
# # json_file = r"X:\_paper4\GeoText-1652-main\GeoText-1652-main\GeoText1652_Dataset\test_mini.json"
# json_file = r"X:\_paper4\GeoText-1652-main\GeoText-1652-main\GeoText1652_Dataset\train_match_data0213.json"
# test_dataset = match_val_dataset(test_transform, image_root, json_file, max_words=200)