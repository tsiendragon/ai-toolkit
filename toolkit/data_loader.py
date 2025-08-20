import copy
import json
import os
import random
import traceback
from functools import lru_cache
from typing import List, TYPE_CHECKING

import cv2
import numpy as np
import torch
from PIL import Image
from PIL.ImageOps import exif_transpose
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from tqdm import tqdm
import albumentations as A

from toolkit import image_utils
from toolkit.buckets import get_bucket_for_image_size, BucketResolution
from toolkit.config_modules import DatasetConfig, preprocess_dataset_raw_config
from toolkit.dataloader_mixins import CaptionMixin, BucketsMixin, LatentCachingMixin, Augments, CLIPCachingMixin, ControlCachingMixin, TextEmbeddingCachingMixin
from toolkit.data_transfer_object.data_loader import FileItemDTO, DataLoaderBatchDTO
from toolkit.print import print_acc
from toolkit.accelerator import get_accelerator

import platform

def is_native_windows():
    return platform.system() == "Windows" and platform.release() != "2"

if TYPE_CHECKING:
    from toolkit.stable_diffusion_model import StableDiffusion


image_extensions = ['.jpg', '.jpeg', '.png', '.webp']
video_extensions = ['.mp4', '.avi', '.mov', '.webm', '.mkv', '.wmv', '.m4v', '.flv']


class RescaleTransform:
    """Transform to rescale images to the range [-1, 1]."""

    def __call__(self, image):
        return image * 2 - 1


class NormalizeSDXLTransform:
    """
    Transforms the range from 0 to 1 to SDXL mean and std per channel based on avgs over thousands of images

    Mean: tensor([ 0.0002, -0.1034, -0.1879])
    Standard Deviation: tensor([0.5436, 0.5116, 0.5033])
    """

    def __call__(self, image):
        return transforms.Normalize(
            mean=[0.0002, -0.1034, -0.1879],
            std=[0.5436, 0.5116, 0.5033],
        )(image)


class NormalizeSD15Transform:
    """
    Transforms the range from 0 to 1 to SDXL mean and std per channel based on avgs over thousands of images

    Mean: tensor([-0.1600, -0.2450, -0.3227])
    Standard Deviation: tensor([0.5319, 0.4997, 0.5139])

    """

    def __call__(self, image):
        return transforms.Normalize(
            mean=[-0.1600, -0.2450, -0.3227],
            std=[0.5319, 0.4997, 0.5139],
        )(image)



class ImageDataset(Dataset, CaptionMixin):
    def __init__(self, config):
        self.config = config
        self.name = self.get_config('name', 'dataset')
        self.path = self.get_config('path', required=True)
        self.scale = self.get_config('scale', 1)
        self.random_scale = self.get_config('random_scale', False)
        self.include_prompt = self.get_config('include_prompt', False)
        self.default_prompt = self.get_config('default_prompt', '')
        if self.include_prompt:
            self.caption_type = self.get_config('caption_ext', 'txt')
        else:
            self.caption_type = None
        # we always random crop if random scale is enabled
        self.random_crop = self.random_scale if self.random_scale else self.get_config('random_crop', False)

        self.resolution = self.get_config('resolution', 256)
        self.file_list = [os.path.join(self.path, file) for file in os.listdir(self.path) if
                          file.lower().endswith(('.jpg', '.jpeg', '.png', '.webp'))]

        # this might take a while
        print_acc(f"  -  Preprocessing image dimensions")
        new_file_list = []
        bad_count = 0
        for file in tqdm(self.file_list):
            try:
                w, h = image_utils.get_image_size(file)
            except image_utils.UnknownImageFormat:
                img = exif_transpose(Image.open(file))
                w, h = img.size
            # img = Image.open(file)
            if int(min([w, h]) * self.scale) >= self.resolution:
                new_file_list.append(file)
            else:
                bad_count += 1

        self.file_list = new_file_list

        print_acc(f"  -  Found {len(self.file_list)} images")
        print_acc(f"  -  Found {bad_count} images that are too small")
        assert len(self.file_list) > 0, f"no images found in {self.path}"

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            RescaleTransform(),
        ])

    def get_config(self, key, default=None, required=False):
        if key in self.config:
            value = self.config[key]
            return value
        elif required:
            raise ValueError(f'config file error. Missing "config.dataset.{key}" key')
        else:
            return default

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, index):
        img_path = self.file_list[index]
        try:
            img = exif_transpose(Image.open(img_path)).convert('RGB')
        except Exception as e:
            print_acc(f"Error opening image: {img_path}")
            print_acc(e)
            # make a noise image if we can't open it
            img = Image.fromarray(np.random.randint(0, 255, (1024, 1024, 3), dtype=np.uint8))

        # Downscale the source image first
        img = img.resize((int(img.size[0] * self.scale), int(img.size[1] * self.scale)), Image.BICUBIC)
        min_img_size = min(img.size)

        if self.random_crop:
            if self.random_scale and min_img_size > self.resolution:
                if min_img_size < self.resolution:
                    print_acc(
                        f"Unexpected values: min_img_size={min_img_size}, self.resolution={self.resolution}, image file={img_path}")
                    scale_size = self.resolution
                else:
                    scale_size = random.randint(self.resolution, int(min_img_size))
                scaler = scale_size / min_img_size
                scale_width = int((img.width + 5) * scaler)
                scale_height = int((img.height + 5) * scaler)
                img = img.resize((scale_width, scale_height), Image.BICUBIC)
            img = transforms.RandomCrop(self.resolution)(img)
        else:
            img = transforms.CenterCrop(min_img_size)(img)
            img = img.resize((self.resolution, self.resolution), Image.BICUBIC)

        img = self.transform(img)

        if self.include_prompt:
            prompt = self.get_caption_item(index)
            return img, prompt
        else:
            return img





class AugmentedImageDataset(ImageDataset):
    def __init__(self, config):
        super().__init__(config)
        self.augmentations = self.get_config('augmentations', [])
        self.augmentations = [Augments(**aug) for aug in self.augmentations]

        augmentation_list = []
        for aug in self.augmentations:
            # make sure method name is valid
            assert hasattr(A, aug.method_name), f"invalid augmentation method: {aug.method_name}"
            # get the method
            method = getattr(A, aug.method_name)
            # add the method to the list
            augmentation_list.append(method(**aug.params))

        self.aug_transform = A.Compose(augmentation_list)
        self.original_transform = self.transform
        # replace transform so we get raw pil image
        self.transform = transforms.Compose([])

    def __getitem__(self, index):
        # get the original image
        # image is a PIL image, convert to bgr
        pil_image = super().__getitem__(index)
        open_cv_image = np.array(pil_image)
        # Convert RGB to BGR
        open_cv_image = open_cv_image[:, :, ::-1].copy()

        # apply augmentations
        augmented = self.aug_transform(image=open_cv_image)["image"]

        # convert back to RGB tensor
        augmented = cv2.cvtColor(augmented, cv2.COLOR_BGR2RGB)

        # convert to PIL image
        augmented = Image.fromarray(augmented)

        # return both # return image as 0 - 1 tensor
        return transforms.ToTensor()(pil_image), transforms.ToTensor()(augmented)


class PairedImageDataset(Dataset):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.size = self.get_config('size', 512)
        self.path = self.get_config('path', None)
        self.pos_folder = self.get_config('pos_folder', None)
        self.neg_folder = self.get_config('neg_folder', None)

        self.default_prompt = self.get_config('default_prompt', '')
        self.network_weight = self.get_config('network_weight', 1.0)
        self.pos_weight = self.get_config('pos_weight', self.network_weight)
        self.neg_weight = self.get_config('neg_weight', self.network_weight)

        supported_exts = ('.jpg', '.jpeg', '.png', '.webp', '.JPEG', '.JPG', '.PNG', '.WEBP')

        if self.pos_folder is not None and self.neg_folder is not None:
            # find matching files
            self.pos_file_list = [os.path.join(self.pos_folder, file) for file in os.listdir(self.pos_folder) if
                                  file.lower().endswith(supported_exts)]
            self.neg_file_list = [os.path.join(self.neg_folder, file) for file in os.listdir(self.neg_folder) if
                                  file.lower().endswith(supported_exts)]

            matched_files = []
            for pos_file in self.pos_file_list:
                pos_file_no_ext = os.path.splitext(pos_file)[0]
                for neg_file in self.neg_file_list:
                    neg_file_no_ext = os.path.splitext(neg_file)[0]
                    if os.path.basename(pos_file_no_ext) == os.path.basename(neg_file_no_ext):
                        matched_files.append((neg_file, pos_file))
                        break

            # remove duplicates
            matched_files = [t for t in (set(tuple(i) for i in matched_files))]

            self.file_list = matched_files
            print_acc(f"  -  Found {len(self.file_list)} matching pairs")
        else:
            self.file_list = [os.path.join(self.path, file) for file in os.listdir(self.path) if
                              file.lower().endswith(supported_exts)]
            print_acc(f"  -  Found {len(self.file_list)} images")

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            RescaleTransform(),
        ])

    def get_all_prompts(self):
        prompts = []
        for index in range(len(self.file_list)):
            prompts.append(self.get_prompt_item(index))

        # remove duplicates
        prompts = list(set(prompts))
        return prompts

    def __len__(self):
        return len(self.file_list)

    def get_config(self, key, default=None, required=False):
        if key in self.config:
            value = self.config[key]
            return value
        elif required:
            raise ValueError(f'config file error. Missing "config.dataset.{key}" key')
        else:
            return default

    def get_prompt_item(self, index):
        img_path_or_tuple = self.file_list[index]
        if isinstance(img_path_or_tuple, tuple):
            # check if either has a prompt file
            path_no_ext = os.path.splitext(img_path_or_tuple[0])[0]
            prompt_path = path_no_ext + '.txt'
            if not os.path.exists(prompt_path):
                path_no_ext = os.path.splitext(img_path_or_tuple[1])[0]
                prompt_path = path_no_ext + '.txt'
        else:
            img_path = img_path_or_tuple
            # see if prompt file exists
            path_no_ext = os.path.splitext(img_path)[0]
            prompt_path = path_no_ext + '.txt'

        if os.path.exists(prompt_path):
            with open(prompt_path, 'r', encoding='utf-8') as f:
                prompt = f.read()
                # remove any newlines
                prompt = prompt.replace('\n', ', ')
                # remove new lines for all operating systems
                prompt = prompt.replace('\r', ', ')
                prompt_split = prompt.split(',')
                # remove empty strings
                prompt_split = [p.strip() for p in prompt_split if p.strip()]
                # join back together
                prompt = ', '.join(prompt_split)
        else:
            prompt = self.default_prompt
        return prompt

    def __getitem__(self, index):
        img_path_or_tuple = self.file_list[index]
        if isinstance(img_path_or_tuple, tuple):
            # load both images
            img_path = img_path_or_tuple[0]
            img1 = exif_transpose(Image.open(img_path)).convert('RGB')
            img_path = img_path_or_tuple[1]
            img2 = exif_transpose(Image.open(img_path)).convert('RGB')

            # always use # 2 (pos)
            bucket_resolution = get_bucket_for_image_size(
                width=img2.width,
                height=img2.height,
                resolution=self.size,
                # divisibility=self.
            )

            # images will be same base dimension, but may be trimmed. We need to shrink and then central crop
            if bucket_resolution['width'] > bucket_resolution['height']:
                img1_scale_to_height = bucket_resolution["height"]
                img1_scale_to_width = int(img1.width * (bucket_resolution["height"] / img1.height))
                img2_scale_to_height = bucket_resolution["height"]
                img2_scale_to_width = int(img2.width * (bucket_resolution["height"] / img2.height))
            else:
                img1_scale_to_width = bucket_resolution["width"]
                img1_scale_to_height = int(img1.height * (bucket_resolution["width"] / img1.width))
                img2_scale_to_width = bucket_resolution["width"]
                img2_scale_to_height = int(img2.height * (bucket_resolution["width"] / img2.width))

            img1_crop_height = bucket_resolution["height"]
            img1_crop_width = bucket_resolution["width"]
            img2_crop_height = bucket_resolution["height"]
            img2_crop_width = bucket_resolution["width"]

            # scale then center crop images
            img1 = img1.resize((img1_scale_to_width, img1_scale_to_height), Image.BICUBIC)
            img1 = transforms.CenterCrop((img1_crop_height, img1_crop_width))(img1)
            img2 = img2.resize((img2_scale_to_width, img2_scale_to_height), Image.BICUBIC)
            img2 = transforms.CenterCrop((img2_crop_height, img2_crop_width))(img2)

            # combine them side by side
            img = Image.new('RGB', (img1.width + img2.width, max(img1.height, img2.height)))
            img.paste(img1, (0, 0))
            img.paste(img2, (img1.width, 0))
        else:
            img_path = img_path_or_tuple
            img = exif_transpose(Image.open(img_path)).convert('RGB')
            height = self.size
            # determine width to keep aspect ratio
            width = int(img.size[0] * height / img.size[1])

            # Downscale the source image first
            img = img.resize((width, height), Image.BICUBIC)

        prompt = self.get_prompt_item(index)
        img = self.transform(img)

        return img, prompt, (self.neg_weight, self.pos_weight)


class AiToolkitDataset(LatentCachingMixin, ControlCachingMixin, CLIPCachingMixin, TextEmbeddingCachingMixin, BucketsMixin, CaptionMixin, Dataset):

    def __init__(
            self,
            dataset_config: 'DatasetConfig',
            batch_size=1,
            sd: 'StableDiffusion' = None,
    ):
        import logging
        logger = logging.getLogger(__name__)
        logger.info(f"🔍 [DATASET_INIT] AiToolkitDataset 初始化开始")

        self.dataset_config = dataset_config
        # update bucket divisibility
        self.dataset_config.bucket_tolerance = sd.get_bucket_divisibility()
        self.is_video = dataset_config.num_frames > 1
        super().__init__()
        folder_path = dataset_config.folder_path
        self.dataset_path = dataset_config.dataset_path
        if self.dataset_path is None:
            self.dataset_path = folder_path

        self.is_caching_latents = dataset_config.cache_latents or dataset_config.cache_latents_to_disk
        self.is_caching_latents_to_memory = dataset_config.cache_latents
        self.is_caching_latents_to_disk = dataset_config.cache_latents_to_disk
        self.is_caching_control_latents_to_disk = dataset_config.cache_control_latents_to_disk
        self.is_caching_clip_vision_to_disk = dataset_config.cache_clip_vision_to_disk
        self.is_generating_controls = len(dataset_config.controls) > 0
        self.epoch_num = 0

        self.sd = sd

        if self.sd is None and self.is_caching_latents:
            raise ValueError(f"sd is required for caching latents")

        self.caption_type = dataset_config.caption_ext
        self.default_caption = dataset_config.default_caption
        self.random_scale = dataset_config.random_scale
        self.scale = dataset_config.scale
        self.batch_size = batch_size
        # we always random crop if random scale is enabled
        self.random_crop = self.random_scale if self.random_scale else dataset_config.random_crop
        self.resolution = dataset_config.resolution
        self.customized_shape = dataset_config.customized_shape  # [width, height] for fixed shape training
        self.caption_dict = None
        self.file_list: List['FileItemDTO'] = []

        # check if dataset_path is a folder or json
        if os.path.isdir(self.dataset_path):
            extensions = image_extensions
            if self.is_video:
                # only look for videos
                extensions = video_extensions
            file_list = [os.path.join(root, file) for root, _, files in os.walk(self.dataset_path) for file in files if file.lower().endswith(tuple(extensions))]
        else:
            # assume json
            with open(self.dataset_path, 'r') as f:
                self.caption_dict = json.load(f)
                # keys are file paths
                file_list = list(self.caption_dict.keys())

        # remove items in the _controls_ folder
        file_list = [x for x in file_list if not os.path.basename(os.path.dirname(x)) == "_controls"]

        if self.dataset_config.num_repeats > 1:
            # repeat the list
            file_list = file_list * self.dataset_config.num_repeats

        if self.dataset_config.standardize_images:
            if self.sd.is_xl or self.sd.is_vega or self.sd.is_ssd:
                NormalizeMethod = NormalizeSDXLTransform
            else:
                NormalizeMethod = NormalizeSD15Transform

            self.transform = transforms.Compose([
                transforms.ToTensor(),
                RescaleTransform(),
                NormalizeMethod(),
            ])
        else:
            self.transform = transforms.Compose([
                transforms.ToTensor(),
                RescaleTransform(),
            ])

        # this might take a while
        print_acc(f"Dataset: {self.dataset_path}")
        if self.is_video:
            print_acc(f"  -  Preprocessing video dimensions")
        else:
            print_acc(f"  -  Preprocessing image dimensions")
        dataset_folder = self.dataset_path
        if not os.path.isdir(self.dataset_path):
            dataset_folder = os.path.dirname(dataset_folder)

        dataset_size_file = os.path.join(dataset_folder, '.aitk_size.json')
        dataloader_version = "0.1.2"
        if os.path.exists(dataset_size_file):
            try:
                with open(dataset_size_file, 'r') as f:
                    self.size_database = json.load(f)

                if "__version__" not in self.size_database or self.size_database["__version__"] != dataloader_version:
                    print_acc("Upgrading size database to new version")
                    # old version, delete and recreate
                    self.size_database = {}
            except Exception as e:
                print_acc(f"Error loading size database: {dataset_size_file}")
                print_acc(e)
                self.size_database = {}
        else:
            self.size_database = {}

        self.size_database["__version__"] = dataloader_version

        bad_count = 0
        for file in tqdm(file_list):
            try:
                file_item = FileItemDTO(
                    sd=self.sd,
                    path=file,
                    dataset_config=dataset_config,
                    dataloader_transforms=self.transform,
                    size_database=self.size_database,
                    dataset_root=dataset_folder,
                )
                self.file_list.append(file_item)
            except Exception as e:
                print_acc(traceback.format_exc())
                if self.is_video:
                    print_acc(f"Error processing video: {file}")
                else:
                    print_acc(f"Error processing image: {file}")
                print_acc(e)
                bad_count += 1

        # save the size database
        with open(dataset_size_file, 'w') as f:
            json.dump(self.size_database, f)

        if self.is_video:
            print_acc(f"  -  Found {len(self.file_list)} videos")
            assert len(self.file_list) > 0, f"no videos found in {self.dataset_path}"
        else:
            print_acc(f"  -  Found {len(self.file_list)} images")
            assert len(self.file_list) > 0, f"no images found in {self.dataset_path}"

        # handle x axis flips
        if self.dataset_config.flip_x:
            print_acc("  -  adding x axis flips")
            current_file_list = [x for x in self.file_list]
            for file_item in current_file_list:
                # create a copy that is flipped on the x axis
                new_file_item = copy.deepcopy(file_item)
                new_file_item.flip_x = True
                self.file_list.append(new_file_item)

        # handle y axis flips
        if self.dataset_config.flip_y:
            print_acc("  -  adding y axis flips")
            current_file_list = [x for x in self.file_list]
            for file_item in current_file_list:
                # create a copy that is flipped on the y axis
                new_file_item = copy.deepcopy(file_item)
                new_file_item.flip_y = True
                self.file_list.append(new_file_item)

        if self.dataset_config.flip_x or self.dataset_config.flip_y:
            if self.is_video:
                print_acc(f"  -  Found {len(self.file_list)} videos after adding flips")
            else:
                print_acc(f"  -  Found {len(self.file_list)} images after adding flips")

        logger.info(f"🔍 [DATASET_INIT] AiToolkitDataset 初始化完成")
        logger.info(f"🔍 [DATASET_INIT] - dataset_path: {self.dataset_path}")
        logger.info(f"🔍 [DATASET_INIT] - batch_size: {batch_size}")
        logger.info(f"🔍 [DATASET_INIT] - buckets: {dataset_config.buckets}")
        logger.info(f"🔍 [DATASET_INIT] - cache_latents: {self.is_caching_latents}")
        logger.info(f"🔍 [DATASET_INIT] - file_list 长度: {len(self.file_list) if hasattr(self, 'file_list') else 'unknown'}")

        self.setup_epoch()

    def setup_epoch(self):
        import logging
        logger = logging.getLogger(__name__)

        logger.info(f"🔍 [SETUP_EPOCH] setup_epoch 开始，epoch_num: {self.epoch_num}")

        # 检查是否在分布式训练中 - by Tsien at 2025-08-19
        has_accelerator = hasattr(self.sd, 'accelerator') and self.sd.accelerator is not None
        if has_accelerator:
            current_rank = self.sd.accelerator.state.process_index
            total_ranks = self.sd.accelerator.state.num_processes
            logger.info(f"🔍 [DISTRIBUTED] rank {current_rank}/{total_ranks} 开始 setup_epoch")

        if self.epoch_num == 0:
            # initial setup
            logger.info(f"🔍 [SETUP_EPOCH] 初始化 epoch (epoch_num=0)")

            if self.dataset_config.buckets:
                logger.info(f"🔍 [SETUP_EPOCH] 设置 buckets")
                self.setup_buckets()
                logger.info(f"🔍 [SETUP_EPOCH] buckets 设置完成，数量: {len(self.buckets) if hasattr(self, 'buckets') else 'unknown'}")
                # 分布式同步点1：buckets设置完成 - by Tsien at 2025-08-19
                if has_accelerator:
                    logger.info(f"🔄 [DISTRIBUTED] rank {current_rank} - buckets设置完成，等待同步...")
                    self.sd.accelerator.wait_for_everyone()
                    logger.info(f"✅ [DISTRIBUTED] rank {current_rank} - buckets同步完成")

            if self.is_caching_latents:
                logger.info(f"🔍 [SETUP_EPOCH] 缓存 latents")
                self.cache_latents_all_latents()
                # 分布式同步点2：latents缓存完成 - by Tsien at 2025-08-19
                if has_accelerator:
                    logger.info(f"🔄 [DISTRIBUTED] rank {current_rank} - latents缓存完成，等待同步...")
                    self.sd.accelerator.wait_for_everyone()
                    logger.info(f"✅ [DISTRIBUTED] rank {current_rank} - latents缓存同步完成")

            if self.is_caching_control_latents_to_disk:
                logger.info(f"🔍 [SETUP_EPOCH] 缓存 control latents")
                self.cache_control_latents_all()
                # 分布式同步点3：control latents缓存完成 - by Tsien at 2025-08-19
                if has_accelerator:
                    logger.info(f"🔄 [DISTRIBUTED] rank {current_rank} - control latents缓存完成，等待同步...")
                    self.sd.accelerator.wait_for_everyone()
                    logger.info(f"✅ [DISTRIBUTED] rank {current_rank} - control latents缓存同步完成")

            if self.is_caching_clip_vision_to_disk:
                logger.info(f"🔍 [SETUP_EPOCH] 缓存 CLIP vision")
                self.cache_clip_vision_to_disk()
                # 分布式同步点4：CLIP vision缓存完成 - by Tsien at 2025-08-19
                if has_accelerator:
                    logger.info(f"🔄 [DISTRIBUTED] rank {current_rank} - CLIP vision缓存完成，等待同步...")
                    self.sd.accelerator.wait_for_everyone()
                    logger.info(f"✅ [DISTRIBUTED] rank {current_rank} - CLIP vision缓存同步完成")

            if self.is_caching_text_embeddings:
                logger.info(f"🔍 [SETUP_EPOCH] 缓存文本嵌入")
                self.cache_text_embeddings()
                # 分布式同步点5：text embeddings缓存完成 - by Tsien at 2025-08-19
                if has_accelerator:
                    logger.info(f"🔄 [DISTRIBUTED] rank {current_rank} - 文本嵌入缓存完成，等待同步...")
                    self.sd.accelerator.wait_for_everyone()
                    logger.info(f"✅ [DISTRIBUTED] rank {current_rank} - 文本嵌入缓存同步完成")

            if self.is_generating_controls:
                logger.info(f"🔍 [SETUP_EPOCH] 设置控制")
                self.setup_controls()
                # 分布式同步点6：controls设置完成 - by Tsien at 2025-08-19
                if has_accelerator:
                    logger.info(f"🔄 [DISTRIBUTED] rank {current_rank} - controls设置完成，等待同步...")
                    self.sd.accelerator.wait_for_everyone()
                    logger.info(f"✅ [DISTRIBUTED] rank {current_rank} - controls设置同步完成")
        else:
            logger.info(f"🔍 [SETUP_EPOCH] 非初始化 epoch (epoch_num={self.epoch_num})")
            if self.dataset_config.poi is not None:
                logger.info(f"🔍 [SETUP_EPOCH] 重新设置 buckets (POI 模式)")
                self.setup_buckets(quiet=True)
                # 分布式同步点：POI buckets重设完成 - by Tsien at 2025-08-19
                if has_accelerator:
                    logger.info(f"🔄 [DISTRIBUTED] rank {current_rank} - POI buckets重设完成，等待同步...")
                    self.sd.accelerator.wait_for_everyone()
                    logger.info(f"✅ [DISTRIBUTED] rank {current_rank} - POI buckets重设同步完成")

        # 最终同步点：整个setup_epoch完成 - by Tsien at 2025-08-19
        if has_accelerator:
            logger.info(f"🔄 [DISTRIBUTED] rank {current_rank} - setup_epoch即将完成，最终同步...")
            self.sd.accelerator.wait_for_everyone()
            logger.info(f"🎉 [DISTRIBUTED] rank {current_rank} - setup_epoch最终同步完成！")

        self.epoch_num += 1
        logger.info(f"🔍 [SETUP_EPOCH] setup_epoch 完成，新的 epoch_num: {self.epoch_num}")

    def __len__(self):
        import logging
        logger = logging.getLogger(__name__)

        if self.dataset_config.buckets:
            length = len(self.batch_indices)
            logger.info(f"🔍 [DATASET] __len__ buckets 模式: {length} 批次")
            logger.info(f"🔍 [DATASET] - batch_indices 详情: {len(self.batch_indices)} 个批次")
            logger.info(f"🔍 [DATASET] - buckets 数量: {len(self.buckets) if hasattr(self, 'buckets') else 'unknown'}")
            return length
        else:
            length = len(self.file_list)
            logger.info(f"🔍 [DATASET] __len__ 标准模式: {length} 个文件")
            return length

    def _get_single_item(self, index) -> 'FileItemDTO':
        file_item: 'FileItemDTO' = copy.deepcopy(self.file_list[index])
        file_item.load_and_process_image(self.transform)
        file_item.load_caption(self.caption_dict)
        return file_item

    def __getitem__(self, item):
        import logging
        logger = logging.getLogger(__name__)

        if self.dataset_config.buckets:
            # for buckets we collate ourselves for now
            # todo allow a scheduler to dynamically make buckets
            # we collate ourselves
            logger.debug(f"🔍 [DATASET] __getitem__ buckets 模式, item: {item}")
            logger.debug(f"🔍 [DATASET] - batch_indices 长度: {len(self.batch_indices)}")

            if len(self.batch_indices) - 1 < item:
                # tried everything to solve this. No way to reset length when redoing things. Pick another index
                item = random.randint(0, len(self.batch_indices) - 1)
            idx_list = self.batch_indices[item]
            return [self._get_single_item(idx) for idx in idx_list]
        else:
            # Dataloader is batching
            return self._get_single_item(item)


def get_dataloader_from_datasets(
        dataset_options,
        batch_size=1,
        sd: 'StableDiffusion' = None,
        train_config=None,
) -> DataLoader:
    import logging
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    logger.info(f"🔍 [DATA_FLOW] get_dataloader_from_datasets 开始")
    logger.info(f"🔍 [DATA_FLOW] dataset_options 数量: {len(dataset_options) if dataset_options else 0}")
    logger.info(f"🔍 [DATA_FLOW] batch_size: {batch_size}")
    logger.info(f"🔍 [DATA_FLOW] sd 模型: {sd.__class__.__name__ if sd else None}")

    # 检查分布式训练配置，如果启用则自动禁用 buckets - by Tsien at 2025-08-18
    is_distributed_training = False
    if train_config and hasattr(train_config, 'distributed_training'):
        is_distributed_training = getattr(train_config, 'distributed_training', False)
        logger.info(f"🔍 [DISTRIBUTED] 分布式训练状态: {is_distributed_training}")

    if is_distributed_training:
        logger.info(f"🔧 [DISTRIBUTED] 检测到分布式训练，自动禁用所有数据集的 buckets 模式以解决架构冲突")
        logger.info(f"📝 [DISTRIBUTED] 原因: buckets 模式的 batch_size=None 与 Accelerate 的分布式训练不兼容")

    if dataset_options is None or len(dataset_options) == 0:
        logger.warning(f"⚠️ [DATA_FLOW] dataset_options 为空，返回 None")
        return None

    datasets = []
    has_buckets = False
    is_caching_latents = False

    dataset_config_list = []
    # preprocess them all
    for dataset_option in dataset_options:
        if isinstance(dataset_option, DatasetConfig):
            config_to_add = dataset_option
            # 如果是分布式训练或有定制化形状，创建一个副本并禁用 buckets
            needs_bucket_disable = (is_distributed_training and config_to_add.buckets) or \
                                   (config_to_add.customized_shape is not None and config_to_add.buckets)

            if needs_bucket_disable:
                if is_distributed_training:
                    logger.info(f"🔧 [DISTRIBUTED] 数据集 {config_to_add.folder_path} 原本启用 buckets，现禁用以支持分布式训练")
                if config_to_add.customized_shape is not None:
                    logger.info(f"🔧 [CUSTOMIZED_SHAPE] 数据集 {config_to_add.folder_path} 原本启用 buckets，现禁用以支持固定形状训练 {config_to_add.customized_shape}")
                # 创建配置副本并禁用 buckets
                config_dict = config_to_add.__dict__.copy()
                config_dict['buckets'] = False
                config_to_add = DatasetConfig(**config_dict)
            dataset_config_list.append(config_to_add)
        else:
            # preprocess raw data
            split_configs = preprocess_dataset_raw_config([dataset_option])
            for x in split_configs:
                config_dict = x.copy()
                # 如果是分布式训练或有定制化形状，禁用 buckets
                needs_bucket_disable = (is_distributed_training and config_dict.get('buckets', True)) or \
                                       (config_dict.get('customized_shape') is not None and config_dict.get('buckets', True))

                if needs_bucket_disable:
                    if is_distributed_training:
                        logger.info(f"🔧 [DISTRIBUTED] 数据集 {config_dict.get('folder_path', 'unknown')} 原本启用 buckets，现禁用以支持分布式训练")
                    if config_dict.get('customized_shape') is not None:
                        logger.info(f"🔧 [CUSTOMIZED_SHAPE] 数据集 {config_dict.get('folder_path', 'unknown')} 原本启用 buckets，现禁用以支持固定形状训练 {config_dict.get('customized_shape')}")
                    config_dict['buckets'] = False
                dataset_config_list.append(DatasetConfig(**config_dict))

    for i, config in enumerate(dataset_config_list):
        logger.info(f"🔍 [DATA_FLOW] 处理数据集 {i+1}/{len(dataset_config_list)}")
        logger.info(f"🔍 [DATA_FLOW] - 类型: {config.type}")
        logger.info(f"🔍 [DATA_FLOW] - 路径: {config.folder_path}")
        logger.info(f"🔍 [DATA_FLOW] - buckets: {config.buckets}")
        logger.info(f"🔍 [DATA_FLOW] - cache_latents: {config.cache_latents}")
        logger.info(f"🔍 [DATA_FLOW] - cache_latents_to_disk: {config.cache_latents_to_disk}")

        if config.type == 'image':
            dataset = AiToolkitDataset(config, batch_size=batch_size, sd=sd)
            datasets.append(dataset)

            logger.info(f"🔍 [DATA_FLOW] - 创建的数据集大小: {len(dataset)}")
            logger.info(f"🔍 [DATA_FLOW] - 数据集文件数量: {len(dataset.file_list) if hasattr(dataset, 'file_list') else 'unknown'}")

            if config.buckets:
                has_buckets = True
                logger.info(f"🔍 [DATA_FLOW] - 启用 buckets 模式")
            if config.cache_latents or config.cache_latents_to_disk:
                is_caching_latents = True
                logger.info(f"🔍 [DATA_FLOW] - 启用 latents 缓存")
        else:
            raise ValueError(f"invalid dataset type: {config.type}")

    concatenated_dataset = ConcatDataset(datasets)
    logger.info(f"🔍 [DATA_FLOW] 合并数据集完成，总大小: {len(concatenated_dataset)}")
    logger.info(f"🔍 [DATA_FLOW] has_buckets: {has_buckets}")
    logger.info(f"🔍 [DATA_FLOW] is_caching_latents: {is_caching_latents}")

    # todo build scheduler that can get buckets from all datasets that match
    # todo and evenly distribute reg images

    def dto_collation(batch: List['FileItemDTO']):
        # create DTO batch
        batch = DataLoaderBatchDTO(
            file_items=batch
        )
        return batch

    # check if is caching latents

    dataloader_kwargs = {}

    if is_native_windows():
        dataloader_kwargs['num_workers'] = 0
    else:
        # 分布式训练时强制使用 num_workers=0 避免卡死 - by Tsien at 2025-01-27
        if is_distributed_training:
            dataloader_kwargs['num_workers'] = 0
            dataloader_kwargs['prefetch_factor'] = 1
            logger.info(f"🔧 [DISTRIBUTED] 强制设置 num_workers=0, prefetch_factor=1 避免 DataLoader 卡死")
            # del dataloader_kwargs['num_workers']
            del dataloader_kwargs['prefetch_factor']
        else:
            dataloader_kwargs['num_workers'] = dataset_config_list[0].num_workers
            dataloader_kwargs['prefetch_factor'] = dataset_config_list[0].prefetch_factor

    if has_buckets:
        # make sure they all have buckets
        for dataset in datasets:
            assert dataset.dataset_config.buckets, f"buckets not found on dataset {dataset.dataset_config.folder_path}, you either need all buckets or none"

        logger.info(f"🔍 [DATA_FLOW] 创建 buckets 模式 DataLoader")
        logger.info(f"🔍 [DATA_FLOW] - batch_size: None (buckets 自行批处理)")
        logger.info(f"🔍 [DATA_FLOW] - drop_last: False")
        logger.info(f"🔍 [DATA_FLOW] - shuffle: True")
        logger.info(f"🔍 [DATA_FLOW] - dataloader_kwargs: {dataloader_kwargs}")

        data_loader = DataLoader(
            concatenated_dataset,
            batch_size=None,  # we batch in the datasets for now
            drop_last=False,
            shuffle=True,
            collate_fn=dto_collation,  # Use the custom collate function
            **dataloader_kwargs
        )
    else:
        logger.info(f"🔍 [DATA_FLOW] 创建标准模式 DataLoader")
        logger.info(f"🔍 [DATA_FLOW] - batch_size: {batch_size}")
        logger.info(f"🔍 [DATA_FLOW] - shuffle: True")
        logger.info(f"🔍 [DATA_FLOW] - dataloader_kwargs: {dataloader_kwargs}")

        data_loader = DataLoader(
            concatenated_dataset,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=dto_collation,
            **dataloader_kwargs
        )

    logger.info(f"🔍 [DATA_FLOW] DataLoader 创建完成")
    logger.info(f"🔍 [DATA_FLOW] - DataLoader 长度: {len(data_loader)}")
    logger.info(f"🔍 [DATA_FLOW] - DataLoader batch_size: {getattr(data_loader, 'batch_size', 'None')}")
    logger.info(f"🔍 [DATA_FLOW] - DataLoader dataset 长度: {len(data_loader.dataset)}")

    return data_loader


def trigger_dataloader_setup_epoch(dataloader: DataLoader):
    # hacky but needed because of different types of datasets and dataloaders
    dataloader.len = None
    if isinstance(dataloader.dataset, list):
        for dataset in dataloader.dataset:
            if hasattr(dataset, 'datasets'):
                for sub_dataset in dataset.datasets:
                    if hasattr(sub_dataset, 'setup_epoch'):
                        sub_dataset.setup_epoch()
                        sub_dataset.len = None
            elif hasattr(dataset, 'setup_epoch'):
                dataset.setup_epoch()
                dataset.len = None
    elif hasattr(dataloader.dataset, 'setup_epoch'):
        dataloader.dataset.setup_epoch()
        dataloader.dataset.len = None
    elif hasattr(dataloader.dataset, 'datasets'):
        dataloader.dataset.len = None
        for sub_dataset in dataloader.dataset.datasets:
            if hasattr(sub_dataset, 'setup_epoch'):
                sub_dataset.setup_epoch()
                sub_dataset.len = None

def get_dataloader_datasets(dataloader: DataLoader):
    # hacky but needed because of different types of datasets and dataloaders
    if isinstance(dataloader.dataset, list):
        datasets = []
        for dataset in dataloader.dataset:
            if hasattr(dataset, 'datasets'):
                for sub_dataset in dataset.datasets:
                    datasets.append(sub_dataset)
            else:
                datasets.append(dataset)
        return datasets
    elif hasattr(dataloader.dataset, 'datasets'):
        return dataloader.dataset.datasets
    else:
        return [dataloader.dataset]
