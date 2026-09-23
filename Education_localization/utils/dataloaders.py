import xml.etree.ElementTree as ET
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from pathlib import Path
from PIL import Image
from torchvision import tv_tensors
from .convert_box import xyxy2xywh
from .bbox_iou import iou_wh
from .augmentation import DataAugment
from torch.utils.data import Subset


class PascalVOC:
    def __init__(self, dirs):
        if isinstance(dirs, str):
            dirs = [dirs]

        if not isinstance(dirs, list):
            raise TypeError("Параметр dirs должен быть или строкой, или списком содержащий путь/пути до папки с изображением")

        self.dirs = dirs
        self.img_formats = ".bmp", ".dng", ".jpeg", ".jpg", ".mpo", ".png", ".tif", ".tiff", ".webp", ".pfm"

        classes = [
            "person", "bird", "cat", "cow", "dog", "horse", "sheep", "bus",
            "aeroplane", "bicycle", "boat", "car", "motorbike", "train",
            "bottle", "chair", "diningtable", "pottedplant", "sofa", "tvmonitor",
        ]

        self.classes = sorted(classes)
        self.class_to_idx = {cls: i for i, cls in enumerate(self.classes)}
        self.idx_to_class = {i: cls for i, cls in enumerate(self.classes)}

    def create_data(self):

        target_data = []
        for path in self.dirs:

            target_data_on_path = []
            self.path = Path(path)
            #print(path)
            self.path_anns = self.path.joinpath("Annotations")
            self.path_imgs = self.path.joinpath("JPEGImages")

            for im_f in sorted(self.path_imgs.iterdir()):
                true_format = im_f.suffix in self.img_formats
                ann_f = Path(self.path_anns, f"{im_f.stem}.xml")
                exists = ann_f.exists()

                if not true_format or not exists:
                    continue

                img_data = {}
                ann_ps = ET.parse(ann_f)
                img_data["path_img"] = im_f
                np_f = Path(im_f).with_suffix(".npy")

                if not np_f.exists():
                    np.save(np_f.as_posix(), np.array(Image.open(im_f).convert("RGB")))

                img_data["img_name"] = ann_ps.find("filename").text

                w = int(ann_ps.find("size").find("width").text)
                h = int(ann_ps.find("size").find("height").text)
                c = int(ann_ps.find("size").find("depth").text)
                img_data["size_img"] = [w, h]
                img_data["depth"] = c

                objs_ann = []

                for obj in ann_ps.findall("object"):
                    obj_ann = {}
                    obj_ann["label"] = self.class_to_idx[obj.find("name").text]
                    bbox_ann = obj.find("bndbox")
                    bbox = [
                        int(float(bbox_ann.find("xmin").text))-1,
                        int(float(bbox_ann.find("ymin").text))-1,
                        int(float(bbox_ann.find("xmax").text))-1,
                        int(float(bbox_ann.find("ymax").text))-1,
                    ]

                    obj_ann["bbox"] = bbox

                    objs_ann.append(obj_ann)

                img_data["objs_ann"] = objs_ann
                target_data_on_path.append(img_data)

            if len(target_data_on_path) == 0:
                print(f"\033[33m\033[1mWarning\033[0m: В папке \033[34m'{self.path}'\033[0m изображений имеющих разметку не найдено.")
                continue

            target_data += target_data_on_path

        assert target_data, f"\033[31mИзображений имеющих разметку не найдено.\033[0m"
        print(f"В наборе данных \033[34m\033[1m{len(target_data)}\033[0m размеченных изображений.")

        return target_data


class DatasetPascalVOC(Dataset):
    def __init__(self, data, mode=None, transform=None, image_size=300):
        self.data = data
        self.image_size = image_size
        self.transform = transform if transform else DataAugment(mode, image_size=image_size)
        self.C = 20

        classes = [
                    "person", "bird", "cat", "cow", "dog", "horse", "sheep", "bus",
                    "aeroplane", "bicycle", "boat", "car", "motorbike", "train",
                    "bottle", "chair", "diningtable", "pottedplant", "sofa", "tvmonitor",
                ]
        
        self.classes = sorted(classes)
        self.class_to_idx = {cls: i for i, cls in enumerate(self.classes)}
        self.idx_to_class = {i: cls for i, cls in enumerate(self.classes)}


    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        data = self.data[index]
        path_img = data["path_img"]

        np_f = Path(path_img).with_suffix(".npy")
        if np_f.exists():
            img = np.load(np_f.as_posix()).transpose((2, 0, 1))
            img = torch.as_tensor(img)

        else:
            img = Image.open(path_img).convert("RGB")

        W, H = data["size_img"]
        bboxes = [obj_ann["bbox"] for obj_ann in data["objs_ann"]]
        bboxes = tv_tensors.BoundingBoxes(
            bboxes,
            format="XYXY",
            canvas_size=[H, W]
        )

        labels = [obj_ann["label"] for obj_ann in data["objs_ann"]]

        target_bl = {
            "boxes" : bboxes,
            "labels" : torch.tensor(labels)
        }
        img, target_bl = self.transform(img, target_bl)

        bboxes = target_bl["boxes"]
        labels = target_bl["labels"]

        whwh = torch.tensor([self.image_size, self.image_size, self.image_size, self.image_size])
        bboxes_xyxy = bboxes / whwh
        boxes_classes = torch.cat([xyxy2xywh(bboxes_xyxy), labels.unsqueeze(dim=1)], dim=1)

        
        return img, boxes_classes

    @staticmethod
    def collate_fn(batch):

        data = list(zip(*batch))
        imgs = torch.stack(data[0])
        boxes_classes = data[1]
        return imgs, list(boxes_classes)

class DatasetPascalVocYOLOv3(DatasetPascalVOC):
    def __init__(self, data, mode=None, transform=None, image_size=300, list_size=[52, 26, 13], achors: torch.Tensor=None):
        super().__init__(data, mode, transform, image_size=image_size)
        self.list_size = list_size
        self.achors = achors
        self.mode = mode

        paths, sizes, boxes, labels, offsets = [], [], [], [], [0]

        for item in data:
            paths.append(str(item["path_img"]))
            sizes.append(item["size_img"])
            for obj in item["objs_ann"]:
                boxes.append(obj["bbox"])
                labels.append(obj["label"])
            offsets.append(len(boxes))

        self.paths = np.array(paths)
        self.sizes = np.array(sizes, dtype=np.int32)
        self.boxes = np.array(boxes, dtype=np.float32) if boxes else np.zeros((0, 4), dtype=np.float32)
        self.labels = np.array(labels, dtype=np.int64)
        self.offsets = np.array(offsets, dtype=np.int64)

        
    def __getitem__(self, index):
        path_img = self.paths[index]
        W, H = self.sizes[index]
        start, end = self.offsets[index], self.offsets[index + 1]
        bboxes_np = self.boxes[start:end]
        labels_np = self.labels[start:end]


        np_f = Path(path_img).with_suffix(".npy")
        if np_f.exists():
            img = np.load(np_f.as_posix()).transpose((2, 0, 1))
            img = torch.as_tensor(img)

        else:
            img = Image.open(path_img).convert("RGB")

        bboxes = tv_tensors.BoundingBoxes(
            torch.from_numpy(bboxes_np.copy()),
            format="XYXY",
            canvas_size=[int(H), int(W)],
        )
        labels = torch.from_numpy(labels_np.copy())

        target_bl = {
            "bboxes" : bboxes,
            "labels" : labels.detach().clone()
        }

        img, target_bl = self.transform(img, target_bl)

        bboxes = target_bl["bboxes"]
        labels = target_bl["labels"]

        whwh = torch.tensor([self.image_size, self.image_size, self.image_size, self.image_size])
        bboxes_xyxy = bboxes / whwh
        if self.mode == "test":
            boxes_classes = torch.cat([bboxes_xyxy.as_subclass(torch.Tensor), labels.unsqueeze(dim=1)], dim=1)
            return img, boxes_classes

        bboxes = xyxy2xywh(bboxes_xyxy)

        iou_matrix = iou_wh(bboxes[:, 2:], self.achors)
        max_similar = iou_matrix.argmax(dim=1)

        scale_idx = torch.div(max_similar, 3, rounding_mode="floor")
        achor_idx = max_similar % 3

        small_objects, medium_objects, large_objects = self._create_output(scale_idx, achor_idx, bboxes, labels)
        gt_boxes = bboxes_xyxy.as_subclass(torch.Tensor)

        return img, small_objects, medium_objects, large_objects, gt_boxes

    def _create_output(self, scale_idx, achor_idx, bboxes, labels):
        small_objects = torch.zeros(len(self.list_size), self.list_size[0], self.list_size[0], 5 + self.C)
        medium_objects = torch.zeros(len(self.list_size), self.list_size[1], self.list_size[1], 5 + self.C)
        large_objects = torch.zeros(len(self.list_size), self.list_size[2], self.list_size[2], 5 + self.C)

        list_objects = [small_objects, medium_objects, large_objects]
        for i in range(3):
            S = list_objects[i].shape[1]
            w_cell = 1/S
            h_cell = 1/S
            wh_cell = torch.tensor([w_cell, h_cell])

            mask = scale_idx == i
            bboxes_mask = bboxes[mask]
            labels_mask = labels[mask]

            n_l = len(labels_mask)
            cls_labels = torch.zeros(n_l, self.C)
            cls_labels[[*range(n_l)], labels_mask] = 1
            conf = torch.ones(n_l, 1)

            idx = torch.floor(bboxes_mask[:, :2] / wh_cell)
            ij = torch.tensor_split(idx.long(), 2, dim=1)
            i_idx = ij[0].squeeze(1)
            j_idx = ij[1].squeeze(1)

            anchor_wh = self.achors[3 * scale_idx[mask] + achor_idx[mask]]

            bboxes_mask[:, :2] = (bboxes_mask[:, :2] % wh_cell) / wh_cell
            bboxes_mask[:, 2:] = torch.log(bboxes_mask[:, 2:]/ anchor_wh + 1e-16)
            bboxes_conf_labels = torch.cat([bboxes_mask, conf, cls_labels], dim=1)
            list_objects[i][achor_idx[mask], j_idx, i_idx] = bboxes_conf_labels[:, :]
            list_objects[i] = list_objects[i].permute(1, 2, 0, 3)

        return list_objects

    @staticmethod
    def collate_fn(batch):
        data = list(zip(*batch))
        if len(data) == 5:
            imgs = torch.stack(data[0])
            small_objects = torch.stack(data[1])
            medium_objects = torch.stack(data[2])
            large_objects = torch.stack(data[3])
            gt_boxes = list(data[4])

            return imgs, (small_objects, medium_objects, large_objects, gt_boxes)

        imgs = torch.stack(data[0])
        boxes_classes = data[1]
        return imgs, list(boxes_classes)

        
def create_loader(
    path,
    split = [0.8, 0.2],
    batch_size=16,
    workers=1,
    pin_memory=True,
    shuffle_train=True,
    image_size=300,
    dataset_class = DatasetPascalVOC,
    seed=42,
    **kwargs,
):
    data = PascalVOC(path).create_data()
    generator = torch.Generator().manual_seed(seed)
    train_data, val_data = random_split(data, split, generator=generator)

    train_set = dataset_class(train_data, mode="train", image_size=image_size, **kwargs)
    #train_set = Subset(train_set, range(10))
    val_set = dataset_class(val_data, mode="test", image_size=image_size, **kwargs)
    #val_set = Subset(val_set, range(10))
    train_loader = DataLoader(
        train_set,
        batch_size,
        shuffle=shuffle_train,
        num_workers=workers,
        pin_memory=pin_memory,
        persistent_workers=True,
        collate_fn=dataset_class.collate_fn
    )
    val_loader = DataLoader(
            val_set,
            batch_size,
            shuffle=False,
            num_workers=workers,
            pin_memory=pin_memory,
            persistent_workers=True,
            collate_fn=dataset_class.collate_fn
        )

    return train_loader, val_loader