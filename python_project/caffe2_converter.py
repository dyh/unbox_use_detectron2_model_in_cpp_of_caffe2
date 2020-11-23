#!/usr/bin/env python
import json
import os
import shutil

import cv2
import numpy as np
from detectron2.config import get_cfg
from detectron2.structures import BoxMode
from detectron2.modeling import build_model
from detectron2.utils.logger import setup_logger
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.data.datasets import register_coco_instances
from detectron2.export import Caffe2Tracer, add_export_config
from detectron2.data.datasets.coco import convert_to_coco_json
from detectron2.evaluation import COCOEvaluator, inference_on_dataset, print_csv_format
from detectron2.data import build_detection_test_loader, DatasetCatalog, MetadataCatalog


def get_dataset_dicts_from_via_file(img_dir, json_file_name):
    """
    load dataset from
    :param img_dir:
    :param json_file_name:
    :return:
    """
    # json_file_name = "via_region_data.json"
    json_file = os.path.join(img_dir, json_file_name)
    with open(json_file) as f:
        imgs_anns = json.load(f)

    dataset_dicts = []
    for idx, v in enumerate(imgs_anns.values()):
        record = {}

        filename = os.path.join(img_dir, v["filename"])
        height, width = cv2.imread(filename).shape[:2]

        record["file_name"] = v["filename"]
        # record["file_name"] = filename
        record["image_id"] = idx
        record["height"] = height
        record["width"] = width

        list_annos = v["regions"]

        objs = []
        for dict_anno in list_annos:
            anno = dict_anno["shape_attributes"]
            px = anno["all_points_x"]
            py = anno["all_points_y"]
            poly = [(x + 0.5, y + 0.5) for x, y in zip(px, py)]
            poly = [p for x in poly for p in x]

            # get type from region_attributes to set different category_id
            attr1 = dict_anno["region_attributes"]
            type1 = attr1["type"]

            if type1 == "fissure":
                cat_id = 0
            elif type1 == "water":
                cat_id = 1
            else:
                cat_id = 0

            obj = {
                "bbox": [np.min(px), np.min(py), np.max(px), np.max(py)],
                "bbox_mode": BoxMode.XYXY_ABS,
                "segmentation": [poly],
                "category_id": cat_id,
            }
            objs.append(obj)

        record["annotations"] = objs
        dataset_dicts.append(record)

    return dataset_dicts


if __name__ == "__main__":
    # pip install torch==1.6.0+cu101 torchvision==0.7.0+cu101 -f https://download.pytorch.org/whl/torch_stable.html
    # python -m pip install detectron2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cu101/torch1.6/index.html
    # pip install opencv-python==4.4.0.46
    # pip install onnx==1.8.0
    # pip install protobuf==3.14.0

    image_root = "images/train"

    # setup the output folder
    output_dir = "output"

    # annotation file from VGG Image Annotator
    # please follow section [4.annotate your own training sample (optional)] in
    # https://github.com/dyh/unbox_detecting_tunnel_fissure/blob/main/tunnel_fissure.ipynb
    via_region_data_file_name = "via_region_data.json"

    # delete and create output folder
    shutil.rmtree(output_dir, ignore_errors=True)
    os.makedirs(output_dir, exist_ok=True)

    temp_dataset_name = "fissures_temp"
    DatasetCatalog.register(temp_dataset_name, lambda f="": get_dataset_dicts_from_via_file(
        image_root, via_region_data_file_name))

    MetadataCatalog.get(temp_dataset_name).set(thing_classes=["fissure", "water"])
    # MetadataCatalog.get(dataset_temp_name).set(evaluator_type="coco")

    # save coco format json file
    temp_coco_json_file_path = os.path.join(output_dir, "fissures_temp_coco_format.json")
    convert_to_coco_json(temp_dataset_name, temp_coco_json_file_path)

    dataset_train_name = "fissures_train"
    register_coco_instances(name=dataset_train_name, metadata={},
                            json_file=temp_coco_json_file_path,
                            image_root=image_root)

    logger = setup_logger()
    # logger.info("Command line arguments: " + str(args))

    cfg = get_cfg()
    # cuda context is initialized before creating dataloader, so we don't fork anymore
    cfg.DATALOADER.NUM_WORKERS = 0
    cfg = add_export_config(cfg)
    cfg.merge_from_file("configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
    cfg.MODEL.WEIGHTS = "weights/model_0124999.pth"
    cfg.MODEL.DEVICE = "cpu"
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 2  # has two classes(fissure, water).
    cfg.freeze()

    # create a torch model
    torch_model = build_model(cfg)
    DetectionCheckpointer(torch_model).resume_or_load(cfg.MODEL.WEIGHTS)

    # get a sample data
    data_loader = build_detection_test_loader(cfg, dataset_train_name)
    first_batch = next(iter(data_loader))

    # convert and save caffe2 model
    tracer = Caffe2Tracer(cfg, torch_model, first_batch)
    caffe2_model = tracer.export_caffe2()
    caffe2_model.save_protobuf(output_dir)

    # draw the caffe2 graph
    caffe2_model.save_graph(os.path.join(output_dir, "model.svg"), inputs=first_batch)
    data_loader = build_detection_test_loader(cfg, dataset_train_name)

    # NOTE: hard-coded evaluator. change to the evaluator for your dataset
    # evaluator = COCOEvaluator(dataset_name, output_dir=output_dir)
    evaluator = COCOEvaluator(dataset_train_name, cfg, True, output_dir)

    metrics = inference_on_dataset(caffe2_model, data_loader, evaluator)
    print_csv_format(metrics)
