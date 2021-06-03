import os
import cv2

import matplotlib.pyplot as plt
from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog

test_image = '/data/coco/val2017/000000411938.jpg'

cfg = get_cfg()
cfg.OUTPUT_DIR = "../output_cascade/"
# cfg.merge_from_file(model_zoo.get_config_file("../results/output_cascade/config.yaml"))
cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")  # path to the model we just trained
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7   # set a custom testing threshold
predictor = DefaultPredictor(cfg)

im = cv2.imread(test_image)
plt.imshow("Image_original", im)
cv2.waitKey(0)
plt.show()

outputs = predictor(im)
v = Visualizer(im[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.2)
out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
plt.imshow("Image_instance_pre", out.get_image()[:, :, ::-1])
cv2.waitKey(0)
plt.show()
