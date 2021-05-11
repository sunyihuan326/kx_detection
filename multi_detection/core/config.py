from easydict import EasyDict as edict

__C = edict()
# Consumers can get config by: from config import cfg

cfg = __C

# YOLO options
__C.YOLO = edict()

# Set the class name
__C.YOLO.LAYER_NUMS                  = 4
__C.YOLO.CLASSES                     = "E:/kx_detection/multi_detection/data/classes/food42.names"
__C.YOLO.ANCHORS                     = "./data/anchors/basline_anchors.txt"
__C.YOLO.MOVING_AVE_DECAY            = 0.9995
__C.YOLO.STRIDES                     = [8, 16, 32]
__C.YOLO.ANCHOR_PER_SCALE            = 3
__C.YOLO.IOU_LOSS_THRESH             = 0.5
__C.YOLO.UPSAMPLE_METHOD             = "resize"
__C.YOLO.ORIGINAL_WEIGHT             = "F:/BaiduNetdiskDownload/tensorflow-yolov3/yolov3_coco.ckpt"
__C.YOLO.DEMO_WEIGHT                 = "./yolo3_coco/yolov3_coco.ckpt"

# Train options
__C.TRAIN = edict()

__C.TRAIN.ANNOT_PATH                 = "E:/DataSets/model_data/train42.txt"
__C.TRAIN.BATCH_SIZE                 = 4
__C.TRAIN.INPUT_SIZE                 = 320
__C.TRAIN.DATA_AUG                   = True
__C.TRAIN.LEARN_RATE_INIT            = 1e-4
__C.TRAIN.LEARN_RATE_END             = 1e-6
__C.TRAIN.WARMUP_EPOCHS              = 10
__C.TRAIN.FISRT_STAGE_EPOCHS         = 56
__C.TRAIN.SECOND_STAGE_EPOCHS        = 20
# __C.TRAIN.INITIAL_WEIGHT             = "E:/ckpt_dirs/Food_detection/multi_food3/20200604_22class/yolov3_train_loss=4.9799.ckpt-158"
__C.TRAIN.INITIAL_WEIGHT             = "./checkpoint/yolov3_train_loss=11.2574.ckpt-56"

# TEST options
__C.TEST = edict()

__C.TEST.ANNOT_PATH                  ="E:/DataSets/model_data/test42.txt"
__C.TEST.BATCH_SIZE                  = 2
__C.TEST.INPUT_SIZE                  = 320
__C.TEST.DATA_AUG                    = False
__C.TEST.WRITE_IMAGE                 = True
__C.TEST.WRITE_IMAGE_PATH            = "./data/detection"
__C.TEST.WRITE_IMAGE_SHOW_LABEL      = False
__C.TEST.WEIGHT_FILE                 = "E:/ckpt_dirs/Food_detection/tf_yolov3/20190823/yolov3_train_loss=9.3529.ckpt-51"
__C.TEST.SHOW_LABEL                  = True
__C.TEST.SCORE_THRESHOLD             = 0.45
__C.TEST.IOU_THRESHOLD               = 0.5
