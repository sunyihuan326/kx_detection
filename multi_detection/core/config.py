
from easydict import EasyDict as edict


__C                             = edict()
# Consumers can get config by: from config import cfg

cfg                             = __C

# YOLO options
__C.YOLO                        = edict()

# Set the class name
__C.YOLO.LAYER_NUMS             = 4
__C.YOLO.CLASSES                = "E:/kx_detection/multi_detection/data/classes/food40.names"
__C.YOLO.ANCHORS                = "./data/anchors/food_anchors.txt"
__C.YOLO.MOVING_AVE_DECAY       = 0.9995
__C.YOLO.STRIDES                = [8, 16, 32]
__C.YOLO.ANCHOR_PER_SCALE       = 3
__C.YOLO.IOU_LOSS_THRESH        = 0.5
__C.YOLO.UPSAMPLE_METHOD        = "resize"
__C.YOLO.ORIGINAL_WEIGHT        = "E:/ckpt_dirs/Food_detection/multi_food3/checkpoint/yolov3_train_loss=11.4018.ckpt-98"
__C.YOLO.DEMO_WEIGHT            = "E:/ckpt_dirs/Food_detection/multi_food3/checkpoint/yolov3_train_loss=11.4018.ckpt-98"

# Train options
__C.TRAIN                       = edict()

__C.TRAIN.ANNOT_PATH            = "E:/DataSets/X_3660_data/train39_zi_hot_and_old_strand_900hotdog.txt"
__C.TRAIN.BATCH_SIZE            = 2
__C.TRAIN.INPUT_SIZE            = 320
__C.TRAIN.DATA_AUG              = True
__C.TRAIN.LEARN_RATE_INIT       = 1e-4
__C.TRAIN.LEARN_RATE_END        = 1e-6
__C.TRAIN.WARMUP_EPOCHS         = 2
__C.TRAIN.FISRT_STAGE_EPOCHS    = 5
__C.TRAIN.SECOND_STAGE_EPOCHS   = 50
__C.TRAIN.INITIAL_WEIGHT        = "./checkpoint/yolov3_train_loss=3.7270.ckpt-33"



# TEST options
__C.TEST                        = edict()

# __C.TEST.ANNOT_PATH             = "./data/dataset/foodSets1105_XandOld_test27.txt"
__C.TEST.ANNOT_PATH             = "E:/DataSets/X_3660_data/test39.txt"
__C.TEST.BATCH_SIZE             = 2
__C.TEST.INPUT_SIZE             = 320
__C.TEST.DATA_AUG               = False
__C.TEST.WRITE_IMAGE            = True
__C.TEST.WRITE_IMAGE_PATH       = "E:/kx_detection/multi_detection/data/detection"
__C.TEST.WRITE_IMAGE_SHOW_LABEL = False
__C.TEST.WEIGHT_FILE            = ""
__C.TEST.SHOW_LABEL             = True
__C.TEST.SCORE_THRESHOLD        = 0.45
__C.TEST.IOU_THRESHOLD          = 0.5






