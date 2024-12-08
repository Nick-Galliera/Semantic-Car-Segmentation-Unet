import os 
import torch

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
LOG_DIR = os.path.join(SCRIPT_DIR, '../logs')

''' Dataset folder and subfolders '''
DATASET_FOLDER = "../car-segmentation"
IMAGE_FOLDER = os.path.join(SCRIPT_DIR, DATASET_FOLDER, 'images')
MASKS_FOLDER = os.path.join(SCRIPT_DIR, DATASET_FOLDER, 'masks')
JSON_FILE = os.path.join(SCRIPT_DIR, DATASET_FOLDER, 'masks.json')

''' Class definition '''
CLASSES = {
    0 : 'background',
    1 : 'car',
    2 : 'wheel',
    3 : 'lights',
    4 : 'window',
}
NUM_CLASSES = 5

if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
elif torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
else:
    DEVICE = torch.device("cpu")

''' runtime flags '''
SHOW_DATASET_PREVIEW = False
AUGMENT = False
TENSORBOARD_VISUALIZE = False

''' dataset params '''
TRAIN_SIZE = 0.8
AUGMENT_SIZE = 0.9
BATCH_SIZE = 32

''' pretrained model params '''
#PRETRAINED_WEIGHTS = os.path.join(SCRIPT_DIR, '../weights', 'pretrained_model_unet_res18_weights.pth')
#PRETRAINED_BACKBONE = 'resnet18'
#PRETRAINED_WEIGHTS = os.path.join(SCRIPT_DIR, '../weights', 'pretrained_model_unet_res34_weights.pth')
#PRETRAINED_BACKBONE = 'resnet34'
PRETRAINED_WEIGHTS = os.path.join(SCRIPT_DIR, '../weights', 'pretrained_model_unet_res101_weights.pth')
PRETRAINED_BACKBONE = 'resnet101'
PRETRAINED_VISUALIZE_MODEL = False
TRAIN_PRETRAINED = False
TEST_PRETRAINED = True
PRETRAINED_LR = 1e-5
PRETRAINED_EPOCHS = 50



