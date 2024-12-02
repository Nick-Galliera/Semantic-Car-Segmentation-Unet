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

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

''' runtime flags '''
SHOW_DATASET_PREVIEW = False
AUGMENT = True
TENSORBOARD_VISUALIZE = True

''' dataset params '''
TRAIN_SIZE = 0.8
AUGMENT_SIZE = 0.8
BATCH_SIZE = 32

''' pretrained model params '''
PRETRAINED_WEIGHTS = os.path.join(SCRIPT_DIR, '../weights', 'pretrained_model_weights.pth')
PRETRAINED_VISUALIZE_MODEL = False
PRETRAINED_BACKBONE = 'resnet18'
PRETRAINED_LR = 1e-4
PRETRAINED_EPOCHS = 1
TRAIN_PRETRAINED = True

