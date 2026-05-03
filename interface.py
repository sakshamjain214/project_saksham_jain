# interface.py

from config import *
from dataset import get_dataloaders
from model import DeepfakeDetectorCNN
from train import train_model
from predict import classify_images