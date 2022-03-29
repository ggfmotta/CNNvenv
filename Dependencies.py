# Tensor Flow Dependencies
from tensorflow.examples.tutorials.mnist import input_data

import tensorflow as tf
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D
from tensorflow.python.keras.callbacks import TensorBoard, ModelCheckpoint
import tensorflow.python.keras.backend as K
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from tensorflow.python.keras.optimizers import Adam
import joblib

# External Library Dependencies
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import csv
import cv2
import random
import pickle
import time

