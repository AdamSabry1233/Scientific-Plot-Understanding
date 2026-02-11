import json
from pathlib import Path
from stage2_ocr import stage2   # import your function
import traceback
import matplotlib.pyplot as plt
import numpy as np
import random
from ultralytics import YOLO
from PIL import Image, ImageDraw
import re
import easyocr
import cv2
import sys

