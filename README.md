# AMLSassignment-

This code trains and tests a facial recognition model aimed at solving several classification tasks. The binary classification tasks which were carried out were: noise detection for no faces, emotional recognition, age identification, glasses detection and human detection. With regards to multiclass classification, a hair colour recognition task. Grid Search Cross-Validation has been utilised for parameter tuning.


Libraries:

import sys
import os
import numpy as np
import random
from collections import Counter
import sklearn


To run this code, use the command line:
python amls_project.py Task Number Algorithim
For example: python amls_project.py 1 svm

Task Options: 1, 2, 3, 4 & 5
Algorithm Options: svm & rf
