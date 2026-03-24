#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

import numpy
import pandas
import matplotlib as mpl
from matplotlib import pyplot as plt
from scipy import stats as sps
from scipy.optimize import curve_fit
import seaborn as sns
from termcolor import colored
from typing import Union

#### Init
mpl.use("webagg")
sns.set_theme(style="whitegrid", palette="Set2")
numpy_rng = numpy.random.default_rng()
