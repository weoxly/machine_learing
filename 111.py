import warnings
warnings.filterwarnings('ignore')
import warnings
warnings.filterwarnings("ignore", module="lightgbm")
import itertools
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import lightgbm as lgbm

pd.set_option('display.max_colwidth',None)

items = pd.read_csv('items.csv')
item_cat = pd.read_csv('item_categories.csv')
shops = pd.read_csv('shops.csv')
train = pd.read_csv('sales_train.csv')
test_dataset = pd.read_csv('test.csv')

a=train.shape
b=test_dataset.shape

print(a,"  ",b)