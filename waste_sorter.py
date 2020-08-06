from fastai.vision import *
from fastai.metrics import error_rate
from pathlib import Path
from glob2 import glob
from sklearn.metrics import confusion_matrix

import pandas as pd
import numpy as np
import os
import zipfile as zf
import shutil
import re
import seaborn as sns
import shutil

def sort_pics(dir_pathway):

 for material in os.listdir(dir_pathway):
  full_pathway = dir_pathway + material + "/"
  total_file_num = len(os.listdir(full_pathway))
  train_num = total_file_num // 2
  train_rem = total_file_num % 2
  veri_num = (train_num + train_rem) // 2
  veri_rem = (train_num + train_rem) % 2
  test_num = veri_num + veri_rem
#  print(str(train_num) + " " + str(veri_num) + " " + str(test_num)
  copy_pathway = "/home/groups/Spellmandata/ram/UW_Hackathon/train/" + material + "/"
  if("glass" not in material):
   for i in range(train_num):
    filename = random.choice(os.listdir(full_pathway))
    file_pathway = full_pathway + filename
    shutil.move(file_pathway, copy_pathway)

  copy_pathway = "/home/groups/Spellmandata/ram/UW_Hackathon/validation/" + material + "/"

  if("glass" not in material):
   for i in range(veri_num):
    filename = random.choice(os.listdir(full_pathway))
    file_pathway = full_pathway + filename
    shutil.move(file_pathway, copy_pathway)
  
  copy_pathway = "/home/groups/Spellmandata/ram/UW_Hackathon/test/"
  if("glass" not in material):
   for i in range(test_num):
    filename = random.choice(os.listdir(full_pathway))
    file_pathway = full_pathway + filename
    shutil.move(file_pathway, copy_pathway)
 
def Img_Data_Bunch():
 path = Path(os.getcwd())/"data"
 tfms = get_transforms(do_flip=True, flip_vert=True)
 data = ImageDataBunch.from_folder(path, test="test", ds_tfms=tfms, bs=100)
# print(data.classes)
# data.show_batch(rows=4, figsize=(10,8), return_fig = True)


# fig.savefig("batch_img.jpg")
 return data

def ml_call(data):
 learn = cnn_learner(data, models.resnet34, metrics=error_rate)
# learn.lr_find(start_lr=1e-6, end_lr=1e1)
 #lr_plot = learn.recorder.plot(return_fig = True)
 #lr_plot.savefig("learning_rate.jpg")
 #learn.fit_one_cycle(20, max_lr=5.13e-3) 
 model_pathway = learn.save("test_model", return_path = True)
 learn.export()
 print(model_pathway)
 return learn

def inc_img(learn):
 interp = ClassificationInterpretation.from_learner(learn)
 losses, indxs = interp.top_losses()
 inc_plots = interp.plot_top_losses(9, figsize=(15, 11), return_fig = True)
 inc_plots.savefig("incorrect_images.jpg")
 doc(interp.plot_top_losses)
 error_matrix = interp.plot_confusion_matrix(figsize = (12,12), dpi=60, return_fig = True)
 error_matrix.savefig("error_matrix.jpg") 

def make_pred(data, learn):
 preds = learn.get_preds(ds_type=DatasetType.Test)
 max_idxs = np.asarray(np.argmax(preds[0], axis=1))
 yhat = []
 for max_idx in max_idxs:
  yhat.append(data.classes[max_idx])
 return yhat

def acc_check(data, yhat):
 waste_types = ['cardboard','glass','metal','paper','plastic','trash']
 y = []
 for label in data.test_ds.items:
  y.append(str(label))
 pattern = re.compile("([a-z]+)[0-9]+")
 for i in range(len(y)):
  y[i] = pattern.search(y[i]).group(1)
 print(yhat[0:15])
 print(y[0:15])
 cm = confusion_matrix(y, yhat)
 df_cm = pd.DataFrame(cm, waste_types, waste_types)
 plt.figure(figsize=(10,8))
 sns_plot = sns.heatmap(df_cm, annot=True, fmt="d", cmap="YlGnBu")
 sns_plot.savefig("matrix_heat_map.jpg")
 correct = 0
 for r in range(len(cm)):
  for c in range(len(cm)):
   if(r == c):
    correct += cm[r, c]
 accuracy = correct / sum(sum(cm)) 
 print(accuracy)

if __name__ == "__main__":
# sort_pics("/home/groups/Spellmandata/ram/UW_Hackathon/dataset-resized/")
 data = Img_Data_Bunch()
 learn = ml_call(data)
# inc_img(learn)
# yhat = make_pred(data, learn)
# acc_check(data, yhat)

# learn = ml_call(data)
# inc_img(learn)
 
