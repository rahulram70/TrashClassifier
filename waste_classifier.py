import os
import numpy as np
from fastai.vision import *
from fastai.metrics import error_rate
from shutil import copyfile
import webbrowser


def Img_Data_Bunch():
 path = Path(os.getcwd())/"data"
 tfms = get_transforms(do_flip=True, flip_vert=True)
 data = ImageDataBunch.from_folder(path, test="test", ds_tfms=tfms, bs=100)
# print(data.classes)
# data.show_batch(rows=4, figsize=(10,8), return_fig = True)


# fig.savefig("batch_img.jpg")
 return data


def load_ml(data):
 learn = cnn_learner(data, models.resnet34, metrics=error_rate).load('trained_finished_model')
# learn.export()
# learn = load_learner("/home/groups/Spellmandata/ram/UW_Hackathon/data/")
 return learn

def make_pred(learn):
 materials = ['cardboard','glass','metal','paper','plastic','trash']
 preds = learn.get_preds(ds_type=DatasetType.Test)
 max_idxs = np.asarray(np.argmax(preds[0], axis=1))
 yhat = []
 for max_idx in max_idxs:
  yhat.append(data.classes[max_idx])
 return yhat

def test_pred(data, learn):
 img = data.train_ds[0][0]
 print(learn.predict(img))

def check_file(pathway, learn, img_dir):
 while(1==1):
  while(len(os.listdir(pathway)) > 0):
   material_yhat = make_pred(learn)
   material = material_yhat[0]
   if("trash" in material):
    disposal = "trash.png"
    reward = "100.png"
   else:
    disposal = "recycle.png"
    reward = "200.png"
   disposal_pathway = img_dir + "imagesAuxiliary/disposal/" + disposal
   material_pathway = img_dir + "imagesAuxiliary/material/" + material + ".png"
   points_pathway = img_dir + "imagesAuxiliary/points/" + reward
   disposal_pathway_dest = img_dir + "ImagesResult/disposal.png"
   material_pathway_dest = img_dir + "ImagesResult/material.png"
   points_pathway_dest = img_dir + "ImagesResult/points.png"
   copyfile(disposal_pathway, disposal_pathway_dest)
   copyfile(material_pathway, material_pathway_dest)
   copyfile(points_pathway, points_pathway_dest)
   filename = img_dir + "result.html"
   webbrowser.open('file://' + os.path.realpath(filename))
#   os.remove(disposal_pathway_dest)
#   os.remove(material_pathway_dest)
#   os.remove(points_pathway_dest) 
if __name__ == "__main__":
 data = Img_Data_Bunch()
 learn = load_ml(data)
# yhat = make_pred(learn)
# test_pred(data, learn)
# print(yhat[0])
 path_test = Path(os.getcwd())/"data/test/"
 path_tia = Path(os.getcwd())/"TrashIdentification/tia/"
 print((str) path_tia)
# check_file(path_test, learn, path_tia)
