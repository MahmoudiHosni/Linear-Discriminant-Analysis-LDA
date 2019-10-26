
from sklearn import svm
import pandas as pd
from pydataset import data
from sklearn import decomposition
import numpy as np
from sklearn import svm
from  sklearn import multiclass
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import confusion_matrix
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from PIL import Image 
import os
Datam=[]
dirs = os.listdir( "/home/hosni/Downloads/YALE/faces/" )
taille=len(dirs)
i=0
for ligne in dirs:
    if(os.path.splitext(ligne)[1] != '.pgm'):
        i+=1
        if(i<taille):
            im = Image.open("/home/hosni/Downloads/YALE/faces/"+ligne)
            data = list(im.getdata())
            trans=np.transpose(data)
            Datam.append(trans)
y=[]
j=0
for i in range(0,15):
   for i1 in range (0,11):
       y.append(j)
   j=j+1
lda = LinearDiscriminantAnalysis(n_components = 14)
x_lda=lda.fit_transform(Datam,y)
x_train,x_test,y_train,y_test = train_test_split(x_lda,y,test_size=0.33)
clf=svm.SVC(kernel='linear', C=1,decision_function_shape='OVR')
clf.fit(x_train,y_train)
predicted=clf.predict(x_test)
accauracy = accuracy_score(y_test,predicted)
print("acc",accauracy)
