import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler

#from google.colab import drive
#drive.mount('/content/drive')
import itertools
import threading
import time
import sys
def animate():
    for c in itertools.cycle(['|', '/', '-', '\\']):
        if done:
            break
        sys.stdout.write('\rloading ' + c)
        sys.stdout.flush()
        time.sleep(0.2)
    sys.stdout.write('\rDone!     ')

done = False
print("hi")

print("loading training dataset")
data_folder = "../ai-classificator/"
dataset = "spezia-section-01-secondotest.xyz"

print("working on csv data")
pcd = pd.read_csv(data_folder+dataset, delimiter=' ')
# del pcd['normal_z_-_normal_y']
pcd.dropna(inplace = True) # may lose some data but it is less then 1%
mask_ground = pcd['Classification'].values == 2
df_ground = pcd.loc[mask_ground]
df_ground_10000 = df_ground.head(250000)
mask_veggie = pcd['Classification'].values == 5
df_veggie = pcd.loc[mask_veggie]
df_veggie_10000 = df_veggie.head(250000)
mask_build = pcd['Classification'].values == 6
df_build = pcd.loc[mask_build]
df_build_10000 = df_build.head(250000)
mask_road = pcd['Classification'].values == 11
df_road = pcd.loc[mask_road]
df_road_10000 = df_road.head(250000)
pcd = pd.concat([df_ground_10000, df_veggie_10000, df_build_10000, df_road_10000 ])


print("create training and testing")
labels = pcd['Classification']
features = pcd[['R','G','B','linearity','planarity','sphericity']]
features_scaled = MinMaxScaler().fit_transform(features)

# split the training sample in two parts: 60% training, 40% testing on the same sample
X_train,X_test,y_train,y_test = train_test_split(features_scaled,labels,test_size=0.4)

print("training phase")
t = threading.Thread(target=animate)
t.start()
from sklearn.feature_selection import SelectFromModel
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline

# LinearSVC(dual=False)
'''clf = Pipeline([
  ('feature_selection', SelectFromModel(LinearSVC(C=0.01, penalty="l1", dual=False))),
  ('classification', RandomForestClassifier())
])
clf.fit(X_train, y_train)'''
# scaling unnecessary for random forests.
rf_classifier = RandomForestClassifier(warm_start=True, n_estimators = 10)

start = time.perf_counter()
# create a classification model
rf_classifier.fit(X_train, y_train)
end = time.perf_counter()
print(f"Downloaded the tutorial in {end - start:0.4f} seconds")

# test on an unseen dataset
rf_predictions = rf_classifier.predict(X_test)

done = True
print(classification_report(y_test, rf_predictions, target_names=['ground','vegetation','buildings','road surface']))


"""print("plotting 3d result")
fig, axs = plt.subplots(1, 3, figsize=(20,5))
axs[0].scatter(X_test[:,0], X_test[:,1], c =y_test, s=0.05)
axs[0].set_title('Ground Truth')
axs[1].scatter(X_test[:,0], X_test[:,1], c = rf_predictions, s=0.05)
axs[1].set_title('Previsioni')
axs[2].scatter(X_test[:,0], X_test[:,1], c = y_test-rf_predictions, cmap = plt.cm.rainbow, s=0.5*(y_test-rf_predictions))
axs[2].set_title('Differenze')
plt.show()"""

print("working on testing csv")
val_dataset="spezia-section-02-secondotest.xyz"
print("reading testing CSV")
val_pcd=pd.read_csv(data_folder+val_dataset,delimiter=' ')
# del val_pcd['normal_z_-_normal_y']
val_pcd.dropna(inplace=True)
val_labels=val_pcd['Classification']
val_features=val_pcd[['R','G','B','linearity','planarity','sphericity']]
val_features_scaled = MinMaxScaler().fit_transform(val_features)

print("testing phase")
rf_classifier.n_estimators+=1
val_predictions = rf_classifier.predict(val_features_scaled)
print(classification_report(val_labels, val_predictions, target_names=['ground','vegetation','buildings','road surface']))

#print("plotting 3d result")
#fig, axs = plt.subplots(1, 3, figsize=(20,5))
#axs[0].scatter(val_features['X'], val_features['Y'], c =val_labels, s=0.05)
#axs[0].set_title('Ground Truth')
#axs[1].scatter(val_features['X'], val_features['Y'], c = val_predictions, s=0.05)
#axs[1].set_title('Previsioni')
#axs[2].scatter(val_features['X'], val_features['Y'], c = val_labels-val_predictions, cmap = plt.cm.rainbow, s=0.5*(val_labels-val_predictions))
#axs[2].set_title('Differenze')
#plt.show()

print("saving results in a csv file")
val_pcd['predictions']=val_predictions
result_folder="../ai-classificator/result/"
val_pcd[['X','Y','Z','R','G','B','predictions']].to_csv(result_folder+val_dataset.split(".")[0]+"_result_final.xyz", index=None, sep=';')

done = True

import pickle
pickle.dump(rf_classifier, open(result_folder+"model_v101.poux", 'wb'))
