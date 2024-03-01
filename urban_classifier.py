import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
from mpl_toolkits import mplot3d

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler

val_dataset="spezia_finale_livelli.xyz"    # <-------- DATASET NAME HERE
data_folder = "../ai_v_101/"
print("loading csv")
val_pcd=pd.read_csv(data_folder+val_dataset,delimiter=' ')
print(val_pcd)

#val_pcd.drop(['normal_z', 'normal_y','normal_x'], axis=1, inplace=True)
val_pcd.dropna(inplace = True)
print(val_pcd)
print("preparing features and labels")
val_labels=val_pcd['Classification']
val_features=val_pcd[['X','Y','Z','R','G','B','Intensity']]
val_features_scaled = MinMaxScaler().fit_transform(val_features)
print("loading model")
model_name="model_v101.poux"    # <-------- MODEL NAME HERE
result_folder="../ai_v_101/result/"
loaded_model = pickle.load(open(result_folder+model_name, 'rb'))
print("testing model")
loaded_predictions = loaded_model.predict(val_features_scaled)    # <--------- CLOUD POINT HERE
print(classification_report(val_labels, loaded_predictions, target_names=['ground','vegetation','buildings']))
print("done")

print("più bisogna modificare il dataset d'addestramento")

