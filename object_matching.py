import glob
import os
import cv2
import numpy as np
import pandas as pd
import warnings
import math
from sklearn.cluster import KMeans
from sklearn.metrics import mean_squared_error
from scipy.spatial.distance import cdist
from sklearn.decomposition import PCA
from sklearn.neighbors.kde import KernelDensity
from sklearn import metrics
from yellowbrick.cluster import KElbowVisualizer



np.random.seed(1)

###This is the location of the image files
####You can place this python file at the location where the images are present. In that case, please comment the location command below
###If your image files and python files are in different location, then please mention the corresponding location in the location command below
#os.chdir('D:\\blue_motion\\test0_cam1\\test0_cam1')
os.chdir('D:\\blue_motion\\test0_cam2\\test0_cam2')

####Reading the image files and binding them together
con = []
name = []
dim = (100, 100)
print("Reading and Binding the images")
warnings.filterwarnings("ignore")
for file in glob.glob('*.png'):
        name.append(file)
        img = None
        img = cv2.imread(file, 0)
        resized = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
        ####Flattening the image matrix
        flat = resized.flatten()
        con.append(flat)
        
cam1 = np.asarray(con)

####Normalzation 
cam1 = cam1/255
print("Input size before PCA")
print(cam1.shape)

#####Performing Principal Component Analysis (PCA) and reducing the number of features########
org = cam1
print("Performing PCA")
pca = PCA(n_components = 10)
cam1 = pca.fit_transform(cam1)
print("Input size after PCA")
print(cam1.shape)

min_cluster_val = 2
max_cluster_val = 100
er = []
diff= []

print("Determining the Cluster size")
###### I have used two methods to determine the cluster size. I have commented one of them. Please uncomment if required
######Using Calinski Harabaz Technique to determine the number of clusters requred#####
######This technique is relatively faster and much accurate for this dataset######
######Takes about 2 minutes to complete


model = KMeans()
visualizer = KElbowVisualizer(model, k=(min_cluster_val,max_cluster_val), metric='calinski_harabaz')
df = visualizer.fit(cam1)
score = df.k_scores_

for i in range(len(score)-1):
        diff.append(abs(score[i] - score[i+1]))
        

######Using Elbow Technique to determine the number of clusters requred#####
######This technique is a slow and less accurate for this dataset######
###### I have commented it out but you are welcome to uncomment it and try it out but comment the previous portion of calinski harabaz before uncommenting this one
###### Takes about 4 minutes to complete
'''
while min_cluster_val <= max_cluster_val:
    kmeans = KMeans(n_clusters=min_cluster_val, random_state=0).fit(cam1)
    h = kmeans.inertia_
    out = h
    print(h)
    er.append(h)
    if min_cluster_val > 2:
        diff.append(abs((out - prev_out)/10000000))
    min_cluster_val = min_cluster_val+1
    prev_out = out

'''


diff1 = []
for i in range(len(diff)-1):
    diff1.append(((diff[i] - diff[i+1])))

#####Calculating the strength value from the obtained scores
strength = []
for i in range(len(diff1)):
    st = diff1[i] - diff[i+1];
    strength.append(st)
cluster_no = strength.index(max(strength))+2
print("Approximate cluster size is:",cluster_no)
######Now that we know the number of clusters, we can perform k-means specifying that number of clusters
print("Clustering the images into various clusters")
kmeans = KMeans(n_clusters=cluster_no, random_state=0).fit(cam1)
out_label = kmeans.labels_
#######Copying the values to a dataframe
p = pd.DataFrame()
p['Image name'] = name
p['Person ID'] = out_label
########Writing it in a csv file
########The csv file will be written in the folder that you have selected at the beginning of the code or in your current folder.
####The name of the csv is file.csv
print("Creating the CSV file")
p.to_csv("file.csv")
print("Done!!!")


