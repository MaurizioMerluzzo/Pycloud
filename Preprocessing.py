import numpy as np
import pandas as pd
from sklearn import preprocessing
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms
from ellipsoid_calculator import EllipsoidTool
import time

def numpy_euclidean_distance_short(point_1,point_2):
    return np.sqrt(np.sum(np.square(np.array(point_1) - np.array(point_2))))

def point_in_range(i, j, distance):
    if numpy_euclidean_distance_short(np_points[i, 0:2], np_points[j, 0:2]) <= distance:
        # mark point as visited and append to an ellipsoid list
        np_points[j, 11] = 1  # mark as visited
        index = int(np_points[j, 15])
        np_pcd[index, 11] = 1
        ellipsoid.append(np_points[j, np.r_[0:6]])  # append point to ellipsoid array

print("hi")
print("loading training dataset")
data_folder = "../ai-classificator/"
dataset = "spezia-section-01-primotest.xyz"

#### == CUBES PREPARATION == ####
print("working on csv data")
pcd = pd.read_csv(data_folder+dataset, delimiter=' ')
scaler = preprocessing.MinMaxScaler()

print("shifting array")
selection = pcd[['X', 'Y', 'Z']]
np_pcd = selection.to_numpy()
min_index = np.argmin(np_pcd, axis = 0)
min_values = np.amin(np_pcd, axis = 0)
# shifting
pcd['X'] = pcd['X'].subtract(min_values[0])
pcd['Y'] = pcd['Y'].subtract(min_values[1])
pcd['Z'] = pcd['Z'].subtract(min_values[2])

l = 1 # <------- insert here cubes dimension
row = 0
rows = np_pcd.shape[0]

# constructing cube of 1 m^3
x_max = np.amax(np_pcd[:,0], axis = 0)
y_max = np.amax(np_pcd[:,1], axis = 0)
z_max = np.amax(np_pcd[:,2], axis = 0)
Z_MAX = int(z_max)
Y_MAX = int(y_max)
X_MAX = int(x_max)

pcd['cube_x'] = pcd['X']*l
pcd['cube_y'] = pcd['Y']*l
pcd['cube_z'] = pcd['Z']*l
pcd = pcd.astype({"cube_x": np.int16, "cube_y": np.int16, "cube_z": np.int16})
pcd['visited'] = 0 # flag for visited point
pcd['linearity'] = 0 # 1D estimator
pcd['planarity'] = 0 # 2D estimator
pcd['sphericity'] = 0 # 3D estimator
pcd['omnivariance'] = 0 # omnivariance estimator
#### == EIGENVALUES COMPUTING == ####
max_x_cube = pcd['cube_x'].max()
max_y_cube = pcd['cube_y'].max()
max_z_cube = pcd['cube_z'].max()
selection = pcd[['X','Y','Z','R','G','B','Intensity','Classification','cube_x','cube_y','cube_z','visited','linearity','planarity','sphericity']]
np_pcd = selection.to_numpy()
# constructs an index column, help to set the flag visited to the right row
indexes = [np.arange(0, rows, 1)]
np_pcd = np.insert(np_pcd, 15, indexes, axis = 1)
header = {'X':[],'Y':[],'Z':[],'R':[],'G':[],'B':[],'Intensity':[],'Classification':[],'cube_X':[],'cube_Y':[],'cube_Z':[],'visited':[],'linearity':[],'planarity':[],'sphericity':[],'idx':[]}
final = pd.DataFrame(header)
x__ = 0
y__ = 0
z__ = 0
# the subarray takes only the points whose are from cube (x__,y__,z__) and its neighbors
# cycle every cube starting from (0,0,0)
start = time.process_time()   
while z__<max_z_cube:
    colZm = (np_pcd[:,10]>=z__-2) & (np_pcd[:,10]<=z__+2)
    while y__<max_y_cube:
        colYm = (np_pcd[:,9]>=y__-2) & (np_pcd[:,9]<=y__+2)
        while x__<max_x_cube:
            colXm = (np_pcd[:,8]>=x__-2) & (np_pcd[:,8]<=x__+2)
            np_points = np_pcd[colXm & colYm & colZm]
            ellipsoid = []
            # the ellipsoid is fixed to be of range 0.2 this operation will be done until there are no point left in the points array
            points_shape = np_points.shape[0]
            # for every points in the cube...
            for i in range(points_shape):
                # ...if point is in cube x,y,z and has not been visited yet...
                if np_points[i,8] == x__ and np_points[i,9] == y__ and np_points[i,10] == z__ and np_points[i,11] == 0:
                    # ...check every points and find which ones are closer then 0.2 from source point (self does count)
                    for j in range(points_shape):
                        point_in_range(i, j, 1.25)
                    ellipsoid = np.asanyarray(ellipsoid)
                    
                    if ellipsoid.size != 0:
                        # eigenvalues obtained through 3x3 covariance matrix
                        data = np.array([ellipsoid[:,0],ellipsoid[:,1],ellipsoid[:,2]])
                        cov_matrix = np.cov(data, bias=True)
                        w, v = np.linalg.eig(cov_matrix)
                        
                        #### == PLOTTING CONFIDENCE ELLIPSOID == ####
                        
                        # selection plotting
                        # singular matrix are not considered
                        if np.linalg.det(cov_matrix) != 0:
                            
                            # eigenvalues
                            l1 = w[0]
                            i__ = 0
                            while(i__<3):
                                if w[i__] > l1:
                                    l1 = w[i__]
                                i__ = i__ + 1

                            l3 = w[0]
                            j__ = 0
                            while (j__ < 3):
                                if w[j__] < l3:
                                    l3 = w[j__]
                                j__ = j__ + 1

                            l2 = w[0]
                            if w[0] == l1 or w[0] == l3:
                                l2 = w[1]
                                if w[1] == l1 or w[1] == l3:
                                    l2 = w[2]
                            
                            # estimator testing
                            linearity = (l1-l2)/l1
                            planarity = (l2-l3)/l1
                            sphericity = l3/l1

                            index = int(np_points[i,15])
                            #print(index)
                            np_pcd[index,12] = linearity
                            np_pcd[index, 13] = planarity
                            np_pcd[index, 14] = sphericity
                            #arr = np.array([x_mean,y_mean,z_mean,int(r_mean),int(g_mean),int(b_mean),estimator_1D,estimator_2D,estimator_3D])
                            arr = np_pcd[index]
                            #print(arr)
                            final = pd.concat([final, pd.DataFrame(arr.reshape(1,-1), columns = list(final))])
                            #print(final)
                            
                            #ellissoide = EllipsoidTool()
                            #center, radii, rotation = ellissoide.getMinVolEllipse(ellipsoid, 0.1)

                            #fig = plt.figure()
                            #ax = fig.add_subplot(111, projection = '3d')
                            #ax.scatter(x_plot, y_plot, z_plot, color = colore, s = 1)

                            # ellissoide.plotEllipsoid(center, radii, rotation, ax = ax, plotAxes = True)

                            # plt.show()
                        
                        ellipsoid = []
            print('[',x__,y__,z__,']')
            x__ = x__ + 1
        x__ = 1
        y__ = y__ + 1
    x__ = 1
    y__ = 1
    z__ = z__ + 1
print(time.process_time() - start)
final.to_csv('final.xyz')
print("Done!")

