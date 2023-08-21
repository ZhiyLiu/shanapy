import os
import numpy as np
from shanapy.models.stats import Classification, SrepFeatures

# distribution of initial s-reps features
root = '/path/to/population/folder/'
os.chdir(root)
class_names = ["pos", "neg"]
labels = np.array([1] * 34 + [0] * 143)
all_feats = []
for class_name in class_names:
    directory = class_name + "_initial_sreps"
    feats_a_group = []
    for file_name in os.listdir(directory):
        if file_name.split('.')[-1] != 'vtk': continue

        skeletal_pts, dirs, radii = SrepFeatures.default_srep_features(root + directory + "/" + file_name)
        # skeletal_pts, dirs, radii = SrepFeatures.euclideanized_srep_features(root + directory + "/" + file_name)
        feats = np.concatenate((skeletal_pts, dirs, radii)).flatten()
        feats_a_group.append(feats)
    feats_a_group = np.array(feats_a_group).T # d x n
    all_feats.append(feats_a_group)
all_feats = np.concatenate((all_feats[0], all_feats[1]), axis=1).T # n x d
print(Classification.classify(all_feats, labels))
