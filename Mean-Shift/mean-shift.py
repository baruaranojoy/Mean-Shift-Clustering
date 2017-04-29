import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
style.use('ggplot')
from sklearn.cluster import MeanShift
from sklearn import preprocessing
import pandas as pd



# reading the data from .XLS file in
# using pandas library
df = pd.read_excel('Dataset_1.xls')
original_df = pd.DataFrame.copy(df)


# function used to convert all data
# to neumaric type data
def handle_non_neumaric_data(df):
    columns = df.columns.values

    for column in columns:
        text_digit_vals = {}
        def convert_to_int(val):
            return text_digit_vals[val]

        if df[column].dtype != np.int64 and df[column].dtype != np.float64:
            column_contents = df[column].values.tolist()
            unique_elements = set(column_contents)
            x = 0
            for unique in unique_elements:
                if unique not in text_digit_vals:
                    text_digit_vals[unique] = x
                    x = x + 1

            df[column] = list(map(convert_to_int, df[column]))

    return df

# call to the function to convert data to
# neumaric type 
df = handle_non_neumaric_data(df)
original_df_1 = pd.DataFrame.copy(df)
#print df.head(149).to_string()


# drop the Class column because clasification
# will take place with respect to this column 
X = np.array(df.drop(['Class'], 1).astype(float))
# scaling the data for better result
X = preprocessing.scale(X)
# storing the Class column in Y for
# futute comparison
Y = np.array(df['Class'])
Z = Y
Y = list(set(Y))

# classifing using KMeans in 3 clusters
clf = MeanShift()
clf.fit(X)







labels = clf.labels_
cluster_centers = clf.cluster_centers_

original_df_1['cluster_group'] = np.nan

for i in range(len(X)):
    original_df_1['cluster_group'].iloc[i] = labels[i]

class_number = {}

n_clusters_ = len(np.unique(labels))

print "Total number of clusters are : ", n_clusters_
count = 0.0

for i in range(n_clusters_):
    temp_df = original_df_1[(original_df_1['cluster_group'] == float(i))]
    class_cluster = temp_df[(temp_df['Class'] == Y[i])]
    #if (temp_df['Class'] == Y[i]):
    #    count = count + 1.0
    class_rate = float(float(len(class_cluster))/float(len(temp_df)))
    class_number[i] = class_rate

print class_number
 






correct = 0.0

for i in range(len(X)):
    predict_me = np.array(X[i].astype(float))
    predict_me = predict_me.reshape(-1, len(predict_me))
    prediction = clf.predict(predict_me)

    if prediction[0] == Z[i]:
        correct = correct + 1.0

print 100.0*(correct/float(len(X))),"% accuracy based on mean-shifting clustering algorithmhm."






























