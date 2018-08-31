import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics,datasets,svm
import matplotlib.image as mimg
from sklearn.externals import joblib

'''path = './database/iris/U1/1.jpg'
v = mimg.imread(path)
print(v.shape)
v1 = v.reshape(1,-1)
print(v1.shape)'''

train_data = np.zeros((160,30000))
train_target = np.zeros((160))

test_data = np.zeros((40,30000))
test_target = np.zeros((40))

c =0
for i in range(1,41):
    for j in range(1,5):
        path = ('C:/Users/rajasharma/Desktop/aedifico/supervised learning/iris/U%d/%d.jpg'%(i,j))
        v= mimg.imread(path)
        feat = v.reshape(1,-1)
        train_data[c,:] = feat
        train_target[c] = i
        c+=1
c= 0        
for i in range(1,41):
    for j in range(5,6):
        path = ('C:/Users/rajasharma/Desktop/aedifico/supervised learning/iris/U%d/%d.jpg'%(i,j))
        v= mimg.imread(path)
        feat = v.reshape(1,-1)
        test_data[c,:] = feat
        test_target[c] = i
        c+=1
        
svm_model = svm.SVC(kernel = 'poly')
svm_model = svm_model.fit(train_data,train_target)
output = svm_model.predict(test_data)

acc = metrics.accuracy_score(test_target,output)

conf_mat = metrics.confusion_matrix(test_target,output)
print("accuracy: ",acc*100)
print("confusion matrix:")
print(conf_mat)
report=metrics.classification_report(test_target,output)
print(report)

joblib.dump(svm_model,'train_svm_model.pkl')