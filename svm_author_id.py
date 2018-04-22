
import sys
from time import time
sys.path.append("../tools/")
import matplotlib.pyplot as plt
from email_preprocess import preprocess
from class_vis import prettyPicture


### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()


from sklearn.svm import SVC

    ### create classifier
clf = SVC(kernel="poly")
t0=time()
clf.fit(features_train[:len(features_train)//100], labels_train[:len(labels_train)//100])
SVC(C=1000.0)

print ('Training time', round(time()-t0, 3), "s")


    ### use the trained classifier to predict labels for the test features
t1 = time()
pred = clf.predict(features_test)
print ('Predict time:', round(time()-t1, 3), "s")
    ### calculate and return the accuracy on the test data
    ### this is slightly different than the example, 
    ### where we just print the accuracy
    ### you might need to import an sklearn module
from sklearn.metrics import accuracy_score 
accuracy= accuracy_score(pred,labels_test)
print('Accuracy is:',accuracy)
grade_fast = [features_train[ii][0] for ii in range(0, len(features_train)) if labels_train[ii]==0]
bumpy_fast = [features_train[ii][1] for ii in range(0, len(features_train)) if labels_train[ii]==0]
grade_slow = [features_train[ii][0] for ii in range(0, len(features_train)) if labels_train[ii]==1]
bumpy_slow = [features_train[ii][1] for ii in range(0, len(features_train)) if labels_train[ii]==1]


#### initial visualization
plt.xlim(0.0, 1.0)
plt.ylim(0.0, 1.0)
plt.scatter(bumpy_fast, grade_fast, color = "b", label="fast")
plt.scatter(grade_slow, bumpy_slow, color = "r", label="slow")
plt.legend()
plt.xlabel("bumpiness")
plt.ylabel("grade")
plt.show()


try:
    prettyPicture(clf, features_test, labels_test)
except NameError:
    pass




#########################################################


