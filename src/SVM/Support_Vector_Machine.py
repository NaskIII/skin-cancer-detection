from sklearn import svm
import numpy
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from joblib import dump, load

numpy.random.seed(2324)

dataset = numpy.loadtxt(
  r"C:\Users\Raphael Nascimento\PycharmProjects\skin-melanoma-detector\src\Data\Dataset_PH2_Normalizado.csv", delimiter=",")
indices = numpy.random.choice(dataset.shape[0], 165, replace=False)
dataset2 = dataset[indices]
dataset = numpy.delete(dataset, indices, axis=0)

X = dataset[:,1:10]
Y = dataset[:,0]
Xtest = dataset2[:,1:10]
Ytest = dataset2[:,0]
# end - do not change

clf = svm.SVC(kernel='rbf', tol=1e-9, C=5, probability=True)
clf.fit(X, Y)

predictions = clf.predict(Xtest)
predictions2 = clf.decision_function(Xtest)

false_positive_rate1, true_positive_rate1, threshold1 = roc_curve(Ytest, predictions2)
auc_SVM = auc(false_positive_rate1, true_positive_rate1)

plt.figure(1)
plt.plot([0, 1], [0, 1], 'k--')
plt.plot(false_positive_rate1, true_positive_rate1, label='SVM - ISIC (area = {:.3f})'.format(auc_SVM))
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.title('ROC curve')
plt.legend(loc='best')
plt.show()

correct = 0
incorrect = 0
totalSpec = 0
totalSensi = 0
TP = 0
FP = 0
TN = 0
FN = 0

for i in Ytest:
  if i == 0:
    totalSpec += 1
  else:
    totalSensi += 1

for i in range(len(Xtest)):
  if predictions[i] == Ytest[i]:
    correct += 1
    if predictions[i] == 0:
      TN += 1
    elif predictions[i] == 1:
      TP += 1
  else:
    incorrect +=1
    if predictions[i] == 0:
      FN += 1
    elif predictions[i] == 1:
      FP += 1

print('''Corrects: %s; 
Incorrects: %s; 
Specificity: %s; 
Sensitivity: %s;
''' %
(correct , incorrect, TN, TP))

print('\nTotal TN: %s; Total TP: %s;' % (totalSpec, totalSensi))

print('Specificity: %s' % ((TN / (TN + FP)) * 100))
print('Sensitivity: %s' % ((TP / (TP + FN)) * 100))
print('Precision: %s' % ((TP / (TP + FP)) * 100))
print('Accuracy: %s' % (((TP + TN) / (TP + TN + FP + FN)) * 100))

dump(clf, '../../SVM/Modelo_ISIC/Modelo_ISIC.joblib')