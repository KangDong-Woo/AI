from sklearn import datasets
from sklearn import svm
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.preprocessing import StandardScaler

stand = StandardScaler()
digit  = datasets.load_wine()
avg_accuracy = 0;

for i in range(0, 100): 
    x_train, x_test, y_train, y_test = train_test_split(digit.data, digit.target, train_size = 0.7)
    x_train_std = stand.fit_transform(x_train)
    x_test_std = stand.transform(x_test)
    
    s = svm.SVC(gamma = 0.001, C = 10, kernel = 'rbf')
    s.fit(x_train_std, y_train)
    
    res = s.predict(x_test_std)
    
    conf = np.zeros((10,10))
    for i in range(len(res)):
        conf[res[i]][y_test[i]] += 1
    print(conf)
    
    correct = 0
    for i in range(10):
        correct += conf[i][i]
    accuracy = correct / len(res)
    print("테스트 집합에 대한 정확률: ", accuracy*100, "%")
    
    avg_accuracy += accuracy

print("평균 정확률: %.1f"%((avg_accuracy / 100)*100))