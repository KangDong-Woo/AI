from sklearn import datasets
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor

stand = StandardScaler()
digit  = datasets.load_wine()
avg_accuracy = 0;

for i in range(0, 100): 
    x_train, x_test, y_train, y_test = train_test_split(digit.data, digit.target, train_size = 0.7)
    x_train_std = stand.fit_transform(x_train)
    x_test_std = stand.transform(x_test)
    
    model = RandomForestRegressor(max_depth=9, min_samples_leaf= 8, min_samples_split=8, n_estimators=10) 
    clf = model.fit(x_train_std, y_train)

    train_score = clf.score(x_train_std, y_train)
    

    print("테스트 집합에 대한 정확률: ", train_score*100, "%")
    
    avg_accuracy += train_score

    print("평균 정확률: %.1f"%((avg_accuracy / 100)*100))

