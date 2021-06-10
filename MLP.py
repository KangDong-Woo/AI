from sklearn import datasets
from sklearn.neural_network import MLPClassifier
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
    mlp = MLPClassifier(hidden_layer_sizes=(100),
                        learning_rate_init=0.001,
                        batch_size='auto',
                        solver='adam',
                        verbose=True)
    mlp.fit(x_train_std, y_train)
    
    res = mlp.predict(x_test_std)
    
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

from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
import pandas as pd

model_params  = {
    "svm" : {
        "model":SVC(gamma="auto"),
        "params":{
            'C' : [1,10,20],
            'kernel':["rbf"]
        }
    },
    
    "decision_tree":{
        "model": DecisionTreeClassifier(),
        "params":{
            'criterion':["entropy","gini"],
            "max_depth":[5,8,9]
        }
    },
    
    "random_forest":{
        "model": RandomForestClassifier(),
        "params":{
            "n_estimators":[1,5,10],
            "max_depth":[5,8,9]
        }
    },
    "naive_bayes":{
        "model": GaussianNB(),
        "params":{}
    },
    
    'logistic_regression' : {
        'model' : LogisticRegression(solver='liblinear',multi_class = 'auto'),
        'params': {
            "C" : [1,5,10]
        }
    }
    
}
score=[]
for model_name,mp in model_params.items():
    clf = GridSearchCV(mp["model"],mp["params"],cv=8,return_train_score=False)
    clf.fit(x_train_std,y_train)
    score.append({
        "Model" : model_name,
        "Best_Score": clf.best_score_,
        "Best_Params": clf.best_params_
    })
digit5 = pd.DataFrame(score,columns=["Model","Best_Score","Best_Params"])
print(digit5)