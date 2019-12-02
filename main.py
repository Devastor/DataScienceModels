
import numpy as np
from pandas import read_csv
import random
import csv
from sklearn.model_selection import train_test_split
import time

def csv_writer(data, path):
    """
    Функция для записи данных в CSV
    """
    with open(path, "w", newline='') as csv_file:
        """
        csv_file - объект с данными
        delimiter - разделитель
        """
        writer = csv.writer(csv_file, delimiter=';')
        for line in data:
            writer.writerow(line)

def generateData(N):
    """
    Случайным образом генерирует данные по 4 параметрам.
    5-й ключевой параметр выбирается при помощи дерева решений
    """
    data = [['age','education','salary', 'house','giveCredit',]]
    educ = ['среднее', 'среднее-специальное', 'высшее']

    for i in range(N):
        tempData = []

        age = int(random.randrange(18, 60))
        education = random.choice(educ)
        salary = int(random.randrange(10000, 300000))
        house = random.randrange(0, 2)

        # дерево решений для генерации значений поля giveCredit
        if salary < 40000:
            giveCredit = False
        elif salary > 270000:
            giveCredit = True
        else:
            if education != 'высшее':
                if house == 0:
                    if age < 30:
                        giveCredit = False
                    elif salary > 120000:
                        giveCredit = True
                    else:
                        giveCredit = False
                elif salary < 60000:
                    giveCredit = False
                else:
                    giveCredit = True
            elif salary > 100000:
                giveCredit = True
            else:
                if house == 0:
                    if age < 40:
                        giveCredit = False
                    elif salary > 85000:
                        giveCredit = True
                    else:
                        giveCredit = False
                elif age > 35:
                    if salary > 80000:
                        giveCredit = True
                    else:
                        giveCredit = False
                else:
                    giveCredit = False

        tempData.append(age)
        tempData.append(education)
        tempData.append(salary)
        tempData.append(house)
        tempData.append(giveCredit)

        data.append(tempData)
    return data

def writeToCSV(data, path):
    csv_writer(data, path)

path = "credit.csv"

data = generateData(int(input("Введите размер генерируемой выборки:(20-20000)")))
writeToCSV(data, "credit.csv")

csv_file = open(path, "r")

dataset = read_csv(csv_file, ';')
print(dataset['giveCredit'].describe())

# замена данных
level_map = {'среднее':0, 'среднее-специальное':1, 'высшее':2}
dataset['education'] = dataset['education'].map(level_map)

X = dataset.iloc[:,:-1].values
y = dataset['giveCredit']

# создаем тестовую и тренировочную выборки
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=27)

from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

start_time = time.time()

SVC_model = SVC(gamma='scale')
SVC_model.fit(X_train, y_train)
SVC_prediction = SVC_model.predict(X_test)

print("SVC:")
print(accuracy_score(SVC_prediction, y_test))
print(confusion_matrix(SVC_prediction, y_test))

print("--- %.3f seconds ---" % (time.time() - start_time))
start_time = time.time()

KNN_model = KNeighborsClassifier(n_neighbors=5)
KNN_model.fit(X_train, y_train)
KNN_prediction = KNN_model.predict(X_test)

print("KNeighborsClassifier:")
print(accuracy_score(KNN_prediction, y_test))
print(confusion_matrix(KNN_prediction, y_test))
print("--- %.3f seconds ---" % (time.time() - start_time))
start_time = time.time()

GBC_model = GradientBoostingClassifier()
GBC_model.fit(X_train, y_train)
GBD_prediction = GBC_model.predict(X_test)

print("GradientBoostingClassifier:")
print(accuracy_score(GBD_prediction, y_test))
print(confusion_matrix(GBD_prediction, y_test))
print("--- %.3f seconds ---" % (time.time() - start_time))
start_time = time.time()

RFC_model = RandomForestClassifier(n_estimators=100)
RFC_model.fit(X_train, y_train)
RFC_prediction = RFC_model.predict(X_test)

print("RandomForestClassifier:")
print(accuracy_score(RFC_prediction, y_test))
print(confusion_matrix(RFC_prediction, y_test))
print("--- %.3f seconds ---" % (time.time() - start_time))
start_time = time.time()