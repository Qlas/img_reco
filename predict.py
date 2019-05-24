# --------------------------------------------------------------------------
# ------------  Metody Systemowe i Decyzyjne w Informatyce  ----------------
# --------------------------------------------------------------------------
#  Zadanie 4: Zadanie zaliczeniowe
#  autorzy: A. Gonczarek, J. Kaczmar, S. Zareba, M Zieba
#  2019
# --------------------------------------------------------------------------

import pickle as pkl
import numpy as np
import time
def predict(x):
    """
    Funkcja pobiera macierz przykladow zapisanych w macierzy X o wymiarach NxD i zwraca wektor y o wymiarach Nx1,
    gdzie kazdy element jest z zakresu {0, ..., 9} i oznacza znak rozpoznany na danym przykladzie.
    :param x: macierz o wymiarach NxD
    :return: wektor o wymiarach Nx1
    """
    file_train = open('trains.pkl', "rb")
    train = pkl.load(file_train)
    y = []
    k = 5
    x_train = train[0]
    y_train = train[1]
    for q in range(100):
        distance = []
        for i in range(800):
            distance.append(np.linalg.norm(x[q] - x_train[i]))

            # distance.append(np.sqrt(sum((x[q] - x_train[i]) ** 2)))
        # u = (x[0] - x_train) ** 2
        # print(distance)
        # distance = np.sqrt([sum(b) for b in u])
        # print(distance)
        minarg = np.argsort(distance)
        i = np.array(np.zeros(10))
        j = 0
        while k not in i:
            i[y_train[minarg[j]]] += 1
            j += 1
        y.append(np.argmax(i))
    return y




file = open("train.pkl", "rb")
x = pkl.load(file)
start = time.time()
y = predict((x[0]))
end = time.time()
print(end - start)
check = 0
print("done")
for i in range(len(y)):
    if y[i] == x[1][i]:
        check += 1
print(check)
print(check/len(y) *100)
