#Librerias
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.datasets import make_moons
from mpl_toolkits.mplot3d import Axes3D

#Generacion de datos no lineales
x, y = make_moons(n_samples=200, noise=0.1, random_state=42)

#Entrenamiento del modelo SVM 3D

#Crearel modelo SVM no lineal
modelo = svm.SVC(kernel='rbf', C=1.0, gamma='scale')
#Entrenamiento del modelo SVM
modelo.fit(x,y)

#Grafica
plt.figure(figsize=(8,6))
plt.scatter(x[:,0], x[:,1], c=y, cmap='coolwarm', s=60, edgecolors='k')
plt.title('Clasificacion SVM con frontera y margenes')
plt.xlabel('x1')
plt.xlabel('x2')
plt.legend()
plt.show()

#Crear un grid

x_min, x_max = x[:,0].min() -1,x[:,0].max()+1
y_min, y_max = x[:,1].min() -1,x[:,1].max()+1
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                     np.linspace(y_min, y_max, 100))

#Calcular valores de decisiones
z = modelo.decision_function(np.c_[xx.ravel(), yy.ravel()])
z = z.reshape(xx.shape)

#Grafica en 3D

fig = plt.figure(figsize=(10,7))
ax = fig.add_subplot(111, projection='3d')

#Superficie del modelo 

ax.plot_surface(xx, yy, z, cmap= 'coolwarm', alpha=0.6)

#Puntos de datos

ax.scatter(x[:,0], modelo.decision_function(x),
           c=y, cmap= 'coolwarm', edgecolor='k', s=60)


ax.set_title('SVM no lineal')
ax.set_xlabel('x1')
ax.set_ylabel('x2')
ax.set_zlabel('Funcion de decision(z)')
plt.show()


