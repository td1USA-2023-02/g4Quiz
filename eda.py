import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris

#PUNTO2
iris = load_iris()
df = pd.DataFrame(data= np.c_[iris['data'], iris['target']], columns= iris['feature_names'] + ['target'])
df['nueva_columna'] = df.apply(lambda row: 'margarita' if (row['sepal length (cm)'] >= 3.1 and row['sepal width (cm)'] >= 3.5 and row['petal length (cm)'] >= 1.3 and row['petal width (cm)'] <= 0.2) else 'no margarita', axis=1)

print(df.head())
# Mostrar las primeras filas del dataframe
print(df.head())

# Obtener información general sobre el dataframe
print(df.info())

# Resumen estadístico de las variables numéricas
print(df.describe())

# Verificar si hay valores nulos en el conjunto de datos
print(df.isnull().sum())


#PUNTO 3 CARGAR LOS DATOS A UN NUEVO ARCHIVO .CSV


df.to_csv('nombre_del_archivo.csv', index=False)

# Histograma de las longitudes del sépalo para cada especie
sns.histplot(data=df, x="sepal length (cm)", hue="target", bins=20)
plt.title("Histograma de Longitud del Sépalo")
plt.xlabel("Longitud del Sépalo (cm)")
plt.ylabel("Frecuencia")
plt.show()

# Diagrama de dispersión de longitud del sépalo vs. ancho del sépalo
sns.scatterplot(data=df, x="sepal length (cm)", y="sepal width (cm)", hue="target")
plt.title("Diagrama de Dispersión Sépalo")
plt.xlabel("Longitud del Sépalo (cm)")
plt.ylabel("Ancho del Sépalo (cm)")
plt.show()

# Diagrama de caja de longitud del pétalo por especie
sns.boxplot(data=df, x="target", y="petal length (cm)")
plt.xticks([0, 1, 2], iris.target_names)
plt.title("Diagrama de Caja de Longitud del Pétalo por Especie")
plt.xlabel("Especie")
plt.ylabel("Longitud del Pétalo (cm)")
plt.show()

# Matriz de correlación
corr_matrix = df.corr()
sns.heatmap(corr_matrix, annot=True, cmap="coolwarm")
plt.title("Matriz de Correlación")
plt.show()
