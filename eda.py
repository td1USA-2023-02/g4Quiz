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


df.to_csv('datosnuevos.csv', index=False)


#PUNTO 5 2 GRAFICOS NUEVOS DE LA TABLA NUEVA
#histograma de los nuevos datos

# Crear un nuevo DataFrame llamado "datosnuevos" basado en la columna "nueva_columna" de "df"
datosnuevos = df[['nueva_columna']]

# Mostrar las primeras filas del nuevo DataFrame
print(datosnuevos.head())



# Histograma de las longitudes del sépalo para cada especie
sns.histplot(data=df, x='nueva_columna', hue="sepal width (cm)", bins=2)
plt.title("Histograma de Longitud del Sépalo")
plt.xlabel("Longitud del Sépalo (cm)")
plt.ylabel("Frecuencia")
plt.show()

sns.scatterplot(data=df, x="sepal length (cm)", y="sepal width (cm)", hue="nueva_columna")
plt.title("Diagrama de Dispersión Sépalo")
plt.xlabel("Longitud del Sépalo (cm)")
plt.ylabel("Ancho del Sépalo (cm)")
plt.show()


#Pregunta 6
#La frecuencia de que una flor cumpla con la longitud del sepalo y sea margarita es de 3 según el histograma 



# Matriz de correlación
corr_matrix = df.corr()
sns.heatmap(corr_matrix, annot=True, cmap="coolwarm")
plt.title("Matriz de Correlación")
plt.show()
