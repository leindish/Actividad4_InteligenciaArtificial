# Importar las librerías necesarias
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder #Para convertir variables categoricas a valores númericos
from sklearn.cluster import KMeans #Para el algoritmo de clustering
import matplotlib.pyplot as plt # Para graficar
import matplotlib.colors as mcolors # Para asignar colores

# Fijar la semilla para reproducibilidad
np.random.seed(42)

# Definición de las variables
dias_semana = ['Lunes', 'Martes', 'Miércoles', 'Jueves', 'Viernes', 'Sábado', 'Domingo']
horas_dia = ['Mañana', 'Tarde', 'Noche']
condiciones_clima = ['Soleado', 'Nublado', 'Lluvioso']

# Función para generar la demanda de pasajeros
def generar_demanda():
    return np.random.poisson(lam=np.random.choice([20, 30, 50]), size=1)[0]

# Función para generar la capacidad del transporte
def generar_capacidad():
    return np.random.choice([50, 80, 100], size=1)[0]

# Generar un dataset con 500 muestras
datos = []
for _ in range(500):
    dia = np.random.choice(dias_semana)
    hora = np.random.choice(horas_dia)
    clima = np.random.choice(condiciones_clima)
    capacidad = generar_capacidad()
    demanda = generar_demanda()

    datos.append([dia, hora, clima, capacidad, demanda])

# Convertir la lista de datos a un DataFrame de pandas
df = pd.DataFrame(datos, columns=['Día de la Semana', 'Hora del Día', 'Condición Climática', 'Capacidad del Transporte', 'Demanda de Pasajeros'])

# Mostrar las primeras filas del dataset
print("Primeras Filas del dataset:")
print(df.head())

# Codificar variables categóricas
df_encoded = df.copy()
label_encoders = {}
for column in ['Día de la Semana', 'Hora del Día', 'Condición Climática']:
    le = LabelEncoder()
    df_encoded[column] = le.fit_transform(df[column])
    label_encoders[column] = le

# Seleccionar las características para el clustering
X = df_encoded[['Día de la Semana', 'Hora del Día', 'Condición Climática', 'Capacidad del Transporte', 'Demanda de Pasajeros']]

# Encontrar el número óptimo de clusters usando el método del codo
inertia = []
for k in range(1, 11):
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X)
    inertia.append(kmeans.inertia_)

# Graficar el método del codo
plt.figure(figsize=(8, 5))
plt.plot(range(1, 11), inertia, marker='o')
plt.title('Método del Codo')
plt.xlabel('Número de Clusters')
plt.ylabel('Inercia')
plt.grid(True)
plt.show()

# Aplicar K-means con un número óptimo de clusters
kmeans = KMeans(n_clusters = 3, random_state = 42)
df_encoded['Cluster'] = kmeans.fit_predict(X)

# Mostrar los resultados agrupados por cluster
print("\nResultados agrupados por cluster:")
print(df_encoded.groupby('Cluster').mean())

colores_clusters = ['red', 'green', 'blue']
n_clusters = len(colores_clusters)

# Graficar los clusters
plt.figure(figsize=(10, 6))
scatter = plt.scatter(df_encoded['Día de la Semana'], 
                      df_encoded['Hora del Día'], 
                      c=df_encoded['Cluster'], 
                      cmap=mcolors.ListedColormap(colores_clusters))
plt.title('Clusters de Transporte Masivo')
plt.xlabel('Día de la Semana')
plt.ylabel('Hora del Día')
cbar = plt.colorbar(scatter, ticks=np.arange(n_clusters))
cbar.ax.set_yticklabels([f'Cluster {i}' for i in range(n_clusters)]) 
cbar.set_label('Cluster')
plt.show()

