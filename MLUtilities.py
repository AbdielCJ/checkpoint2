# Librerías útiles para el módulo Machine Learning
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PowerTransformer
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix
from scipy import stats
import cv2
import warnings
warnings.filterwarnings('ignore')
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# SESIÓN 02 (Separación, validación y evaluación de algoritmos para ML)

# Subconjuntos de entrenamiento, validación y prueba
def particionar(entradas, salidas, porcentaje_entrenamiento, porcentaje_validacion, porcentaje_prueba):
    temp_size = porcentaje_validacion + porcentaje_prueba
    print(temp_size)
    x_train, x_temp, y_train, y_temp = train_test_split(entradas, salidas, test_size =temp_size)
    if(porcentaje_validacion > 0):
        test_size = porcentaje_prueba/temp_size
        x_val, x_test, y_val, y_test = train_test_split(x_temp, y_temp, test_size = test_size)
    else:
        return [x_train, None, x_temp, y_train, None, y_temp]
    return [x_train, x_val, x_test, y_train, y_val, y_test]

# Funciones para calcular la precisión, sensibilidad y especificidad
def Accuracy(TP, TN, FP, FN):
    accuracy = (TP + TN) / (TP + TN + FP + FN)
    accuracy = accuracy * 100
    return accuracy

def Sensibilidad(TP, TN, FP, FN):
    sensibilidad = TP / (TP + FN)
    sensibilidad = sensibilidad * 100
    return sensibilidad

def Especificidad(TP, TN, FP, FN):
    especificidad = TN / (TN + FP)
    especificidad = especificidad * 100
    return especificidad

def evaluar(y_test, y_pred):
    resultado = confusion_matrix(y_test, y_pred)
    print(resultado)
    (TN, FP, FN, TP) = resultado.ravel()
    print("True positives: "+str(TP))
    print("True negatives: "+str(TN))
    print("False positives: "+str(FP))
    print("False negative: "+str(FN))
    
    acc = calcularAccuracy(TP, TN, FP, FN)
    sen = calcularSensibilidad(TP, TN, FP, FN)
    spec = calcularEspecificidad(TP, TN, FP, FN)
    print("Precision:"+str(acc.round(2))+"%")
    print("Sensibilidad:"+str(sen.round(2))+"%")
    print("Especificidad:"+str(spec.round(2))+"%")

# Métricas básicas de clasificación binaria
def metricasclass(matrizconfusion):
    """
    Calcula las métricas básicas de un modelo de clasificación binario.

    Parameters
    ----------
    matrizconfusion : 2D-Array
        Matriz de confusión con las siguientes entradas
        (0,0) = TN, (0,1) = FP, (1,0) = FN y (1,1) = TP
        
    Returns
    -------
    precision : Float
        Porcentaje de observaciones correctamente clasificadas.
    sensibilidad : Float
        Porcentaje de observaciones positivas correctamente clasificadas.
    especificidad : Float
        Porcentaje de observaciones negativas correctamente clasificadas.

    """
    (TN, FP, FN, TP) = matrizconfusion.ravel()
    precision = ((TP + TN) / (TP + TN + FP + FN)) * 100
    sensibilidad = (TP / (TP + FN)) * 100
    especificidad = (TN / (TN + FP)) * 100
    
    return precision, sensibilidad, especificidad

# Función de clasificación múltiple
def class_accuracy(matriz_confusion):
    acc = (np.trace(matriz_confusion)/matriz_confusion.sum()) * 100
    return acc

# Función de separación de datasets con K - Fold & Leave-One-Out Cross Validation
def kfoldleaveoneout(K, aleatorio):
    K = K
    aleatorio = aleatorio
    kfold = KFold(K, aleatorio)
    ciclo = 1
    for indices_train, indices_test in kfold.split(data):
        print("Ciclo: "+str(ciclo))
        print("\t datos para entrenamiento:"+str(data[indices_train]))
        print("\t datos para prueba:"+str(data[indices_test]))
        ciclo+=1

# Función que compara a 2 clasificadores
def comparar_clasificadores(y_test_1, y_pred_1, y_test_2, y_pred_2):
    resultado_1 = confusion_matrix(y_test_1, y_pred_1)
    print(resultado_1)
    (TN, FP, FN, TP) = resultado_1.ravel()
    print("True positives: "+str(TP))
    print("True negatives: "+str(TN))
    print("False positives: "+str(FP))
    print("False negative: "+str(FN))
    
    acc_1 = calcularAccuracy(TP, TN, FP, FN)
    sen_1 = calcularSensibilidad(TP, TN, FP, FN)
    spec_1 = calcularEspecificidad(TP, TN, FP, FN)
    print("Precisión del clasificador 1:"+str(acc_1.round(2))+"%")
    print("Sensibilidad del clasificador 1:"+str(sen_1.round(2))+"%")
    print("Especificidad el clasificador 1:"+str(spec_1.round(2))+"%")
    
    resultado_2 = confusion_matrix(y_test_2, y_pred_2)
    print(resultado_2)
    (TN, FP, FN, TP) = resultado_2.ravel()
    print("True positives: "+str(TP))
    print("True negatives: "+str(TN))
    print("False positives: "+str(FP))
    print("False negative: "+str(FN))
    
    acc_2 = calcularAccuracy(TP, TN, FP, FN)
    sen_2 = calcularSensibilidad(TP, TN, FP, FN)
    spec_2 = calcularEspecificidad(TP, TN, FP, FN)
    print("Precisión del clasificador 2:"+str(acc_2.round(2))+"%")
    print("Sensibilidad del clasificador 2:"+str(sen_2.round(2))+"%")
    print("Especificidad del clasificador 2:"+str(spec_2.round(2))+"%")
    
    if acc_1 > acc_2:
        print("El accuracy del clasificador 1 es mejor")
    elif acc_1 < acc_2:
        print("El accuracy del clasificador 2 es mejor")
    if sen_1 > sen_2:
        print("La sensibilidad del clasificador 1 es mejor")
    elif sen_1 < sen_2:
        print("La sensibilidad del clasificador 2 es mejor")
    if spec_1 > spec_2:
        print("La especificidad del clasificador 1 es mejor")
    elif spec_1 < spec_2:
        print("La especificidad del clasificador 2 es mejor")
        
# SESIÓN 03 (Algoritmos no supervisados)

# Función para calcular distancias euclidianas
def distEuclidiana(muestra, dataset):
    distancias = np.zeros((dataset.shape[0],1))
    for counter in range(0,dataset.shape[0]):
        distancias[counter] = np.linalg.norm(muestra-dataset[counter])
    return distancias

# Función para calcular el centroide más cercano
def centroideCercano(muestra, listaCentroides):
    listaDistancias = distEuclidiana(muestra, listaCentroides)
    centroideCercano = np.argmin(listaDistancias) ## Busca una función de Numpy para encontrar el argumento mínimo de un array
    return centroideCercano

# Función para clasificar centroides
def clasificarPorCentroides(muestras, centroides):
    resultado = np.zeros((muestras.shape[0],1))
    for counter in range(0, muestras.shape[0]):
        resultado[counter] = centroideCercano(muestras[counter], centroides)
    return resultado

# Función para filtrar los datos en base a la categoría esperada a la que pertenecen
def separarDatos(muestras, valoresEsperados, valorAFiltrar):
    indices = np.where(valoresEsperados == valorAFiltrar)
    return muestras[indices], valoresEsperados[indices]

# Función para obtener la moda
def obtenerModa(resultados):
    moda = (stats.mode(resultados)[0]).reshape(-1)
    return moda[0]

# Función para obtener el Accuracy de K Medias
def obtenerAccuracy_kmedias(muestras, centroides):
    numMuestras = muestras.shape[0]
    
    resultados = clasificarPorCentroides(muestras, centroides)
    moda = obtenerModa(resultados)
    
    indicesErrores = np.where(resultados != moda)
    cantidadErrores = len(resultados[indicesErrores])
    accuracy = ((numMuestras - cantidadErrores)/numMuestras) * 100 ## Calcula el accuracy
    return accuracy

# Función para recomendar películas
def recomiendamePeliculas(listaDePeliculas,datosPeliculas,peliculaEjemplo,centroides):
    #Vamos a buscar el centroide mas cercano (con MLUtilities ;) )
    clasificacionDeseada = centroideCercano(peliculaEjemplo, centroides)
    
    #Luego, vamos a clasificar todas las peliculas por centroides.
    clasificaciones = clasificarPorCentroides(datosPeliculas, centroides)
    
    #Finalmente, sacaremos los indices que hacen match entre clasificaciones.
    indices = np.where(clasificaciones == clasificacionDeseada)[0]
    
    #Y regresamos la lista de peliculas.
    return listaDePeliculas[indices]

# SESIÓN 04 (Regresión y series de tiempo)

# Función que transforma una serie de tiempo a un dataset
def transformarSerieADataset(serie, elementosPorMuestra):
    dataset = None
    salidasDataset = None
    for counter in range (len(serie)-elementosPorMuestra-1):        
        muestra = np.array([serie[counter:counter+elementosPorMuestra]])        
        salida = np.array([serie[counter+elementosPorMuestra]])
        if dataset is None:
            dataset = muestra
        else:
            dataset = np.append(dataset,muestra,axis = 0)
        if salidasDataset is None:
            salidasDataset = salida    
        else:        
            salidasDataset = np.append(salidasDataset,salida)
    return dataset, salidasDataset

# SESIÓN 05 (Bosques aleatorios)

# Sospechosos de adivina quién
def mostrarSospechosos(nombres, sospechosos):
    print("Sospechosos que quedan:")
    for contador in range(len(sospechosos)):
        if(sospechosos[contador] == True):
            print(nombres[contador])
            
# Función para descartar sospechosos de adivina quién
def descartarSospechosos(caracteristica, valorQueBuscas, sospechosos):
    respuesta = np.where(x[:,caracteristica] == valorQueBuscas, True, False)
    for contador in range(len(sospechosos)):
        respuesta[contador] = sospechosos[contador] and respuesta[contador]
    return respuesta

# SESIÓN 06 (Clasificación y redes neuronales)

# Función para aplicar el producto punto de vectores
def calcularZ(w, x, b):
    z = np.dot(w,x) + b # Investiga la forma de calcular el producto punto con numpy
    return z

# Función de activación de función identidad
def activacion(z):
    y_predecida = z # Función identidad
    return y_predecida

# Función para calcular la neurona con función identidad
def neurona(w,x,b):
    z = calcularZ(w,x,b)
    y = activacion(z)
    return y

# Función de activación de función sigmoidal
def activacionSigmoide(z):
    y = 1 / (1 + np.exp(-z)) # Función sigmoidal
    return y

# Función para calcular la neurona con función sigmoidal
def neuronaSigmoide(w,x,b):
    z = calcularZ(w,x,b)
    y_pred = activacionSigmoide(z)
    return y_pred

# Función de activación de función tangencial hiperbólica
def activacionTanh(z):
    y = np.tanh(z) # Función tangencial hiperbólica
    return y
 
# Función para calcular la neurona con función tangencial hiperbólica
def neuronaTanh(w,x,b):
    z = calcularZ(w,x,b)
    y_pred = activacionTanh(z)
    return y_pred

# Función para inicializar capa neuronal
def inicializarCapa(numCaracteristicas, numNeuronas):
    w = np.random.rand(numCaracteristicas,numNeuronas)
    w_transpose = w.T
    b = np.random.rand(numNeuronas)
    return w_transpose, b

# SESIÓN 08 (Procesamiento de imágenes)

# Función para convertir a escala de grises
def convertirAGrayScale(imagen):
    imagenGris = np.sum(imagen, axis = 2) / 3    
    return imagenGris

# Función para binarizar
def binarizar(imagenGris, threshold):
    imgBinaria = np.where(imagenGris > threshold, 255, 0)
    return imgBinaria

# Función para reducir espacios de color
def reducirColores(imagenGris, cantidadDeColores):
    if(cantidadDeColores <= 0):
        return np.zeros_like(imagenGris)
    
    stepSize = int(255 / (cantidadDeColores))
    for counter in range (0, 255, stepSize):
        if(counter == 0):
            buffer = np.zeros_like(imagenGris)
        else:
            imgFiltrada = np.where(((imagenGris > (counter - stepSize)) & (imagenGris <= counter)), counter, 0)
            buffer = np.add(buffer, imgFiltrada)
    return buffer

# Función para obtener el negativo
def obtenerNegativo(imagen):
    negativo = np.abs(imagen - 255)
    return negativo

# Función para recortar imagen
def recortar(imgOriginal, imgBinarizada):
    patronBinario = np.where(imgBinarizada > 0, 0, 1)
    imgRecortada_rojo = np.multiply(imgOriginal[:,:,0],patronBinario)
    imgRecortada_verde = np.multiply(imgOriginal[:,:,1],patronBinario)
    imgRecortada_azul = np.multiply(imgOriginal[:,:,2],patronBinario)
    imgRecortada = np.dstack((imgRecortada_rojo, imgRecortada_verde, imgRecortada_azul))
    return imgRecortada

# Función para crear un histograma de colores
def crearHistograma(imagen):
    histograma = np.zeros((256))
    imgEnArray = np.ravel(imagen)
    for counter in range(0, len(imgEnArray)):
        histograma[int(imgEnArray[counter])]+=1    
    return histograma

# Función para resumir el proceso de extracción y apertura del archivo en un sólo paso
def abrirImagen(ruta, mostrar = False):
    imageOpenCV = cv2.imread(ruta, cv2.IMREAD_COLOR)
    imagen = cv2.cvtColor(imageOpenCV, cv2.COLOR_BGR2RGB)
    if(mostrar):
        plt.figure(figsize=(7,7))
        plt.imshow(imagen)
        plt.show()
    return imagen

# Función para crear el descriptor
def crearDescriptor(imagen):
    histogramaRojo = crearHistograma(imagen[:,:,0])
    histogramaVerde = crearHistograma(imagen[:,:,1])
    histogramaAzul = crearHistograma(imagen[:,:,2])
    descriptor = np.concatenate([histogramaRojo, histogramaVerde, histogramaAzul])
    print(descriptor.shape)
    return descriptor
