import tensorflow as tf
from tensorflow  import keras

import numpy as np
import matplotlib.pyplot as plt 


fashion_mnist = keras.datasets.fashion_mnist    

(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()







#Data Preprocessing
#  en este paso lo que hago es reducir los pesos que voy a introducir en la red neuronal dividiendo por 255 que es el valor maximo de un pixel
test_images = test_images / 255.0
train_images = train_images / 255.0


## Building the model
# estamos hablando de un modelo secuencial es decir de una red neuronal feed forward no recurrente ni convolucional
# por eso es keras.secuecial , dentro de este modelo tenemos que definir las capas que va a tener la red neuronal

model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28,28)), #input layer (1)  #28*28 = 784 pixels
    keras.layers.Dense(128, activation='relu'), #hidden layer (2) #128 neuronas
    keras.layers.Dense(10, activation='softmax') #output layer (3) #10 neuronas ya que son 10 clases que queresmos clasificar
])


## Compiling the model
# en este paso lo que hacemos es definir el optimizador y la funcion de perdida
# el optimizador es el que se encarga de ajustar los pesos de la red neuronal
# la funcion de perdida es la que se encarga de medir el error de la red neuronal
# metrics es la que se encarga de medir la precision de la red neuronal 
# Estos son hiperparametros que se pueden modificar para mejorar la precision de la red neuronal
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
## Training the model
# en este paso lo que hacemos es entrenar la red neuronal
# epochs es el numero de veces que va a iterar sobre el conjunto de datos
# Estos son hiperparametros que se pueden modificar para mejorar la precision de la red neuronal
model.fit(train_images, train_labels, epochs=8,)

## Evaluating the model
# en este paso lo que hacemos es evaluar la red neuronal
test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=1)
print('Test accuracy:', test_acc)

## Making predictions
# en este paso lo que hacemos es hacer predicciones con la red neuronal
# print(class_names[np.argmax(predictions[65])])
# plt.figure()
# plt.imshow(test_images[65])
# plt.colorbar()
# plt.grid(False)
# plt.show()


def  predict_image(model , img  , label):
    class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
                    'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']   
    predictions = model.predict(np.array([img]))
    predicted_class = class_names[np.argmax(predictions)]
    show_image(img , class_names[label] , predicted_class)




def show_image(img ,label ,predictions):
    plt.figure()
    plt.imshow(img , cmap=plt.cm.binary)
    plt.title("Expected : " +label)
    plt.xlabel("Neuronal Network prediction "  + predictions)
    plt.colorbar()
    plt.grid(False)
    plt.show()



def get_number():
    while True:
        num = input("Pick a number: ")
        if num.isdigit():
            num = int(num)
            if 0 <= num <= 1000:
                return int(num)
            else:
                print("Try again...")
        else:
            print("Try again...")


num = get_number()
image = test_images[num]
label = test_labels[num]
predict_image(model , image , label)