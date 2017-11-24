## Red neuronal recurrente

Aqui mostraremos un poco de lo que es una red neuronal recurrente asi como unas aplicaciones
con algunos documentos creados.

### Que es una RNN?

Las redes neuronales recurrentes completamente conectadas tienen caminos de retro alimentacion
entre todos los elementos que la conforman. Una sola neurona esta entonces conectada a las 
neuronas posteriores en la siguiente capa, las neuronas pasadas de la capa anterior y a ella
misma a traves de vectores de pesos variables que sufren alteraciones en cada epoch.

La mayoria de las aplicaciones de este tipo de red son la identificacion y clasificacion de patrones
secuenciales con distintas posibilidades de ocurrir a traves del tiempo, por ejemplo para traducciones
generacion de texto como en nuestro caso o en donde usemos datos en secuencia en donde el orden tenga
importancia

### Simple red recurrente

Para la creacion de esta red recurrente simple usamos un codigo hecho todo en numpy obtenido de [aqui](https://gist.github.com/karpathy/d4dee566867f8291f086).
La cual modificamos para nuesto uso y hacer la generacion de municipios creada en un libreta
la cual esta ubicada en el repositorio. [Libreta](codigos/simple-RNN.ipynb)




