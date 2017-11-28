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

### Nombres generados

1. Iteracion 800
* Simulacuipas
* Sanamo
* Sanéchilospo

2. Iteracion 300,000+
* Amamúlón
* Atamelcuba
* Chilpab
* Coatlán de Zuininal
* Caltiza

dado la construccion del archivo usado como los datos estan ordenados alfabeticamente
la generacion suele tambien ser en ese orden.

### Red Recurrente LSTM

Para los siguientes resultados usamos una red LSTM (Long short term memory) para generar textos.
Esta red esta hecha usando tensorflow y obtenida de [Aqui](https://github.com/spiglerg/RNN_Text_Generation_Tensorflow) la cual tambien 
modificamos para nuestro caso. Usamos para el aprendizaje de la red algunos discursos politicos de mexico y alguno de otro pais hispanohablante.

### Generacion de discursos politicos


Palabra inicial "el"
Sentencia:
el mundo por su invaluable apoyo para todos los puestos son los cuerdas por derechos sobre el pueblo como. 
pero támico, permita que sibe de tuvio a nuestros coraciones de nuestro pasado y las proporciones de nuestro tiempo.
la historia ha de ser liberos metilitares, más importando 31 de esos igualios de progreso y 
bandándos y estaba en todo su libertad. y en la que más derecho, de las insuficiencias, 
la realización son sustituidas por el fatigante método de las promesas, condena es nuestro partido;

Palabra inicial "viva"
Sentencia:
viva por la libertad, que salvó a su asperar más que lo lo que piensa, o no se atreve a decir lo que piensa, no es un hombre honrado. 
un hombre de trabajo que confía más libertades, a mutilar nuestra dignidad y a truncar nuestro 
porvenir como pueblo ni fue en el orden en la habana, por la coutadiona de poder y el golpe consemano, 
que en méxico para todas las revoluciones llevan dentro de sí el germen de su propia síntesis. 
entre nosotros el plano de la eclositación etapa en que la luciencia

Palabra inicial "señores"
Sentencia:
señores de la vida como razón de la plaza de sangrese favor de la integridad en el servicio público y el combate a la corrupción.
se queó el gobierno ha querido subrayar hoy la armonía constitucional en que las inicias nuestra democracia, 
de la globalidad y del conocimiento. tenemos todo. tenemos el gran potencial para que cada mexicano escriba 
su propia historia de éxito. la correalidad de la falta de empleo, que no se otraín someterla afirarse para el desgreño, 
ausente de seriemos de corrupción


### Aun no subo el codigo falta comentarlo
