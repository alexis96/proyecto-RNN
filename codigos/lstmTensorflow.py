# -*- coding: utf-8 -*-
"""
Created on Mon Nov 27 19:54:49 2017

@author: alexi
"""
"""
Generacion de textos usando una Red Neuronal Recurrente (LSTM).
"""


import tensorflow as tf
import numpy as np
import random
import time
import sys



## RNN con num_layers LSTM capas y una completamente conectada capa de salida
## La red permite un número dinámico de iteraciones, dependiendo de las entradas que recibe.
##
##    out   (fc layer; out_size)
##     ^
##    lstm
##     ^
##    lstm  (lstm size)
##     ^
##     in   (in_size)
class ModelNetwork:
	def __init__(self, in_size, lstm_size, num_layers, out_size, session, learning_rate=0.003, name="rnn"):
		self.scope = name

		self.in_size = in_size
		self.lstm_size = lstm_size
		self.num_layers = num_layers
		self.out_size = out_size

		self.session = session

		self.learning_rate = tf.constant( learning_rate )

		# ultimo estado de LSTM, usado cuando se corre la red en modo de prueba
		self.lstm_last_state = np.zeros((self.num_layers*2*self.lstm_size,))

		with tf.variable_scope(self.scope):
			## (batch_size, timesteps, in_size)
			self.xinput = tf.placeholder(tf.float32, shape=(None, None, self.in_size), name="xinput")
			self.lstm_init_value = tf.placeholder(tf.float32, shape=(None, self.num_layers*2*self.lstm_size), name="lstm_init_value")

			# LSTM
			self.lstm_cells = [ tf.contrib.rnn.BasicLSTMCell(self.lstm_size, forget_bias=1.0, state_is_tuple=False) for i in range(self.num_layers)]
			self.lstm = tf.contrib.rnn.MultiRNNCell(self.lstm_cells, state_is_tuple=False)

			# Computo iterativo de la salidas de la red recurrente
			outputs, self.lstm_new_state = tf.nn.dynamic_rnn(self.lstm, self.xinput, initial_state=self.lstm_init_value, dtype=tf.float32)

			# activacion lineal (capa FC en la parte superior de la red LSTM)
			self.rnn_out_W = tf.Variable(tf.random_normal( (self.lstm_size, self.out_size), stddev=0.01 ))
			self.rnn_out_B = tf.Variable(tf.random_normal( (self.out_size, ), stddev=0.01 ))

			outputs_reshaped = tf.reshape( outputs, [-1, self.lstm_size] )
			network_output = ( tf.matmul( outputs_reshaped, self.rnn_out_W ) + self.rnn_out_B )

			batch_time_shape = tf.shape(outputs)
			self.final_outputs = tf.reshape( tf.nn.softmax( network_output), (batch_time_shape[0], batch_time_shape[1], self.out_size) )


			## Entrenamiento: proporciona los objetivos de salida para el entrenamiento supervisado
			self.y_batch = tf.placeholder(tf.float32, (None, None, self.out_size))
			y_batch_long = tf.reshape(self.y_batch, [-1, self.out_size])

			self.cost = tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits(logits=network_output, labels=y_batch_long) )
			self.train_op = tf.train.RMSPropOptimizer(self.learning_rate, 0.9).minimize(self.cost)


	## Entrada: X es un solo elemento, no una lista
	def run_step(self, x, init_zero_state=True):
		## Reinicia el estado inicial de la red
		if init_zero_state:
			init_value = np.zeros((self.num_layers*2*self.lstm_size,))
		else:
			init_value = self.lstm_last_state

		out, next_lstm_state = self.session.run([self.final_outputs, self.lstm_new_state], feed_dict={self.xinput:[x], self.lstm_init_value:[init_value]   } )

		self.lstm_last_state = next_lstm_state[0]

		return out[0][0]


	## xbatch debe ser (batch_size, timesteps, input_size)
	## ybatch debe ser (batch_size, timesteps, output_size)
	def train_batch(self, xbatch, ybatch):
		init_value = np.zeros((xbatch.shape[0], self.num_layers*2*self.lstm_size))

		cost, _ = self.session.run([self.cost, self.train_op], feed_dict={self.xinput:xbatch, self.y_batch:ybatch, self.lstm_init_value:init_value   } )

		return cost




# convierte cadenas a arreglos de caracteres -- genera un arreglo len(data) x len(vocab)
# Vocab es una lista de elementos
def embed_to_vocab(data_, vocab):
	data = np.zeros((len(data_), len(vocab)))

	cnt=0
	for s in data_:
		v = [0.0]*len(vocab)
		v[vocab.index(s)] = 1.0
		data[cnt, :] = v
		cnt += 1

	return data

def decode_embed(array, vocab):
	return vocab[ array.index(1) ]






ckpt_file = ""
TEST_PREFIX = "señores" # Prefijo disponible para hacer la prueba

print ("Usage:")
print ('\t\t ', sys.argv[0], ' [ckpt model to load] [prefix, e.g., "The "]')
if len(sys.argv)>=2:
	ckpt_file=sys.argv[1]
if len(sys.argv)==3:
	TEST_PREFIX = sys.argv[2]


ckpt_file = "saved/model.ckpt"

## cargar los datos
data_ = ""
with open('discursos.txt', 'r') as f:
	data_ += f.read()
data_ = data_.lower()

## Convierte a codificacion 1-hot
vocab = sorted(list(set(data_)))

data = embed_to_vocab(data_, vocab)


in_size = out_size = len(vocab)
lstm_size = 256 #128
num_layers = 2
batch_size = 64 #128
time_steps = 100 #50

NUM_TRAIN_BATCHES = 10000

# Numero de caracteres de prueba generados por la red despues del entrenamiento
LEN_TEST_TEXT = 500


## Inicializa la red
config = tf.ConfigProto()
config.gpu_options.allow_growth=True
sess = tf.InteractiveSession(config=config)

net = ModelNetwork(in_size = in_size,
					lstm_size = lstm_size,
					num_layers = num_layers,
					out_size = out_size,
					session = sess,
					learning_rate = 0.003,
					name = "char_rnn_network")

sess.run(tf.global_variables_initializer())

saver = tf.train.Saver(tf.global_variables())



## 1) ENTRENAR LA RED
if ckpt_file == "":
	last_time = time.time()

	batch = np.zeros((batch_size, time_steps, in_size))
	batch_y = np.zeros((batch_size, time_steps, in_size))

	possible_batch_ids = range(data.shape[0]-time_steps-1)
	for i in range(NUM_TRAIN_BATCHES):
		# Muestra tiempo por paso de ejemplos consecutivos del archivo de texto del conjunto de datos
		batch_id = random.sample( possible_batch_ids, batch_size )

		for j in range(time_steps):
			ind1 = [k+j for k in batch_id]
			ind2 = [k+j+1 for k in batch_id]

			batch[:, j, :] = data[ind1, :]
			batch_y[:, j, :] = data[ind2, :]


		cst = net.train_batch(batch, batch_y)

		if (i%100) == 0:
			new_time = time.time()
			diff = new_time - last_time
			last_time = new_time

			print ("batch: ",i,"   loss: ",cst,"   speed: ",(100.0/diff)," batches / s")

	saver.save(sess, "saved/model.ckpt")




## 2) GENERA LEN_TEST_TEXT CARACTERES USANDO LA RED ENTRENADA

if ckpt_file != "":
	saver.restore(sess, ckpt_file)

TEST_PREFIX = TEST_PREFIX.lower()
for i in range(len(TEST_PREFIX)):
	out = net.run_step( embed_to_vocab(TEST_PREFIX[i], vocab) , i==0)

print ("SENTENCE:")
gen_str = TEST_PREFIX
for i in range(LEN_TEST_TEXT):
	element = np.random.choice( range(len(vocab)), p=out ) # Carácter de muestra de la red según las probabilidades de salida generadas
	gen_str += vocab[element]

	out = net.run_step( embed_to_vocab(vocab[element], vocab) , False )
print (gen_str)
