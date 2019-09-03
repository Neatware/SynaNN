# SynaNN - Synaptic Neural Network

Synapses play an important role in biological neural networks.  They're joint points of neurons where learning and memory happened. Inspired by the synapse research of neuroscience, we found that a simple model can describe the key property of a synapse. By combining deep learning, we expect to build ultra large scale neural networks to solve real-world AI problems. At the same time, we want to create an explainable neural network model to better understand what an AI model doing instead of a black box solution.

Based on the analysis of excitatory and inhibitory channels of synapses, we proposed a synapse model called Synaptic Neural Network (SynaNN) where a synapse function is represented as the inputs of probabilities of both excitatory and inhibitory channels along with their joint probability as the output. SynaNN is constructed by synapses and neurons. A synapse graph is a connection of synapses. In particular, a synapse tensor is fully connected synapses from input neurons to output neurons. Synapse learning can work with gradient descent and backpropagation algorithms. SynaNN can be applied to construct MLP, CNN, and RNN models.  

<<<<<<< HEAD
SynaNN Key Features:

* Synapses are joint points of neurons with electronic and chemical functions, location of learning and memory
* A synapse function is nonlinear, log concavity, infinite derivative in surprisal space (negative log space)
* Surprisal synapse is Bose-Einstein distribution with coefficient as negative chemical potential
* SynaNN graph & tensor, surprisal space, commutative diagram, topological conjugacy, backpropagation algorithm
* SynaMLP, SynaCNN, SynaRNN are models for various neural network architecture
* Synapse block can be embedded into other neural network models
* Swap equation links between swap and odds ratio for healthcare, fin-tech, and insurance applications

One challenge was to represent the links of synapses as tensors so we can apply the neural network framework such as tensorflow for deep learning. A key step is to construct a Synapse layer to replace Dense layer in Keras so we can embed synapse in deep learning neural network. This has been done by defining a custom model of Synapse.  

Synapse pluses BatchNormalization can greatly speed up the processing to achieve an accuracy goal. We can think of a synapse as a statistical distribution computing unit while batch normalization makes evolution faster. 

Refrences:

"SynaNN: A Synaptic Neural Network and Synapse Learning"
https://www.researchgate.net/publication/327433405_SynaNN_A_Synaptic_Neural_Network_and_Synapse_Learning



=======
>>>>>>> origin/master
