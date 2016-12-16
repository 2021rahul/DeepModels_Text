#TensorFlow
TensorFlow is Google Brain's second generation machine learning system, released as open source software on November 9, 2015. TensorFlow can run on multiple CPUs and GPUs. TensorFlow computations are expressed as stateful dataflow graphs. The name TensorFlow itself derives from the operations which such neural networks perform on multidimensional data arrays. These multidimensional arrays are referred to as "tensors". Its purpose is to train neural networks to detect and decipher patterns and correlations.

https://www.tensorflow.org/tutorials/mnist/tf/

##Basic Usage
TensorFlow represents computations as graphs and data as tensors. The data-flow is carried using feeds and fetches to get data into and out of arbitrary operations. It executes graphs in the context of sessions and maintains state with Variables.

TensorFlow is a programming system in which you represent computations as graphs. Nodes in the graph are called ops (short for operations). An op takes zero or more Tensors, performs some computation, and produces zero or more Tensors. In TensorFlow terminology, a Tensor is a typed multi-dimensional array. For example, we can represent a mini-batch of images as a 4-D array of floating point numbers with dimensions [batch, height, width, channels]. 

###Session
To compute anything, a graph must be launched in a Session. A Session places the graph ops onto Devices, such as CPUs or GPUs, and provides methods to execute them. These methods return tensors produced by ops as numpy ndarray objects in Python, and as tensorflow::Tensor instances in C and C++. 

###Phases
TensorFlow programs are usually structured into a construction phase, that assembles a graph, and an execution phase that uses a session to execute ops in the graph.For example, it is common to create a graph to represent and train a neural network in the construction phase, and then repeatedly execute a set of training ops in the graph in the execution phase.

###Building the graph
To build a graph start with ops that do not need any input (source ops), such as Constant, and pass their output to other ops that do computation. The ops constructors in the Python library return objects that stand for the output of the constructed ops. You can pass these to other ops constructors to use as inputs. The TensorFlow Python library has a default graph to which ops constructors add nodes. The default graph is sufficient for many applications.

```python
import tensorflow as tf
matrix1 = tf.constant([[3., 3.]])
matrix2 = tf.constant([[2.],[2.]])
product = tf.matmul(matrix1, matrix2)
```
