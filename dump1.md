TensorFlow


TensorFlow is Google Brain's second generation machine learning system(DistBelief being their first-generation, proprietary, machine learning system), released as open source software on November 9, 2015. TensorFlow can run on multiple CPUs and GPUs. TensorFlow computations are expressed as stateful dataflow graphs. The name TensorFlow itself derives from the operations which such neural networks perform on multidimensional data arrays. These multidimensional arrays are referred to as "tensors". Its purpose is to train neural networks to detect and decipher patterns and correlations.



Basic Usage

TensorFlow represents computations as graphs and data as tensors which is a n-dimensional array or list. A tensor has a static type, a rank, and a shape. The data-flow is carried using feeds and fetches to get data into and out of arbitrary operations. It executes graphs in the context of sessions and maintains state with Variables. Nodes in the graph are called ops (short for operations). An op takes zero or more Tensors, performs some computation, and produces zero or more Tensors.

Phases

TensorFlow programs are usually structured into a construction phase, that assembles a graph, and an execution phase that uses a session to execute ops in the graph.For example, it is common to create a graph to represent and train a neural network in the construction phase, and then repeatedly execute a set of training ops in the graph in the execution phase.

Building the graph

To build a graph start with ops that do not need any input (source ops), such as Constant, and pass their output to other ops that do computation. The ops constructors in the Python library return objects that stand for the output of the constructed ops. You can pass these to other ops constructors to use as inputs. The TensorFlow Python library has a default graph to which ops constructors add nodes. The default graph is sufficient for many applications.

Session

To compute anything, a graph must be launched in a Session. A Session places the graph ops onto Devices, such as CPUs or GPUs, and provides methods to execute them. These methods return tensors produced by ops as numpy ndarray objects in Python, and as tensorflow::Tensor instances in C and C++. To actually multiply the matrices, and get the result of the multiplication, the graph must be launched in a session.

Fetches

To fetch the outputs of operations, execute the graph with a run() call on the Session object and pass in the tensors to retrieve. In the previous example we fetched the single node state, but you can also fetch multiple tensors:

Feeds

The examples above introduce tensors into the computation graph by storing them in Constants and Variables. TensorFlow also provides a feed mechanism for patching a tensor directly into any operation in the graph.

A feed temporarily replaces the output of an operation with a tensor value. You supply feed data as an argument to a run() call. The feed is only used for the run call to which it is passed. The most common use case involves designating specific operations to be "feed" operations by using tf.placeholder() to create them:
