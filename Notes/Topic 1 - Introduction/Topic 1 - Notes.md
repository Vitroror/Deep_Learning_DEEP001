
# Introduction to Deep Learning

**Deep Learning** is a concept hosted under the **Machine Learning** subtopic of the **Artificial Inteligence** umbrella. Deep learning itself is an area of computation that is transversal to other machine learning topics, being **Supervised Learning**, **Unsupervised Learning** and **Reinforcement Learning**. Mainly, deep learning involves large ammounts of data and parameters, although the exact definitions are not as static as one would expect.

## Supervised Learning

The first type of machine learning we approach in this course is **Supervised Learning (SL)**. SL mainly involves mapping an input to an output, and learning this mapping by way of associating multiple input/output pairs, for example:

![[Pasted image 20261006163535.png]]

The idea in this example is to use a pair of data, being the age of a child and the height of this said child, to map an equation that properly predicts this data pair's causality. In practice, we can notice that, if fed with enough data, a supervised model could predict the height of a child based on the child's age.

Another common use for machine learning is **mapping**. Mapping is the use of supervised models to encode and decode certain behaviors and phenomena and create a model capable of understanding the task at hand.

![[Pasted image 20261006164447.png]]

![[Pasted image 20261006164509.png]]

Notice how, for supervised models, training requires constant **NOTATION**, i.e., every input and output pair requires constant encoding of respective values to make sure that the machine understands that one input legitimately grants one output. Therefore, supervised learning models have the downside of being very expensive models, that require a lot of effort to train.

## The Supervised x Unsupervised Intersection

Here are other three examples:

![[Pasted image 20261006164027.png]]

![[Pasted image 20261006164042.png]]

![[Pasted image 20261006164103.png]]

Notice that these three examples have one thing in common: a complex relationship between input and output. Naturally, there may be many possible outputs to an input, but outputs have rules which can be learned without need for output labels.

## Unsupervised Learning

**Unsupervised Learning (SL)** is based on the absence of labels for learning a dataset. One classic example is **clustering**, i.e., grouping a dataset based on similarities between inputs.

![[Pasted image 20261006170137.png]]

## Reinforcement Learning

**Reinforcement Learning (RL)** is based on associating states to actions and rewards. You wake actions to reach a state in which you are rewarded. A good example of this is chess, where the states are the valid positions of the board, actions are valid moves at a given time, and rewards are losing or taking a piece.

![[Pasted image 20261006171815.png]]

However, RL is very difficult, for a few reasons:

* It is very **stochastic**. A machine could take the same action twice, and the outcome might not be the same. The reward itself could be different as well.
* The **Temporal Credit Assignment** problem: Is the reward corresponding to the most recent action? Or is it an outcome of an action taken long ago?
* **Exploration-exploitation trade-off**: If there is one possible outcome, should an action be taken in this context? Is this the best possible outcome? Or should there be a search for another outcome?