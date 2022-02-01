# 1 - The Machine Learning Landscape

# Types of Machine Learning Systems

## Supervised/Unsupervised Learning

### 1 - Supervised Learning

In supervised learning, the training set is **labeled**. Predicting spam mails is a good example for supervised learning, the model will learn from already labeled examples. Another typical task is predicting target numeric value, such as the price of car, given set of features. This sort of task is called regression.

<aside>
    💡 <i>Some regression algorithms can be used for classification and vice versa. For example, Logistic Regression is commonly used for classification.</i>
</aside>


![Untitled](1%20-%20The%20Machine%20Learning%20Landscape/Untitled.png)

**Some Supervised Learning Algorithms**

- k-Nearest Neighbors
- Linear Regression
- Logistic Regression
- Support Vector Machines
- Decision Trees and Random Forests
- Neural Networks

### 2 - Unsupervised Learning

In unsupervised learning the training data is **unlabelled.** A clustering algorithm to detect groups of similar website visitors is an example.

![Untitled](1%20-%20The%20Machine%20Learning%20Landscape/Untitled%201.png)

**Some Unsupervised Learning Algorithms**

- Clustering
    - K-Means
    - DBSCAN
    - Hierarchical Cluster Analysis (HCA)
- Anomaly detection and novelty detection
    - One-class SVM
    - Isolation Forest
- Visualization and dimensionality reduction
    - Principal Component Analysis (PCA)
    - Kernel PCA
    - Locally Linear Embedding (LLE)
    - t-Distributed Stochastic Neighbor Embedding (t-SNE)
- Association rule learning
    - Apriori
    - Eclat
    

<aside>
    💡 <b>TIP</b>: It’s a good idea to try to reduce the dimension of the training data using a dimensionality reduction algorithm before feeding it to another ML algorithm. It will run faster and the data will keep less disk and memory space.
</aside>




### 3 - Semisupervised Learning

In semi-supervised learning, the training data is **partially labeled**. Google’s tagging persons algorithm is a good example, once you tag a person, it is able to name the same person in every photo.

![Untitled](1%20-%20The%20Machine%20Learning%20Landscape/Untitled%202.png)

**Some Semisupervised Learning Algorithms**

- Deep Belief Networks (DBNs)
- Boltzmann Machines (RBMs)

### 4 - Reinforcement Learning

In reinforcement learning, an *agent* observes the environment, selects and performs actions. After the action, it gets *rewards* or *penalties*. It must then learn by itself what is the best strategy, called *policy*, to get the most reward over time.

![Untitled](1%20-%20The%20Machine%20Learning%20Landscape/Untitled%203.png)

## Batch and Online Learning

### 1 - Batch Learning

In batch learning, the system is incapable of incrementally learning, it must be trained using all the available data. After training, the system doesn’t learn anymore on production, it is just applying what it learned, this is called *offline learning*. If you want the system to know about the new data, you need to train the system with the new data and also the old data. 

### 2 - Online Learning

In online learning, you train the system incrementally by feeding it data sequentially -individually or mini-batches-.

 It can be used on huge datasets that cannot fit in memory. The algorithm loads part of the data and runs a training step on that data, and repeats the process until has run all of the data. This is called *out-of-core learning.* But it’s usually done offline, think it as **incremental learning**.

The l**earning rate** parameter determines how fast the system adapts to changing data. With a low learning rate will learn more slowly but also be less sensitive to noises or outliers. With a high learning rate the system will rapidly adapt to the new data but it also quickly forget the old data.

## Instance-Based Versus Model-Based Learning

### 1 - Instance-based Learning

In instance-based learning, the system learns the examples by heart, then generalizes to new cases using a similarity measure.

**Some Instance-based Learning Algorithms**

- k-Nearest Neighbors
- Kernel Machines
- RBF networks

![Untitled](1%20-%20The%20Machine%20Learning%20Landscape/Untitled%204.png)

### 2 - Model-based Learning

In model-based learning, you build a model of the examples and use that model to make predictions. (that you always do)

# Main Challenges of Machine Learning

## Insufficient Quantity of Training Data

Machine Learning algorithms need a lot of data to work properly. In *2001 Microsoft researchers’ paper* shows that data is matter.

> These results suggest that we may want to reconsider the trade-off between spending time and money on algorithm development versus spending it on corpus development.
> 

It should be noted, however, that small- and medium-sized datasets are still very common, and it is not always easy or cheap to get extra training data⁠—so don’t abandon algorithms just yet.

## Nonrepresentative Training Data

It is crucial to use a training set that is representative of the cases you want to generalize to.

If the sample is too small, you will have *sampling noise.* But even very large samples can be nonrepresentative if the sampling method is flawed. This is called *sampling bias. - 1936 USA elections -* 

## Poor-Quality Data

If training data full of errors, outliers, and noise, it will make the system is less likely to perform well. 

**For outliers**: Discard them or try to fix them manually

**For missing features**: Discard the feature or fill it - e.g with median or mean - or train one model with feature and one model without it. 

## Irrelevant Features

Your model will perform well in case of there are enough relevant features and not many irrelevant ones. 

**Feature Engineering**

* Feature selection: Selecting the most useful features

* Feature extraction: Combining existing features to produce a more useful one

* Creating new features by gathering new data