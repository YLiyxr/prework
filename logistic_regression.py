import numpy as np
import tensorflow as tf


def sigmoid(x):
    """Vectorized sigmoid function

    Args:
        x: NumPy array

    Length: 1-2 lines

    Returns: element-wise sigmoid of x
    """
    raise NotImplementedError()


class LogisticRegressionBase(object):
    def __init__(self, dim):
        """Abstract class for a logistic regression model

        Args:
            dim: dimension of model weights
        """
        self.d = dim

    def _update_weights(self, X, y, alpha):
        """Applies an SGD step by updating self.w and self.b

        Args:
            X: training batch data matrix (2D NumPy array)
            y: training batch labels (NumPy vector)
            alpha: learning rate

        Returns: batch loss after update
        """
        raise NotImplementedError()

    def _initialize(self):
        """Initialize model variables"""
        pass        

    def train(self, X, y, lr=0.5, lr_decay=0.99, n_epochs=10, batch_size=10,
        verbose=False, seed=1701):
        """Train a logistic regression model

        Args:
            X: training data matrix (2D NumPy array)
            y: training data labels (NumPy vector)
            lr: learning rate
            lr_decay: multiplicative decay for learning rate
            n_epochs: number of epochs
            batch_size: size of each SGD batch
            verbose: print training progress?
            seed: random seed; if None, no random seeding
        """
        # Initialize model
        np.random.seed(seed)
        self._initialize()
        # SGD training loop
        for t in xrange(n_epochs):
            losses = []
            # Shuffle training set
            idxs = np.random.permutation(X.shape[0])
            X = X[idxs]
            y = y[idxs]
            # Run batches
            for i in range(0, X.shape[0], batch_size):
                # Get batch data
                X_batch = X[i : i+batch_size]
                y_batch = y[i : i+batch_size]
                # Update weights and get loss
                losses.append(self._update_weights(X_batch, y_batch, lr))
            # Decay learning rate
            lr = max(1e-6, lr_decay * lr)
            # Print status
            if verbose:
                avg_loss = np.mean(losses)
                print "Epoch {0:<2}\tAverage loss = {1:.6f}".format(t, avg_loss)

    def predict(self, X):
        """Get positive class probabilities for test data

        Args:
            X: test data matrix (2D NumPy array)

        Returns: NumPy vector of positive class probabilities for test data
        """
        raise NotImplementedError()

    def accuracy(self, X, y, b=0.5):
        """Get model accuracy on test data

        Args:
            X: test data matrix (2D NumPy array)
            y: test data labels (NumPy vector)

        Returns: fraction of correct predictions
        """
        y_hat = self.predict(X)
        return np.mean((y_hat > b) == (y > b))


class LogisticRegression(LogisticRegressionBase):

    def __init__(self, dim):
        super(LogisticRegression, self).__init__(dim)
        self.w = None
        self.b = None


    def _initialize(self):
        """Initialize model variables"""
        self.w = np.random.normal(scale=0.1, size=self.d)
        self.b = 0.
    
    def _score(self, X):
        """Get positive class score for data

        Args:
            X: data matrix (2D NumPy array)

        Returns: NumPy vector of positive class scores for data
        """
        return np.ravel(X.dot(self.w) + self.b)

    def _loss(self, X, y):
        """Get logistic loss for data with respect to labels

        Args:
            X: data matrix (2D NumPy array)
            y: data labels (NumPy vector)

        Returns: average logistic loss
        """
        y = (2*y - 1).copy()
        z = y * self._score(X)
        pos = z > 0
        q = np.empty(z.size, dtype=np.float)
        q[pos] = np.log1p(np.exp(-z[pos]))
        q[~pos] = (-z[~pos] + np.log1p(np.exp(z[~pos])))
        return np.mean(q)

    def _grad(self, X, y):
        """Compute gradient of logistic loss for batch

        Args:
            X: training batch data matrix (2D NumPy array)
            y: training batch labels (NumPy vector)

        Calls:
            sigmoid
            self._score

        Length: 2-6 lines

        Returns: tuple of a self.d dimensional vector and a scalar which are
            - gradient of log loss with respect to weights (self.w)
            - gradient of log loss with respect to biases (self.b)
        """
        raise NotImplementedError()

    def _update_weights(self, X, y, alpha):
        """Applies an SGD step by updating self.w and self.b

        Args:
            X: training batch data matrix (2D NumPy array)
            y: training batch labels (NumPy vector)
            alpha: learning rate

        Calls:
            self._grad
            self._loss

        Length: 4-6 lines

        Returns: average batch loss after update
        """
        raise NotImplementedError()

    def predict(self, X):
        """Get positive class probabilities for test data

        Args:
            X: test data matrix (2D NumPy array)

        Length: 1-2 lines

        Calls:
            sigmoid
            self._score

        Returns: NumPy vector of positive class probabilities for test data
        """
        raise NotImplementedError()


class TFLogisticRegression(LogisticRegressionBase):
    def __init__(self, dim):
        super(TFLogisticRegression, self).__init__(dim)
        self.session = tf.Session()
        self._build()

    def _initialize(self):
        """Initialize model variables"""
        self.session.run(tf.global_variables_initializer())

    def _loss(self, h):
        """Get cross entropy loss for data with respect to labels self.y

        Args:
            h: Tensor of logits

        Calls:
            tf.reduce_mean
            tf.nn.sigmoid_cross_entropy_with_logits

        Length: 1-3 lines

        Returns: scalar Tensor of average logistic loss
        """
        raise NotImplementedError()

    def _build(self):
        """Build TensorFlow model graph

        Populates input placeholders (self.X, self.y, self.lr),
        model variables (self.w, self.b),
        and core ops (self.predict_op, self.loss_op, self.train_op)
        """
        # Data placeholders
        self.X = tf.placeholder(tf.float32, (None, self.d))
        self.y = tf.placeholder(tf.float32, (None,))
        self.lr = tf.placeholder(tf.float32)
        # Compute linear "layer"
        with tf.variable_scope('linear'):
            self.w = tf.get_variable('w', (self.d, 1), dtype=tf.float32)
            self.b = tf.get_variable('b', (1,), dtype=tf.float32)
        h = tf.squeeze(tf.nn.bias_add(tf.matmul(self.X, self.w), self.b))
        # Prediction op
        self.predict_op = tf.sigmoid(h)
        # Train op
        self.loss_op = self._loss(h)
        trainer = tf.train.GradientDescentOptimizer(self.lr)
        self.train_op = trainer.minimize(self.loss_op)

    def _update_weights(self, X, y, lr):
        """Applies an SGD step by updating self.w and self.b

        Args:
            X: training batch data matrix (2D NumPy array)
            y: training batch labels (NumPy vector)
            alpha: learning rate

        Calls:
            self.session.run

        Length: 2-3 lines

        Returns: average batch loss after update
        """
        raise NotImplementedError()

    def predict(self, X):
        """Get positive class probabilities for test data

        Args:
            X: test data matrix (2D NumPy array)

        Calls:
            self.session.run

        Length: 1-2 lines

        Returns: NumPy vector of positive class probabilities for test data
        """
        raise NotImplementedError()
