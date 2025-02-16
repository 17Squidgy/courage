from dataclasses import dataclass
from enum import auto
from enum import Enum
from typing import Dict
from typing import Type
import numpy as np



@dataclass
class LearningRate:
    lambda_: float = 1e-3
    s0: float = 1
    p: float = 0.5

    iteration: int = 0

    def __call__(self):
        """
        Calculate learning rate according to lambda (s0/(s0 + t))^p formula
        """
        self.iteration += 1
        return self.lambda_ * (self.s0 / (self.s0 + self.iteration)) ** self.p


class LossFunction(Enum):
    MSE = auto()
    MAE = auto()
    LogCosh = auto()
    Huber = auto()


class BaseDescent:
    """
    A base class and templates for all functions
    """

    def __init__(self, dimension: int, lambda_: float = 1e-3, loss_function: LossFunction = LossFunction.MSE):
        """
        :param dimension: feature space dimension
        :param lambda_: learning rate parameter
        :param loss_function: optimized loss function
        """
        self.w: np.ndarray = np.random.rand(dimension)
        self.lr: LearningRate = LearningRate(lambda_=lambda_)
        self.loss_function: LossFunction = loss_function

    def step(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        return self.update_weights(self.calc_gradient(x, y))

    def update_weights(self, gradient: np.ndarray) -> np.ndarray:
        """
        Template for update_weights function
        Update weights with respect to gradient
        :param gradient: gradient
        :return: weight difference (w_{k + 1} - w_k): np.ndarray
        """
        pass

    def calc_gradient(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Template for calc_gradient function
        Calculate gradient of loss function with respect to weights
        :param x: features array
        :param y: targets array
        :return: gradient: np.ndarray
        """
        pass

    def calc_loss(self, x: np.ndarray, y: np.ndarray) -> float:
        """
        Calculate loss for x and y with our weights
        :param x: features array
        :param y: targets array
        :return: loss: float
        """
        if self.loss_function is LossFunction.MSE:
            return ((self.predict(x) - y)**2).mean()
        elif self.loss_function is LossFunction.LogCosh:
            return np.mean(np.log(np.cosh(self.predict(x) - y)))
        elif self.loss_function is LossFunction.MAE:
            return np.abs(self.predict(x) - y).mean()
        elif self.loss_function is LossFunction.Huber:
            delta = 1.0
            miss = self.predict(x) - y
            return np.mean(np.where(np.abs(miss) < delta, 0.5 * miss ** 2, delta * (np.abs(miss) - 0.5 * delta)))

    def predict(self, x: np.ndarray) -> np.ndarray:
        """
        Calculate predictions for x
        :param x: features array
        :return: prediction: np.ndarray
        """
        return x @ self.w

class VanillaGradientDescent(BaseDescent):
    """
    Full gradient descent class
    """

    def update_weights(self, gradient: np.ndarray) -> np.ndarray:
        """
        :return: weight difference (w_{k + 1} - w_k): np.ndarray
        """
        weight_difference = -self.lr.__call__() * gradient
        self.w = self.w + weight_difference
        return weight_difference

    def calc_gradient(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:

        if self.loss_function is LossFunction.MSE:
            return 2 * (x.T @ (self.predict(x) - y)) / x.shape[0]
        elif self.loss_function is LossFunction.LogCosh:
            return x.T @ np.tanh(self.predict(x) - y) / x.shape[0]
        elif self.loss_function is LossFunction.MAE:
            return (x.T @ np.sign(self.predict(x) - y)) / x.shape[0]
        elif self.loss_function is LossFunction.Huber:
            delta = 1.0 # можно менять в зависимости от задачи
            miss = self.predict(x) - y
            return (x.T @ (np.where(np.abs(miss) < delta, miss, delta * np.sign(miss)))) / x.shape[0]



class StochasticDescent(VanillaGradientDescent):
    """
    Stochastic gradient descent class
    """

    def __init__(self, dimension: int, lambda_: float = 1e-3, batch_size: int = 400,
                 loss_function: LossFunction = LossFunction.MSE):
        """
        :param batch_size: batch size (int)
        """
        super().__init__(dimension, lambda_, loss_function)
        self.batch_size = batch_size

    def calc_gradient(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        batch_indices = np.random.randint(0, x.shape[0], self.batch_size)
        x_batch = x[batch_indices]
        if isinstance(y, np.ndarray):
            y_batch = y[batch_indices]
        else:
            y_batch = y.iloc[batch_indices]

        if self.loss_function is LossFunction.MSE:
            return 2 * (x_batch.T @ (self.predict(x_batch) - y_batch)) / self.batch_size
        elif self.loss_function is LossFunction.LogCosh:
            return x_batch.T @ np.tanh(self.predict(x_batch) - y_batch) / self.batch_size
        elif self.loss_function is LossFunction.MAE:
            return (x_batch.T @ np.sign(self.predict(x_batch) - y_batch)) / x_batch.shape[0]
        elif self.loss_function is LossFunction.Huber:
            delta = 1.0  # можно менять в зависимости от задачи
            miss = self.predict(x_batch) - y_batch
            return (x_batch.T @ (np.where(np.abs(miss) < delta, miss, delta * np.sign(miss)))) / x_batch.shape[0]



class MomentumDescent(VanillaGradientDescent):
    """
    Momentum gradient descent class
    """

    def __init__(self, dimension: int, lambda_: float = 1e-3, loss_function: LossFunction = LossFunction.MSE):
        super().__init__(dimension, lambda_, loss_function)
        self.alpha: float = 0.9

        self.h: np.ndarray = np.zeros(dimension)

    def update_weights(self, gradient: np.ndarray) -> np.ndarray:
        """
        :return: weight difference (w_{k + 1} - w_k): np.ndarray
        """
        self.h = self.alpha * self.h + gradient * self.lr.__call__()
        self.w = self.w - self.h
        return -self.h

class AMSGrad(VanillaGradientDescent):
    """
    AMSGrad descent class
    """

    def __init__(self, dimension: int, lambda_: float = 1e-3, loss_function: LossFunction = LossFunction.MSE):
        super().__init__(dimension, lambda_, loss_function)
        self.eps: float = 1e-8

        self.m: np.ndarray = np.zeros(dimension)
        self.v: np.ndarray = np.zeros(dimension)
        self.v_max: np.ndarray = np.zeros(dimension)

        self.beta_1: float = 0.9
        self.beta_2: float = 0.999

        self.iteration: int = 0

    def update_weights(self, gradient: np.ndarray) -> np.ndarray:
        """
        :return: weight difference (w_{k + 1} - w_k): np.ndarray
        """
        self.iteration += 1
        self.m = self.beta_1 * self.m + (1 - self.beta_1)*gradient
        self.v = self.beta_2 * self.v + (1 - self.beta_2) * np.square(gradient)
        m_ = self.m / (1 - self.beta_1**self.iteration)
        self.v_max = np.maximum(self.v_max, self.v)
        weight_diff = self.lr.__call__() * m_ / (np.sqrt(self.v_max) + self.eps)
        self.w = self.w - weight_diff
        return -weight_diff

class Adam(VanillaGradientDescent):
    """
    Adaptive Moment Estimation gradient descent class
    """

    def __init__(self, dimension: int, lambda_: float = 1e-3, loss_function: LossFunction = LossFunction.MSE):
        super().__init__(dimension, lambda_, loss_function)
        self.eps: float = 1e-8

        self.m: np.ndarray = np.zeros(dimension)
        self.v: np.ndarray = np.zeros(dimension)

        self.beta_1: float = 0.9
        self.beta_2: float = 0.999

        self.iteration: int = 0

    def update_weights(self, gradient: np.ndarray) -> np.ndarray:
        """
        :return: weight difference (w_{k + 1} - w_k): np.ndarray
        """
        self.iteration += 1
        self.m = self.beta_1 * self.m + (1 - self.beta_1) * gradient
        self.v = self.beta_2 * self.v + (1 - self.beta_2) * np.square(gradient)
        m_ = self.m / (1 - self.beta_1 ** self.iteration)
        v_ = self.v / (1 - self.beta_2 ** self.iteration)
        weight_diff = self.lr.__call__() * m_ / (np.sqrt(v_) + self.eps)
        self.w = self.w - weight_diff
        return -weight_diff

class BaseDescentReg(BaseDescent):
    """
    A base class with regularization
    """

    def __init__(self, *args, mu: float = 0, **kwargs):
        """
        :param mu: regularization coefficient (float)
        """
        super().__init__(*args, **kwargs)

        self.mu = mu

    def calc_gradient(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Calculate gradient of loss function and L2 regularization with respect to weights
        """
        l2_gradient: np.ndarray = self.w.copy()
        l2_gradient[-1] = 0

        return super().calc_gradient(x, y) + l2_gradient * self.mu


class VanillaGradientDescentReg(BaseDescentReg, VanillaGradientDescent):
    """
    Full gradient descent with regularization class
    """


class StochasticDescentReg(BaseDescentReg, StochasticDescent):
    """
    Stochastic gradient descent with regularization class
    """


class MomentumDescentReg(BaseDescentReg, MomentumDescent):
    """
    Momentum gradient descent with regularization class
    """


class AdamReg(BaseDescentReg, Adam):
    """
    Adaptive gradient algorithm with regularization class
    """

class AMSGradReg(BaseDescentReg, AMSGrad):
    """
    Adaptive gradient algorithm with regularization class
    """


def get_descent(descent_config: dict) -> BaseDescent:
    descent_name = descent_config.get('descent_name', 'full')
    regularized = descent_config.get('regularized', False)

    descent_mapping: Dict[str, Type[BaseDescent]] = {
        'full': VanillaGradientDescent if not regularized else VanillaGradientDescentReg,
        'stochastic': StochasticDescent if not regularized else StochasticDescentReg,
        'momentum': MomentumDescent if not regularized else MomentumDescentReg,
        'adam': Adam if not regularized else AdamReg,
        'amsgrad': AMSGrad if not regularized else AMSGradReg
    }

    if descent_name not in descent_mapping:
        raise ValueError(f'Incorrect descent name, use one of these: {descent_mapping.keys()}')

    descent_class = descent_mapping[descent_name]

    return descent_class(**descent_config.get('kwargs', {}))
