from abc import abstractmethod
import math


class ActivationBase:
    @classmethod
    @abstractmethod
    def calc(cls, x):
        pass

    @classmethod
    @abstractmethod
    def derivative(cls, x):
        pass


class ActivationTransparent(ActivationBase):
    @classmethod
    def calc(cls, x):
        return x

    @classmethod
    def derivative(cls, x):
        return 1


class ActivationRelu(ActivationBase):
    @classmethod
    def calc(cls, x):
        return max(0, x)

    @classmethod
    def derivative(cls, x):
        return 0 if x < 0 else 1


class ActivationSigmoid(ActivationBase):
    @classmethod
    def calc(cls, x):
        return 1 / (math.exp(-x) + 1)

    @classmethod
    def derivative(cls, x):
        return cls.calc(x) * (1 - cls.calc(x))
