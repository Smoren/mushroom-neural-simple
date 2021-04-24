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


class ActivationConst(ActivationBase):
    @classmethod
    def calc(cls, x):
        return 1

    @classmethod
    def derivative(cls, x):
        return 0


class ActivationRelu(ActivationBase):
    @classmethod
    def calc(cls, x):
        return max(0, x)

    @classmethod
    def derivative(cls, x):
        return 0 if x < 0 else 1


class ActivationLRelu(ActivationBase):
    @classmethod
    def calc(cls, x):
        return x if x >= 0 else 0.01*x

    @classmethod
    def derivative(cls, x):
        return 1 if x >= 0 else 0.01


class ActivationElu(ActivationBase):
    @classmethod
    def calc(cls, x, alpha=1):
        return x if x >= 0 else alpha*math.exp(x) - 1

    @classmethod
    def derivative(cls, x, alpha=1):
        return 1 if x >= 0 else cls.calc(x, alpha)+alpha


class ActivationSigmoid(ActivationBase):
    @classmethod
    def calc(cls, x):
        return 1 / (math.exp(-x) + 1)

    @classmethod
    def derivative(cls, x):
        return cls.calc(x) * (1 - cls.calc(x))


class ActivationGauss(ActivationBase):
    @classmethod
    def calc(cls, x):
        return math.exp(-x**2)

    @classmethod
    def derivative(cls, x):
        return -2*x*math.exp(-x**2)


class ActivationSinc(ActivationBase):
    @classmethod
    def calc(cls, x):
        return 1 if x == 0 else math.sin(x)/x

    @classmethod
    def derivative(cls, x):
        return 0 if x == 0 else math.cos(x)/x - math.sin(x)/(x**2)


class ActivationSoftPlus(ActivationBase):
    @classmethod
    def calc(cls, x):
        return math.log(1+math.exp(x))

    @classmethod
    def derivative(cls, x):
        return 1/(1+math.exp(-x))


class ActivationSoftSign(ActivationBase):
    @classmethod
    def calc(cls, x):
        return x/(1+abs(x))

    @classmethod
    def derivative(cls, x):
        return 1/((1+abs(x))**2)


class ActivationSoftSignSquare(ActivationBase):
    @classmethod
    def calc(cls, x):
        return x/(1+abs(x**2))

    @classmethod
    def derivative(cls, x):
        return (1 - x**2)/((1 + x**2)**2)
