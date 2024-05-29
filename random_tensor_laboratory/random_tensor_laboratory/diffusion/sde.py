
class StochasticDifferentialEquation:
    def __init__(self, f, G):
        """
        This class implements a stochastic differential equation (SDE) of the form dx = f(x, t) dt + G(x, t) dw
        where f is a vector-valued function representing the drift term and G is a matrix-valued function representing the gaussian noise rates.


        parameters:
            f: callable
                The drift term of the SDE. It should take x and t as input and return a tensor of the same shape as x.
            G: callable
                The diffusion term of the SDE. It should take x and t as input and return a rtl.LinearOperator that can act on a tensor of the same shape as x.
        """

        self.f = f
        self.G = G