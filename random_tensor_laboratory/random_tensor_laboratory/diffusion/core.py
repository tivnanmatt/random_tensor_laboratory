import torch

class DiffusionModel(torch.nn.Module):
    def __init__(self,
                 stochastic_differential_equation,
                 neural_network):
        """
        This is an abstract base class for diffusion models.

        It inherits from torch.nn.Module.
        
        It requires the methods sample_x_t_given_x_0 and sample_x_t_minus_dt_given_x_t to be implemented.

        parameters:
            None
        """

        assert isinstance(stochastic_differential_equation, StochasticDifferentialEquation)
        assert isinstance(neural_network, torch.nn.Module)

        super(DiffusionModel, self).__init__()
    
    def forward(self, x_0: torch.Tensor, t: torch.Tensor):
        """
        This method implements the forward pass of the linear operator, i.e. the matrix-vector product.

        parameters:
            x: torch.Tensor 
                The input tensor to the linear operator.
        returns:
            result: torch.Tensor of shape [batch_size, num_channel, *output_shape]
                The result of applying the linear operator to the input tensor.
        """

        return self.sample_x_t_given_x_0(x_0, t)
    
    def sample_x_t_given_x_0(self, x_0: torch.Tensor, t: torch.Tensor):
        """
        This method samples x_t given x_0.

        parameters:
            x_0: torch.Tensor 
                The initial condition.
            t: float
                The time step.
        returns:
            x_t: torch.Tensor
                The sample at time t.
        """

        return self.stochastic_differential_equation.sample_x_t_given_x_0(x_0, t)
    
    def sample_x_t_plus_dt_given_x_t(self, x_t: torch.Tensor, t: torch.Tensor, dt: torch.Tensor):
        return self.stochastic_differential_equation.sample_x_t_plus_dt_given_x_t(x_t, t, dt)
    
    


    
class ScoreBasedDiffusionModel(DiffusionModel):
    def __init__(self,
                 stochastic_differential_equation,
                 score_estimator_neural_network,
                 ):
        """
        This is an abstract base class for score-based diffusion models.

        It inherits from DiffusionModel.
        
        It requires the method sample_x_t_minus_dt_given_x_t to be implemented.

        parameters:
            None
        """

        assert isinstance(score_estimator_neural_network, torch.nn.Module)

        super(ScoreBasedDiffusionModel, self).__init__(stochastic_differential_equation, score_estimator_neural_network)
    
    def sample_x_t_minus_dt_given_x_t(self, x_t: torch.Tensor, t: torch.Tensor, dt: torch.Tensor):
        """
        This method samples x_t_minus_dt given x_t.

        parameters:
            x_t: torch.Tensor 
                The sample at time t.
            t: float
                The time step.
            dt: float
                The time increment.
        returns:
            x_t_minus_dt: torch.Tensor
                The sample at time t - dt.
        """

        return self.stochastic_differential_equation.sample_x_t_minus_dt_given_x_t_and_score_estimator(self.score_estimator_neural_network, x_t, t, dt)




class StochasticDifferentialEquation(torch.nn.Module):
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

    def sample_x_t_plus_dt_given_x_t(self, x_t, t, dt):
        """
        This method samples x_t_plus_dt given x_t.

        parameters:
            x_t: torch.Tensor 
                The sample at time t.
            t: float
                The time step.
            dt: float
                The time increment.
        returns:
            x_t_plus_dt: torch.Tensor
                The sample at time t + dt.
        """

        if isinstance(t, float):
            t = torch.tensor(t, dtype=x_t.dtype, device=x_t.device)

        if isinstance(dt, float):
            dt = torch.tensor(dt, dtype=x_t.dtype, device=x_t.device)

        dw_t = torch.randn_like(x_t) * torch.sqrt(dt)

        dx_t = self.f(x_t, t) * dt + self.G(x_t, t) @ dw_t

        return x_t + dx_t
    
    def sample_x_t_minus_dt_given_x_t_and_score_estimator(self, s, x_t, t, dt):
        """
        This method samples x_t_minus_dt given x_t.

        parameters:
            s: callable
                The score estimator of the SDE. It should take x and t as input and return a tensor of the same shape as x.
            x_t: torch.Tensor 
                The sample at time t.
            t: float
                The time step.
            dt: float
                The time increment.

        returns:
            x_t_minus_dt: torch.Tensor
                The sample at time t - dt.
        """
            
        if isinstance(t, float):
            t = torch.tensor(t, dtype=x_t.dtype, device=x_t.device)

        if isinstance(dt, float):
            dt = torch.tensor(dt, dtype=x_t.dtype, device=x_t.device)

        dw_t = torch.randn_like(x_t) * torch.sqrt(dt)

        dx_t = (self.f(x_t, t) - s(x_t,t)) * (-dt) + self.G(x_t, t) @ dw_t

        return x_t + dx_t
    
    def sample_x_t_given_x_0(self, x_0, t, num_steps=128):
        """
        This method samples x_t given x_0.

        parameters:
            x_0: torch.Tensor 
                The initial condition.
            t: float
                The time step to sample from
            num_steps: int
                The number of steps to take in the Euler-Maruyama forward sampling scheme.

        returns:
            x_t: torch.Tensor
                The sample at time t.
        """

        if isinstance(t, float):
            t = torch.tensor(t, dtype=x_0.dtype, device=x_0.device)

        dt = t / num_steps

        x_t = x_0
        for _ in range(num_steps):
            x_t = self.sample_x_t_plus_dt_given_x_t(x_t, t, dt)
            t = t + dt
        
        return x_t
    
