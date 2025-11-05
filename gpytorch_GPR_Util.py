#############################################################################
# gpytorch_GPR_Util.py
#
# Main support for gpytorch_GPR.py based on the following classes:
# - GPRegressionModel(), used to create GPR models applied to classification.
# - EarlyStopping(), used to determine when to stop training.
#
# Version: 1.0, Date: 23 October 2025, refactored gpytorch_GPR.py, which
#                     simplifies the understanding of Gaussian Process code.
# Version: 1.2, Date: 30 October 2025, support for approximate inference --
#                     see class GPRegressionModel_Approx().
# Contact: hcuayahuitl@lincoln.ac.uk
#############################################################################

import torch
import gpytorch
from gpytorch.models import ApproximateGP
from gpytorch.variational import VariationalStrategy
from gpytorch.variational import CholeskyVariationalDistribution
from sklearn.cluster import KMeans

# Gaussian Process regressor using the Radial Basis Function kernel (others possible)
class GPRegressionModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood, GP_METHOD2EMPLOY, K):
        super(GPRegressionModel, self).__init__(train_x, train_y, likelihood)

        self.mean_module = gpytorch.means.ConstantMean()

        if GP_METHOD2EMPLOY == 'GPR':
            self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())
            #self.covar_module = gpytorch.kernels.MaternKernel(nu=0.5) # matern12
            #self.covar_module = gpytorch.kernels.MaternKernel(nu=1.5) # matern32
            #self.covar_module = gpytorch.kernels.MaternKernel(nu=2.5) # matern52
            #self.covar_module = gpytorch.kernels.RQKernel(alpha=1.0)

        elif GP_METHOD2EMPLOY == 'GPR_Sparse': # reasonable values of K={10, 20, 100, 200,..., 500}
            kmeans = KMeans(n_clusters=K, init='k-means++', random_state=0).fit(train_x.cpu().numpy())
            inducing_points = torch.tensor(kmeans.cluster_centers_, dtype=torch.float).to(train_x.device)
            self.base_covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())
            self.covar_module = gpytorch.kernels.InducingPointKernel(self.base_covar_module, 
                                                                     inducing_points=inducing_points, 
                                                                     likelihood=likelihood)

        else:
            print("UNKNOWN GP_METHOD2EMPLOY="+str(GP_METHOD2EMPLOY))
            exit(0)

        # parameters for *learnt* mean and variance scaling
        self.a = torch.nn.Parameter(torch.tensor(1.0)) # mean scaler
        self.b = torch.nn.Parameter(torch.tensor(0.0)) # bias
        self.c = torch.nn.Parameter(torch.tensor(torch.pi/8)) # variance scaler

    def forward(self, x):
        mean = self.mean_module(x)
        covariance = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean, covariance)
    
    def predict_probability(self, x):
        self.eval() # this line causes a warning which we can ignore
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            pred = self.likelihood(self(x)) 
            pred_mean = pred.mean
            pred_variance = pred.variance

        # preventing divisions by zero
        self.c.data = torch.clamp(self.c.data, min=1e-6, max=20.0)

        # MacKay's approximation with scaled mean and variance for probabilities
        numerator = pred_mean*self.a + self.b 
        denominator = torch.sqrt(1 + (torch.pi*pred_variance*self.c) / 8)
        prob = torch.sigmoid(numerator/denominator)
            
        return prob


# Determines when to finish training earlier than the full set of epochs (training iterations)
class EarlyStopping:
    def __init__(self, patience=3):
        self.patience = patience
        self.best_loss = float('inf')
        self.patience_count = 0

    def check_early_stopping(self, i, loss_value, stop=False):
        if i % 10 == 0:
            if loss_value < self.best_loss-1e-4:
                self.best_loss = loss_value
                self.patience_count = 0
                improved = True
                stop =  False
            else:
                self.patience_count += 1
                improved = False
                stop = True if self.patience_count >= self.patience else False

            print("Iteration %s, improved=%s: Loss=%s" % (i+1, improved, loss_value))

            if stop:
                print("Early termination at iteration %s! ... patience=%s" %(i, self.patience_count))

        return stop
    
    def reset(self):
        self.best_loss = float('inf')
        self.patience_count = 0


# GPR using approximate inference -- for scaling up to large data, based on (Titsias, 2009, AISTATS)
class GPRegressionModel_VariationalSparse(ApproximateGP):
    def __init__(self, inducing_points):
        number_of_inducing_points = inducing_points.size(0) # first dimension of the tensor (num rows)
        variational_distribution = CholeskyVariationalDistribution(number_of_inducing_points)
        variational_strategy = VariationalStrategy(
            self,
            inducing_points,
            variational_distribution,
            learn_inducing_locations=True
        )
        
        super(GPRegressionModel_VariationalSparse, self).__init__(variational_strategy)
        
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())

        # parameters for *learnt* mean and variance scaling
        self.a = torch.nn.Parameter(torch.tensor(1.0)) # mean scaler
        self.b = torch.nn.Parameter(torch.tensor(0.0)) # bias
        self.c = torch.nn.Parameter(torch.tensor(torch.pi/8)) # variance scaler

    def forward(self, x):
        mean = self.mean_module(x)
        covar = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean, covar)

    def predict_probability(self, x, likelihood):
        self.eval() # this line causes a warning which we can ignore
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            pred = likelihood(self(x)) 
            pred_mean = pred.mean
            pred_variance = pred.variance

        # preventing divisions by zero
        self.c.data = torch.clamp(self.c.data, min=1e-6, max=20.0)

        # MacKay's approximation with scaled mean and variance for probabilities
        numerator = pred_mean*self.a + self.b 
        denominator = torch.sqrt(1 + (torch.pi*pred_variance*self.c) / 8)
        prob = torch.sigmoid(numerator/denominator)
            
        return prob
