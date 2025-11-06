#############################################################################
# gpytorch_GPR.py
#
# Implements functionality for Gaussian Process classification via GP regression (GPR).
# It is powered by GPyTorch: https://docs.gpytorch.ai/en/stable/
#
# This program is a baseline classifier with class predictions derived from
# estimated mean vector and covariance matrices of a GPR using a normalised ratio.
# 
# This program supports Approximate inference via maximising the Evidence Lower 
# Bound (ELBO) to optimise kernel, likelihood, and inducing points.
#
# This program can run on CPU or GPU devices, as detected at runtime.
#
# Version: 1.0, Date: 25 October 2024, functionality tested on multiple datasets
#               for binary classification -- and coupled with ModelEvaluator.py 
# Version: 1.1, Date: 26 October 2024, support for plotting data in 3D.
# Version: 1.3, Date: 30 October 2025, support for GPR with approximate inference.
# Contact: hcuayahuitl@lincoln.ac.uk
#############################################################################

import sys
import time
import torch
import gpytorch
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from gpytorch_GPR_Util import GPRegressionModel
from gpytorch_GPR_Util import EarlyStopping
from gpytorch_GPR_Util import GPRegressionModel_VariationalSparse
from sklearn.cluster import KMeans


# Gaussian Process regressor applied to classification tasks
class GPR():
    device = None
    MAX_TRAIN_DATA=0 # only needed for full GPs
    STANDARDISE_DATA = True
    LEARNING_RATE = 0.10
    MAX_NUM_EPOCHS = 1000
    GP_METHOD2EMPLOY = 'GPR_VarSparse' # choices are 'GPR_Sparse', 'GPR_VarSparse'
    NUM_INDUCING_POINTS= 20 # for GPR_Sparse and GPR_VarSparse
    VERBOSE = False
    SILENT = False # Set to True to suppress all print statements

    def __init__(self, datafile_train, datafile_test, STANDARDISE_DATA=True, GP_METHOD2EMPLOY='GPR_VarSparse', NUM_INDUCING_POINTS=20):
        
        # Update parameters
        self.STANDARDISE_DATA = STANDARDISE_DATA
        self.GP_METHOD2EMPLOY = GP_METHOD2EMPLOY
        self.NUM_INDUCING_POINTS = NUM_INDUCING_POINTS

        # Load training and test data from two separate CVS files
        X_train, Y_train = self.load_data(datafile_train, False) # False to use all training data
        X_test, Y_test = self.load_data(datafile_test, False) # False to use the entire test set
        X_train, X_test = self.get_standardised_data(X_train, X_test)
        print("%s training instances" % (len(X_train)))
        print("%s test instances" % (len(X_test)))
        print("%s inducing points" % (self.NUM_INDUCING_POINTS))

        # Convert the data to PyTorch tensors
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        X_train = torch.Tensor(X_train).to(self.device)
        Y_train = torch.Tensor(Y_train.values).to(self.device)
        X_test = torch.Tensor(X_test)
        Y_test = torch.Tensor(Y_test.values)

        # train GP model via regression and evaluate it with test data
        model, likelihood, training_time = self.train_GPR(X_train, Y_train)
        self.model, self.likelihood, self.results = self.evaluate_GPR(
            X_test, Y_test, model, likelihood, training_time, 
            model_name="GPR", return_predictions=True
        )

    def load_data(self, csv_file, useDataSampling_NotFullSet=False):
        print("LOADING and PROCESSING data...")
        df = pd.read_csv(csv_file, encoding='latin')
        X = df.iloc[:, :-1]  
        Y = df.iloc[:, -1]   
        if useDataSampling_NotFullSet and len(X)>self.MAX_TRAIN_DATA:
            random_indices = np.random.choice(X.shape[0], self.MAX_TRAIN_DATA, replace=False)
            X = X.iloc[random_indices]
            Y = Y.iloc[random_indices]
        return X, Y

    # standardize the features 
    def get_standardised_data(self, X_train, X_test):
        print("STANDARDISE_DATA=%s" % (self.STANDARDISE_DATA))
        if self.STANDARDISE_DATA:
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)
        else:
            X_train = X_train.values
            X_test = X_test.values
        return X_train, X_test

    # returns the probability density of a univariate Gaussian
    def get_gaussian_probability_density(self, x, mean, var):
        e_val = -np.power((x-mean), 2)/(2*var)
        return (1/(np.sqrt(2*np.pi*var))) * np.exp(e_val)
    
    # print the optimised parameters after training
    def print_learnt_parameters(self, model, likelihood):
        print("\nGaussian Process learnt parameters (after optimisation):")
        if self.GP_METHOD2EMPLOY == 'GPR_Sparse':
            print("Lengthscale=%s" % (model.covar_module.base_kernel.base_kernel.lengthscale.item()))
            print("Outputscale=%s" % (model.covar_module.base_kernel.outputscale.item()))

        else: # in the case of 'GPR_VarSparse'
            print("Lengthscale=%s" % (model.covar_module.base_kernel.lengthscale.item()))
            print("Outputscale=%s" % (model.covar_module.outputscale.item()))
        
        if self.GP_METHOD2EMPLOY == 'GPR_VarSparse':
            print("Noise=%s" % (likelihood.noise.item()))

            # additional parameters learnt by the Variational Sparse GP
            Z = model.variational_strategy.inducing_points
            f_Z = model(Z)
            V_m = f_Z.mean
            V_c = f_Z.covariance_matrix
            q_u = model.variational_strategy.variational_distribution
            p_u = model.variational_strategy.prior_distribution
            print("Inducing Points Z=%s" % (Z))
            print("V_m=%s" % (V_m))
            print("V_c=%s" % (V_c))
            print("q_u (mean)=%s" % (q_u.mean))
            print("q_u (covariance)=%s" % (q_u.covariance_matrix))
            print("p_u (mean)=%s" % (p_u.mean))
            print("p_u (covariance)=%s" % (p_u.covariance_matrix))

        else:
            print("Noise=%s" % (model.likelihood.noise.item()))

    # training procedure for the GPR
    def train_GPR(self, X_train, y_train):
        print("\nTRAINING Gaussian Process model...")
        training_time = time.time()

        # Initialise the likelihood and model. Whilst the former defines the noise, 
        # the later is used to learn the mean vector and covariance matrix. 
        likelihood = gpytorch.likelihoods.GaussianLikelihood().to(self.device)
        es = EarlyStopping() # to interrupt training when convergence has been reached

        # the model is created based on the type of GP selected in GP_METHOD2EMPLOY
        if self.GP_METHOD2EMPLOY == 'GPR_Sparse': 
            model = GPRegressionModel(X_train, y_train, likelihood, self.GP_METHOD2EMPLOY, self.NUM_INDUCING_POINTS).to(self.device)

            # optimiser for the GP model via the MLL loss function
            print("Loss function: Marginal Log Likelihood (MLL)")
            optimiser1 = torch.optim.Adam(model.parameters(), lr=self.LEARNING_RATE)  
            mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

        if self.GP_METHOD2EMPLOY == 'GPR_VarSparse': 
            kmeans = KMeans(n_clusters=self.NUM_INDUCING_POINTS, init='k-means++', random_state=0).fit(X_train.cpu().numpy())
            inducing_points = torch.tensor(kmeans.cluster_centers_, dtype=torch.float).to(X_train.device)
            model = GPRegressionModel_VariationalSparse(inducing_points).to(self.device)

            # optimiser for the GP model via the MLL loss function
            # note that the likelihood is included in the optimiser because is not part of the model as in 'GPR'
            print("Loss function: Evidence Lower Bound (ELBO)")
            parameters = [{'params': model.parameters()}, {'params': likelihood.parameters()}]
            optimiser1 = torch.optim.Adam(parameters, lr=self.LEARNING_RATE/2) # smaller lr than Exact GP
            mll = gpytorch.mlls.VariationalELBO(likelihood, model, num_data=X_train.size(0))

        # Set the model in training mode
        model.train() # MultivariateNormal with mean and covariance/kernel
        likelihood.train() #MultivariateNormal with mean and covariance+noise

        # Training loop for the GPR model using gradient descent on negative MLL
        # MLL (marginal log likelihood): log P(y|X,theta), where theta includes
        # the parameters Lengthscale, Outputscale and Noise -- among others in the case of VSGPs.
        for i in range(self.MAX_NUM_EPOCHS):
            optimiser1.zero_grad()
            output = model(X_train)
            loss = -mll(output, y_train)
            loss.backward()
            optimiser1.step()
            if es.check_early_stopping(i, loss.item()):
                break

        if self.VERBOSE:
            self.print_learnt_parameters(model, likelihood) # once tranining of model & likelihood is done

        training_time = time.time() - training_time
        return model, likelihood, training_time

    
    def evaluate_GPR(self, X_test, Y_test, model, likelihood, training_time, model_name="GPR", return_predictions=False):
        """
        Evaluates the trained GP model on test data and computes comprehensive metrics.
        
        Args:
            X_test: Test input features
            Y_test: True test labels (binary: 0 or 1)
            model: Trained GP model
            likelihood: Trained likelihood function
            training_time: Time taken to train the model (in seconds)
            model_name: Name identifier for the model (default: "GPR")
            return_predictions: If True, includes predictions in the results dictionary
        
        Returns:
            tuple: (model, likelihood, results_dict) where results_dict contains:
                - Discrimination metrics: Balanced Accuracy, F1 Score, AUC
                - Calibration metrics: Brier Score, KL Divergence, Expected Calibration Loss
                - Timing information: Training Time, Inference Time
                - Optional: Y_pred (binary predictions), Y_prob (probability predictions)
        """
        from sklearn.metrics import balanced_accuracy_score, f1_score, roc_auc_score, brier_score_loss
        
        print("\nEVALUATING Gaussian Process model...")
        test_time = time.time()
        
        # Set model and likelihood to evaluation mode (disables dropout, batchnorm, etc.)
        model.eval()
        likelihood.eval()

        # Prepare ground truth labels
        Y_true = Y_test.cpu().numpy() if torch.is_tensor(Y_test) else Y_test.values
        Y_pred = []  # Binary predictions (0 or 1)
        Y_prob = []  # Probability predictions P(y=1|x)
        
        # INFERENCE LOOP: Generate predictions for each test instance
        for values in X_test:
            test_case = torch.Tensor(np.array([values])).to(self.device)
            
            # Compute probability using normalised likelihood ratio:
            # P(y=1|x) = p(1|μ,σ²) / [p(1|μ,σ²) + p(0|μ,σ²)]
            # where p(y|μ,σ²) is the Gaussian probability density at y given posterior mean μ and variance σ²
            with torch.no_grad(), gpytorch.settings.fast_pred_var():
                predictions = likelihood(model(test_case))
                pred_mean = predictions.mean.item()      # Posterior mean μ
                pred_var = predictions.variance.item()   # Posterior variance σ²
                
                # Calculate Gaussian PDF at y=1 and y=0
                pdf_1 = self.get_gaussian_probability_density(1, pred_mean, pred_var)
                pdf_0 = self.get_gaussian_probability_density(0, pred_mean, pred_var)
                
                # Normalised ratio gives P(y=1|x)
                prob = pdf_1 / (pdf_1 + pdf_0)

            Y_prob.append(prob)
            Y_pred.append(np.round(prob))  # Threshold at 0.5 for binary prediction

        test_time = time.time() - test_time
        
        # Convert to numpy arrays
        Y_pred = np.array(Y_pred)
        Y_prob = np.array(Y_prob)
        
        # Calculate metrics
        bal_acc = balanced_accuracy_score(Y_true, Y_pred)
        f1 = f1_score(Y_true, Y_pred)
        auc = roc_auc_score(Y_true, Y_prob)
        brier = brier_score_loss(Y_true, Y_prob)
        
        # KL Divergence
        epsilon = 1e-10
        Y_prob_safe = np.clip(Y_prob, epsilon, 1 - epsilon)
        kl_div = np.mean(Y_true * np.log(Y_true / Y_prob_safe + epsilon) + 
                        (1 - Y_true) * np.log((1 - Y_true) / (1 - Y_prob_safe) + epsilon))
        
        # Expected Calibration Error
        n_bins = 10
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]
        
        ec_loss = 0.0
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            in_bin = np.logical_and(Y_prob >= bin_lower, Y_prob < bin_upper)
            prop_in_bin = np.mean(in_bin)
            if prop_in_bin > 0:
                accuracy_in_bin = np.mean(Y_true[in_bin] == Y_pred[in_bin])
                avg_confidence_in_bin = np.mean(Y_prob[in_bin])
                ec_loss += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
        
        # Create results dictionary
        results = {
            "Model": model_name,
            "Balanced Accuracy": round(bal_acc, 4),
            "F1 Score": round(f1, 4),
            "AUC": round(auc, 4),
            "Brier Score": round(brier, 4),
            "KL Divergence": round(float(kl_div), 4),
            "Expected Calibration Loss": round(float(ec_loss), 4),
            "Training Time (s)": f"{round(training_time, 4)} s",
            "Inference Time (s)": f"{round(test_time, 4)} s"
        }
        
        # Optionally include predictions
        if return_predictions:
            results["Y_pred"] = Y_pred
            results["Y_prob"] = Y_prob
        
        return model, likelihood, results
