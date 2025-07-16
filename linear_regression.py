import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px 
import pickle

class LinearRegression:
    """Linear regression model with gradient descent
    parameters:
        learning_rate: float -> the learning rate used in gradient descent.
        convergence_tol: float -> the tolerance for covergence (stopping criterion). = 1e-6
    attributes: 
        w: np.ndarray -> the coefficients(weights) of the independent variable.
        b: float -> intercept(bias) of the regression model
    
    methods:
        initialize_parameter()
        forward_propagation()
        compute_cost()
        backward_propagation()
        predict()
        save_model()
        load_model()
    """
    
    def __init__(self,learning_rate:float,convergence_tol:float=1e-6) -> None:
        self.learning_rate = learning_rate
        self.convergence_tol = convergence_tol
        self.w: np.ndarray = None
        self.b: float = None
        pass
    
    def initialize_parameter(self,n_features):
        """Initialize the model parameters

        Args:
            n_features (int): the number of features in the input data
        """
        self.w = np.random.randn(n_features) * 0.01
        self.b = 0
        
    def forward_propagation(self,X):
        """Compute the forward pass of the linear regression model

        Args:
            X (np.ndarray): input data of shape(m,n_features), m = number of observations
            
        Returns:
            np.ndarray: Y' (prediction of shape(m,))
        """
        return np.dot(X,self.w) + self.b
    
    def compute_cost(self,predictions):
        """compute the mean squared error cost

        Args:
            prediction (numpy.ndarray): prediction
        """
        m = len(predictions)
        cost = np.sum(np.square(predictions - self.y)) / (2*m)
        
        return cost
    
    def backward(self,predictions):
        """Compute gradient for model parameters

        Args:
            predictions (numpy.ndarray): predictions of shape (m,)
        """
        m = len(predictions)
        self.dW = np.dot(predictions - self.y, self.X) / m
        self.db = np.sum(predictions - self.y)
        
        
    def fit(self,X,y,iterations,plot_cost=True):
        """Fit the linear regression model to the training data

        Args:
            X (numpy.ndarray): Training input data of shape (m,n_features)
            y (numpy.ndarray): Training lebels of shape (m,)
            iterations (int): the nunmber of iterations for gradient descent.
            plot_cost (bool, optional): Whether to plot the cost during the training. Defaults to True.
        """
        assert isinstance(X,np.ndarray), "X must be a Numpy array"
        assert isinstance(y, np.ndarray), "y must be a Numpy array"
        assert X.shape[0] == y.shape[0], "X and y must have the same number of observations"
        assert iterations > 0, "Iterations must be greater than 0"
        
        self.X = X
        self.y = y
        self.initialize_parameter(X.shape[1])
        costs = []
        
        for i in range(iterations):
            predictions = self.forward_propagation(X)
            cost = self.compute_cost(predictions)
            self.backward(predictions)
            
            self.w -= self.learning_rate * self.dW
            self.b -= self.learning_rate * self.db
            
            costs.append(cost)
            
            if i%100 == 0:
                print(f'Iteration: {i}, Cost: {cost}')
            if i > 0 and abs(costs[-1]-costs[-2]) < self.convergence_tol:
                print(f'Converged after {i} iterations.')
                break
            
        if plot_cost:
            fig = px.line(
                y=costs,
                title="Cost vs Iteration",
            )
            fig.update_layout(
                title_font_color = "green",
                xaxis = dict(color='blue',title='Iteration'),
                yaxis = dict(color='blue',title='Cost')
            )
                
            fig.show()
                
    
    def predict(self,X):
        """Predict target values for new input data.

        Args:
            X (numpy.ndarray): Input data of shape(m,)
        """
        return self.forward_propagation(X)
    
    def save_model(self,filename=None):
        """Save the trained model to a file using pickle

        Args:
            filename (Str, optional): Name of the file to save the model to. Defaults to None.
        """
        model_data = {
            'learning_rate' : self.learning_rate,
            'convergence_tol' : self.convergence_tol,
            'W' : self.w,
            'b' : self.b
        }
        with open(filename,'wb') as file:
            pickle.dump(model_data,file)
            
    
    @classmethod
    def load_model(cls,filename):
        """Load a trained model from a file using pickle.
    

        Args:
            filename (str): The name of the file to load the model from.
            
        Returns:
            LinearRegression: An instance of the LinearRegression class with the loaded parameters.
        """
        
        with open(filename,'rb') as file:
            model_data = pickle.load(file)
            
        loaded_model = cls(model_data['learning_rate'],model_data['convergence_tol'])
        loaded_model.W = model_data['W']
        loaded_model.b = model_data['b']
        
        return loaded_model
    
    
class RegressionMetrics:
    """different evaluation metrics used for regression problem.
    """
    @staticmethod
    def mean_squared_error(y_true,y_pred):
        """Calculate the Mean Squared Error(MSE)

        Args:
            y_true (numpy.ndarray): the true target values.
            y_pred (numpy.ndarray): the predicted target values 
        """
        
        assert len(y_true) == len(y_pred), "Miss match of the input arrays"
        
        mse = np.mean(np.square(y_true - y_pred))
        
        return mse
    
    @staticmethod
    def root_mean_squared_error(y_true,y_pred):
        """calculate root mean squared 

        Args:
            y_true (np.ndarray): the true target value
            y_pred (np.ndarray): the predicted target value
        """
        
        assert len(y_true) == len(y_pred),  "Miss match of the input arrays"
        
        mse = RegressionMetrics.mean_squared_error(y_true,y_pred)
        rmse = np.sqrt(mse)
        
        return rmse
    
    @staticmethod
    def r_squared(y_true,y_pred):
        """Calculate the R-squared, coefficient of determination.

        Args:
           y_true (np.ndarray): the true target value
           y_pred (np.ndarray): the predicted target value
        """
        assert len(y_true) == len(y_pred),  "Miss match of the input arrays"
        
        mean_y = np.mean(y_true)
        ss_total = np.sum(np.square(y_true - mean_y))
        ss_residual = np.sum(np.square(y_true,y_pred))
        
        r2 = 1 - (ss_residual/ss_total)
        
        return r2
