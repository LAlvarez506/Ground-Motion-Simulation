# -*- coding: utf-8 -*-
"""
Created on Fri Oct 11 17:04:50 2019

@author: A35479
"""

"""
28/10/19:
Bayesian optimization for N parameters of a function.  Maximizes of minimizes a cost function based on 
Gaussian processes.  Optimal solution is found for a given number of iterations.  Parallel processing is employed to speed up computations
"""
import os
from BO_output import bo_output
from BO_IMs import *
from Methods import GMSM
from EQ_Otarola import EQ
#from sim_parallel import simulation
import multiprocessing
import numpy as np
import math as m
import time
import pickle
#############################################
import sklearn.gaussian_process as gp
from sklearn.linear_model import LinearRegression
from scipy.stats import norm
from scipy.optimize import minimize
#############################################
Methods = GMSM()

class calibrate(object):
    
    freqs = np.linspace(0.01, 24., 20)
    periods = np.concatenate((np.linspace(0., 0.1, 11), np.linspace(0.2, 1., 9), np.linspace(1.5, 5., 8)))
    damping = 0.05

    def __init__(self, observed, input_parameters, cal_metric, cal_variables, n_sim=5,
                 duration_boolean=False, source_file=None, point_source_boolean=True, iterations=5,
                 nu=0.5, n_processors=4, id=None, fit_range=None, n_pre_samples=None, message=None,
                 n_not_improv_max=100, print_stats=True):
        
        # ------------------------------------- #
        ## - ENTRY DATA
        self.observed = observed
        self.input_parameters = input_parameters
        self.cal_metric = cal_metric
        self.cal_variables = cal_variables
        self.n_sim = n_sim
        self.duration_boolean = duration_boolean
        self.source_file = source_file
        self.point_source_boolean = point_source_boolean
        self.iterations = iterations
        self.nu = nu
        self.n_processors = n_processors
        self.id = id 
        self.fit_range = fit_range 
        self.n_pre_samples = n_pre_samples
        self.message = message
        self.n_not_improv_max = n_not_improv_max
        self.print_stats = print_stats
        
        # ------------------------------------- #
        ## - SIDE COMPUTATIONS
        
        self.output = bo_output()
        
        # - Calibration variables
        self.cal_variables_keys = list(self.cal_variables.keys())
        self.cal_variables_bounds = np.array([self.cal_variables[key] for key in self.cal_variables_keys])
        
        # - Define the abcissa
        self.abcissa = calibrate.periods if cal_metric is 'sa' else calibrate.freqs
        
        # - Compute observed IMs
        self.reference_metric()
        
        # - Time step
        self.dt = self.input_parameters['dt']

        # - Control variables
        self.error_min = 1e9
        self.best_performance = {}
        self.n_not_improv = 0
        
        # - Create a folder to save the results and count iteration
        if id is None:
            self.id = 'Trial'
        else:
            self.id = id
        if not os.path.exists(self.id):
            os.mkdir(self.id)
            print("Directory " , self.id ,  " folder created ")

        # ------------------------------------- #
        ## - REFERENCE INFORMATION 
        print('###########################################')
        print('### - Bayesian Optimization Algorithm - ###')
        print('####### - Luis Alvarez - 2019 - ###########')
        print('Project = ' + str(id))
        print('Variables = ' + str(cal_variables))
        print('Calibration component = ' + str(input_parameters['component']))
        print('Calibration metric = ' + str(cal_metric))
        print('Number of simulations = ' + str(n_sim))
        print('Number of iterations = ' + str(iterations))
        print('Number of processors = ' + str(n_processors))
        print('Kerne-nu = ' + str(nu))
        print('###########################################')
    
        self.output.process_log(self.id, init=True, init_dict={'variables':cal_variables, 'component':input_parameters['component'],
                                                               'kernel_nu':str(nu), 'metric':cal_metric, 'simulations':n_sim, 'iterations':iterations, 
                                                               'processors':n_processors, 'message': message})

#--------------------------------------------------------------------------------------------#            
    ### - ACQUISITION FUNCTION - ###
    def expected_improvement(self,x, gaussian_process, evaluated_loss, maximize=False, n_params=1):
        
        """ expected_improvement
        Expected improvement acquisition function.
        Arguments:
        ----------
            x: array-like, shape = [n_samples, n_hyperparams]
                The point for which the expected improvement needs to be computed.
            gaussian_process: GaussianProcessRegressor object.
                Gaussian process trained on previously evaluated hyperparameters.
            evaluated_loss: Numpy array.
                Numpy array that contains the values off the loss function for the previously
                evaluated hyperparameters.
            objective: string.
                String indicating the purpose.  'max': Maximize, 'min': Minimize the function.
            n_params: int.
                Dimension of the hyperparameter space.
        """
    
        x_to_predict = x.reshape(-1, n_params)
    
        mu, sigma = gaussian_process.predict(x_to_predict, return_std=True)
    
        if maximize:
            loss_optimum = np.max(evaluated_loss)
        else:
            loss_optimum = np.min(evaluated_loss)
    
        scaling_factor = (-1) ** (not maximize)
    
        # In case sigma equals zero
        with np.errstate(divide='ignore'):
            Z = scaling_factor * (mu - loss_optimum) / sigma
            expected_improvement = scaling_factor * (mu - loss_optimum) * norm.cdf(Z) + sigma * norm.pdf(Z)
            expected_improvement[sigma == 0.0] == 0.0
    
        return -1 * expected_improvement

#--------------------------------------------------------------------------------------------#            
    ### - SAMPLING FUNCTION - ###
    def sample_next_hyperparameter(self, acquisition_func, gaussian_process, evaluated_loss, maximize=False,
                                bounds=(0, 10), n_restarts=100):
        """ sample_next_hyperparameter
        Proposes the next hyperparameter to sample the loss function for.
        Arguments:
        ----------
            acquisition_func: function.
                Acquisition function to optimise.
            gaussian_process: GaussianProcessRegressor object.
                Gaussian process trained on previously evaluated hyperparameters.
            evaluated_loss: array-like, shape = [n_obs,]
                Numpy array that contains the values off the loss function for the previously
                evaluated hyperparameters.
            objective: string.
                String indicating the purpose.  'max': Maximize, 'min': Minimize the function.
            bounds: Tuple.
                Bounds for the L-BFGS optimiser.
            n_restarts: integer.
                Number of times to run the minimiser with different starting points.
        """
        best_x = None
        best_acquisition_value = 1
        n_params = bounds.shape[0]

        for starting_point in np.random.uniform(bounds[:, 0], bounds[:, 1], size=(n_restarts, n_params)):
            res = minimize(fun=acquisition_func,
                        x0=starting_point.reshape(1, -1),
                        bounds=bounds,
                        method='L-BFGS-B',
                        args=(gaussian_process, evaluated_loss, maximize, n_params))
    
            if res.fun < best_acquisition_value:
                best_acquisition_value = res.fun
                best_x = res.x
    
        return best_x
    
#--------------------------------------------------------------------------------------------#            
    ### - BAYESIAN OPTIMIZATION - ###
    def bayesian_optimisation(self, n_iters_max, sample_loss, bounds, x0=None, n_pre_samples=5,
                            gp_params=None, random_search=False, alpha=1e-5, epsilon=1e-7,maximize=False):
        """ bayesian_optimisation
        Uses Gaussian Processes to optimise the loss function `sample_loss`.
        Arguments:
        ----------
            n_iters: integer.
                Number of iterations to run the search algorithm.
            sample_loss: function.
                Function to be optimised.
            bounds: array-like, shape = [n_params, 2].
                Lower and upper bounds on the parameters of the function `sample_loss`.
            x0: array-like, shape = [n_pre_samples, n_params].
                Array of initial points to sample the loss function for. If None, randomly
                samples from the loss function.
            n_pre_samples: integer.
                If x0 is None, samples `n_pre_samples` initial points from the loss function.
            gp_params: dictionary.
                Dictionary of parameters to pass on to the underlying Gaussian Process.
            random_search: integer.
                Flag that indicates whether to perform random search or L-BFGS-B optimisation
                over the acquisition function.
            alpha: double.
                Variance of the error term of the GP.
            epsilon: double.
                Precision tolerance for floats.
            objective: string.
                String indicating the purpose.  'max': Maximize, 'min': Minimize the function.
        """
 
        # ------------------------------------- #
        ## - START THE ITERATIVE PROCESS 
        
        # ------------------------------------- #
        # - Initiate variables
        self.n_iters = 0
        x_list = []
        y_list = []
    
        # ------------------------------------- #
        # - Run initial evaluations
        n_params = bounds.shape[0]
        print('### - Running initial evaluations - ###')
        if x0 is None:
            for params in np.random.uniform(bounds[:, 0], bounds[:, 1], (n_pre_samples, bounds.shape[0])):
                x_list.append(params)
                y_list.append(sample_loss(params))
        else:
            for params in x0:
                x_list.append(params)
                y_list.append(sample_loss(params))
                
        self.xp = np.array(x_list)
        self.yp = np.array(y_list)
    
        # ------------------------------------- #
        # - Create the GP
        if gp_params is not None:
            model = gp.GaussianProcessRegressor(**gp_params)
        else:
            # kernel =  gp.kernels.Matern(length_scale=1.0, length_scale_bounds=(1e-3, 1e3),nu=self.nu)
            kernel =  gp.kernels.RBF()
            model = gp.GaussianProcessRegressor(kernel=kernel,
                                                alpha=alpha,
                                                n_restarts_optimizer=100,
                                                normalize_y=True)                  
        # ------------------------------------- #
        # - Run iterations 
        print('### - Running iterations - ###')
        while self.n_iters <= n_iters_max: 

            # ------------------------------------- #
            # - Fit the GP to the data
            model.fit(self.xp, self.yp)
            
            # ------------------------------------- #
            # - Sample next hyperparameter
            if random_search:
                x_random = np.random.uniform(bounds[:, 0], bounds[:, 1], size=(random_search, n_params))
                ei = -1 * self.expected_improvement(x_random, model, self.yp, maximize=maximize, n_params=n_params)
                next_sample = x_random[np.argmax(ei), :]
            else:
                next_sample = self.sample_next_hyperparameter(self.expected_improvement, model, self.yp, maximize=maximize, bounds=bounds, n_restarts=100)
    
            # ------------------------------------- #
            # - Handle duplicates
            """
            Duplicates will break the GP. In case of a duplicate, we will randomly sample a next query point.
            """
            if np.any(np.abs(next_sample - self.xp) <= epsilon):
                next_sample = np.random.uniform(bounds[:, 0], bounds[:, 1], bounds.shape[0])

            # ------------------------------------- #
            # - Sample loss for new set of parameters
            cv_score = sample_loss(next_sample)
    
            # ------------------------------------- #
            # - Update lists
            x_list.append(next_sample)
            y_list.append(cv_score)
            self.xp = np.array(x_list)
            self.yp = np.array(y_list)
            print('Iteration = ' + str(self.n_iters))
    
            if self.n_not_improv >= self.n_not_improv_max:
                self.n_iters = self.n_iters*1e8
                
#--------------------------------------------------------------------------------------------#            
    ### - STANDARIZE IMs - ###
    def reference_metric(self):
        
        # ------------------------------------- #
        # - Standarize
        self.metric_reference = {}
        im_reference  = []
        for key_event in self.observed:
            self.metric_reference[key_event] = np.interp(self.abcissa, self.observed[key_event]['abcissas'], self.observed[key_event]['ordinates'])
            im_reference.append(self.metric_reference[key_event])

        # ------------------------------------- #
        # - Set average IM   
        im_reference = np.array(im_reference)
        self.IM_reference = im_reference.mean(axis=0)
                
#--------------------------------------------------------------------------------------------#            
    ### - COMPARE IMs - ###
    def IM_comparison(self,theta):
        t0 = time.time() 
        # --------------------------------------------------- #
        ## - PREPARATION - ##

        # - Print followup message
        self.n_iters += 1
        self.output.process_log(self.id, message='#### - Iteration #' + str(self.n_iters) + ' - ###')

        # - Assign the trial value to the parameters
        info = ''
        for variable, o in zip(self.cal_variables_keys, theta) :
            info = info + variable + ' = ' + "%.2f" % o + ', '
            print(variable + ' = ' + str(o))
            self.input_parameters[variable] = o
            
        # --------------------------------------------------- #
        ## - SIMULATION - ## 

        # ------------------------------------- #
        # - Variable initialization
        Error, Bias, Sim_dist, Sim_acc, Sim_im = {},{},{},{},{}
        if self.duration_boolean is False:
            Bias_Dgm, Bias_Dsm, Error_Dgm, Error_Dsm, Bias_arias, Error_arias = None, None, None, None, None, None
        else:
            Bias_Dgm, Bias_Dsm, Error_Dgm, Error_Dsm, Bias_arias, Error_arias = {}, {}, {}, {}, {}, {}

        # ------------------------------------- #
        # - Enter individual stations
        for key_event in self.observed:
            if self.print_stats == True: print('Started event: ' + key_event)
            error, error_mean = 0., 0.
            Bias[key_event] = {}

            # ------------------------------------- #
            # - Introduce station/event features
            if 'Depth' in self.observed[key_event]:
                self.input_parameters['Depth'] =  self.observed[key_event]['Depth']
            if 'Alpha' in self.observed[key_event]:
                self.input_parameters['Alpha'] =  self.observed[key_event]['Alpha']   
            if 'Betha' in self.observed[key_event]:
                self.input_parameters['Betha'] =  self.observed[key_event]['Betha']
            if 'vs' in self.observed[key_event]:
                self.input_parameters['vs'] =  self.observed[key_event]['vs']   
            if 'vp' in self.observed[key_event]:
                self.input_parameters['vp'] =  self.observed[key_event]['vp']
            if 'hypocenter' in self.observed[key_event]:
                self.input_parameters['hypocenter'] =  self.observed[key_event]['hypocenter']
            if 'TF' in self.observed[key_event]:
                self.input_parameters['TF'] =  self.observed[key_event]['TF']            
            if 'site' in self.observed[key_event]:
                self.input_parameters['site'] =  self.observed[key_event]['site']
            elif 'site' not in self.input_parameters:
                print('No station coordinates defined')

            Mw, component = self.observed[key_event]['Mw'], self.input_parameters['component']
            
            # ------------------------------------- #
            # - Simulation
            a = EQ(Mw, source_file=self.source_file, parameters=self.input_parameters, n_sim=self.n_sim, n_processors=self.n_processors, 
                    point_source=self.point_source_boolean)
            sim_acc = a.sim_acc
            R_jb = a.R_jb

            # ------------------------------------- #
            # - Compute IMs
            sim_im, sim_Dgm, sim_Dsm, sim_arias = compute_sim_im(sim_acc, self.input_parameters['component'], self.cal_metric, self.dt, 
                                                                    self.abcissa, calibrate.damping, duration=self.duration_boolean)
            
            # ------------------------------------- #
            ## - COMPUTE BIASES
            # - Spectral response            
            im_sim = []
            for i in range(len(self.abcissa)):
                bias_temp,sim_temp = [],[]
                for j in range(len(sim_im)):
                    sim_temp.append(sim_im[j][i])
                    bias_temp.append(np.log(self.metric_reference[key_event][i]/sim_im[j][i]))
                
                Bias[key_event][self.abcissa[i]] = bias_temp
                im_sim.append(np.mean(sim_temp))
                
                # - Error per event
                if self.fit_range is None:
                    error += np.mean(bias_temp)**2.
                else:
                    if self.abcissa[i] >= self.fit_range[0] and self.abcissa[i] <= self.fit_range[1]:
                        error += np.mean(bias_temp)**2.

            # - Durations
            if self.duration_boolean is not False:
                error_arias, error_Dsm, error_Dgm = 0., 0., 0.
                Bias_Dgm[key_event], Bias_Dsm[key_event], Bias_arias[key_event] = [], [], []
                dgm_bias, dsm_bias, arias_bias = [], [], []
                for j in range(len(sim_Dsm)):
                    dgm_bias.append(np.log(self.observed[key_event]['Dgm']/sim_Dgm[j]))
                    dsm_bias.append(np.log(self.observed[key_event]['Dsm']/sim_Dsm[j]))
                    arias_bias.append(np.log(self.observed[key_event]['arias']/sim_arias[j]))
                
                error_arias, error_dgm, error_dsm = np.mean(arias_bias)**2., np.mean(dgm_bias)**2., np.mean(dsm_bias)**2.
                Bias_Dgm[key_event], Bias_Dsm[key_event], Bias_arias[key_event] = dgm_bias, dsm_bias, arias_bias
            
                
            # ------------------------------------- #
            ## - COMPUTE ERRORS PER EVENT
            # - Save event results
            Sim_acc[key_event] = sim_acc
            Sim_im[key_event] = sim_im
            Error[key_event] = error
            if self.duration_boolean is not False:
                Error_arias[key_event], Error_Dgm[key_event], Error_Dsm[key_event] =  error_arias, error_dgm, error_dsm
            
            if self.print_stats == True: print('   ' + key_event + ' - Rjb: ' + str(R_jb) + ' - Spectral Error: ' + str(error))
            
        # ------------------------------------- #
        ## - AGGREGATE THE ERRORS OF ALL EVENTS
        error_aggregated = 0.0
        for key in Error:
            if self.duration_boolean is False:
                error_aggregated += Error[key]**0.5
            else:
                error_aggregated += (Error[key] + Error_arias[key] + Error_Dsm[key])**0.5

        error_aggregated = (error_aggregated/len(Error))**0.5    
                
        # ------------------------------------- #
        ## - PLOT THE BIAS for ITERATION
        Bias_Mean, Bias_Std = self.output.bias_per_iteration(self.abcissa, Bias, duration=Bias_Dsm, arias=Bias_arias,
                                        save_path=self.id + '//bias_' + ' iteration ' + str(self.n_iters) + '.jpg')        


        # ------------------------------------- #
        ## - STORE BEST SOLUTION
        if error_aggregated <= self.error_min:
            self.n_not_improv = 0
            self.error_min = error_aggregated
            self.best_performance['sim_acc'] = Sim_acc
            self.best_performance['sim_im'] = Sim_im
            self.best_performance['sim_r_rup'] = Sim_dist
            self.best_performance['bias'] = {'total':{'mean':Bias_Mean, 'std':Bias_Std}, 'individual':Bias}
                
        else:
            self.n_not_improv += 1

        # ------------------------------------- #
        ## - SAVE PERFORMANCE FOR THE ITERATION
        if self.print_stats == True:
            t3 = time.time()    
            running_time = t3-t0
            print('Iteration time = ' + str("%.3f" % round(running_time,3)) + 's')
            print('Error = ' + str(error_aggregated))

        # - Log up the process
        info = info + ' Error = ' + str(error_aggregated)
        self.output.process_log(self.id,message=info)

        return np.array(error_aggregated)

#--------------------------------------------------------------------------------------------#            
    ### - MINIMIZATION - ###      
    def opt_minimize(self):
        # ------------------------------------- #
        # - Launch the bayesian optimization
        self.bayesian_optimisation(self.iterations, self.IM_comparison, self.cal_variables_bounds, x0=None, n_pre_samples=self.n_pre_samples, gp_params=None,
                                   random_search=False, alpha=1e-5, epsilon=1e-7, maximize=False)
        
        # ------------------------------------- #
        # - Plot trial - error evolution
        #trial = np.arange(0, self.n_iters, 1)
        #title = str(self.id) + ' Error evolution'
        #self.output.error_evolution(trial, self.yp, save_path=self.id + '//' + title + '.jpg',title=title)

        # ------------------------------------- #
        # - Save results
        with open(self.id + '//' + 'results.pkl', 'wb') as f:  # Python 3: open(..., 'wb')
            pickle.dump(self.best_performance, f)
        f.close() 
