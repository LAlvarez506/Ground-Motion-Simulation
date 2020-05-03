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
from BO_output import bo_output
from BO_simulator import *
from LibraryGMPE import Zhao_2006
from Methods import GMSM
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

    
    # - Default signal properties
    ## - Frequencies
    freq_max = 24
    freqs = np.linspace(0.01, 24., 20)
    freqs_calibration = []
    
    # - Periods
    periods = np.linspace(0., 5., 51)
    periods_calibration = [0., 5.]
    
    dt = 0.01
    logscale = True
    y = []
    delta_az = 5.0
    damping = 0.05
    pp = 50.0
    b = 40.0

    # - Default bayesian optimizaton parameters
    n_pre_samples = 5
    stats_boolean = True

    # - Storing results properties
    save_figures_boolean = True
    DistBias_freqs = [1.0,5.0,10.0,20.0]
    error_type = 1
    
    # - Parallel code
    n_processors = 3
    n_sim = 3
    n_not_improv_max = 20

    # - Color map parameter
    c_map = 'RdYlBu'
    
    # - Resume calibration
    resume_boolean = False
    partial_data = {}
    
    def __init__(self,observed, input_parameters, cal_metric, variable, iterations=5, n_sim=5, nu=0.5,
                 error_type=1, n_processors=None, wd_GMSM=None, id=None,resume=None, resume_file=None, 
                 fit_range=None, n_pre_samples=None, message=None):
        
        # - Instance dedicated to safe the gaussian models during the optimization process
        self.gaussian_model = []
        
        # - Instantiate the outpu module
        self.output = bo_output()
        
        if resume == True:
            calibrate.resume_boolean = True
        self.performances = {}
        # --------------------------------- #
        # - Initiate calibration 
        
        if n_pre_samples is None:
            calibrate.n_pre_samples = calibrate.n_pre_samples
        else:
            calibrate.n_pre_samples = n_pre_samples

        calibrate.error_type = error_type
        if calibrate.error_type == 1:
            error_descript = 'SRMS'
        elif calibrate.error_type == 2:
            error_descript = 'Maximum'
        elif calibrate.error_type == 3:
            error_descript = 'SRMS + Slope'
        
        if fit_range is not None:
            if cal_metric is 'fas':
                if fit_range is None:
                    calibrate.freqs_calibration = [calibrate.freqs[0],calibrate.freqs[-1]]
                else: 
                    calibrate.freqs_calibration = fit_range
            if (cal_metric is 'spectrum' or cal_metric is 'spectra' or cal_metric is 'GMPE'):
                if fit_range is None:
                    calibrate.periods_calibration = [calibrate.periods[0],calibrate.periods[-1]]
                else: 
                    calibrate.periods_calibration = fit_range
            
        if calibrate.resume_boolean is False:

            # - Create a folder to save the results and count iteration
            import os
            if id is None:
                self.id = 'Trial'
            else:
                self.id = id
            if not os.path.exists(self.id):
                os.mkdir(self.id)
                print("Directory " , self.id ,  " folder created ")

            
            ## - Optimization variables
            self.nu = nu
            self.var = variable
            self.opt_param = list(self.var.keys())
            self.tried_variables = {}
            bounds = []
            for key in variable:
                self.tried_variables[key] = []
                bounds.append(self.var[key])
            self.bounds = np.array(bounds)
            
            ## - Iterations and Parallel processing
            if iterations == None:
                self.iterations = 5
            else:
                self.iterations = iterations

            if n_sim != None:
                calibrate.n_sim = n_sim

            if n_processors != None:
                calibrate.n_processors = n_processors
            
            ## - Calibration data
            self.observed = observed
            self.input_parameters = input_parameters
            self.n_events = len(observed)
            self.cal_metric = cal_metric
            
            if 'dt' in self.input_parameters:
                calibrate.dt = self.input_parameters['dt']

            ## - Control variables
            self.distances = {}
            self.error_min = 1e9
            self.n = 0
            self.best_performance = {}
            self.points = []
            self.sim_acc = []
            self.sim_im = []
            self.sim_r_rup = []
            self.error = []
            self.bias_aggregated = []
            self.bias_individual = []
            self.n_not_improv = 0

            ## - Compute statistics for reference parameters
            self.reference_metric()
            
            ## - Print reference informations
            print('###########################################')
            print('### - Bayesian Optimization Algorithm - ###')
            print('####### - Luis Alvarez - 2019 - ###########')
            print('Project = ' + str(id))
            print('Variables = ' + str(variable))
            print('Calibration component = ' + str(self.input_parameters['component']))
            print('Calibration metric = ' + str(cal_metric))
            print('Number of simulations = ' + str(n_sim))
            print('Number of iterations = ' + str(iterations))
            print('Number of processors = ' + str(calibrate.n_processors))
            print('Kerne-nu = ' + str(self.nu))
            print('Error type = ' + str(error_descript))
            print('###########################################')
        
            self.output.process_log(self.id, init=True, init_dict={'variables':variable, 'component':str(self.input_parameters['component']),'kernel_nu':str(self.nu),'freq corr':self.input_parameters['correlation'], 'metric':cal_metric, 'nw':self.input_parameters['nw'], 'nl':self.input_parameters['nl'], 'error_type':error_descript,                                                                'simulations':n_sim, 'iterations':iterations, 'processors':calibrate.n_processors, 'message': message})
        # --------------------------------- #
        # - Resume calibration
        else:
            with open(resume_file, 'rb') as file:  # Python 3: open(..., 'wb')
                calibrate.partial_data = pickle.load(file)
            file.close()
            
            ## - Optimization variables
            self.nu = nu
            self.var = variable
            self.opt_param = list(self.var.keys())
            self.tried_variables = {}
            bounds = []
            for key in variable:
                self.tried_variables[key] = []
                bounds.append(self.var[key])
            self.bounds = np.array(bounds)
            
            ## - Iterations and Parallel processing
            if iterations == None:
                self.iterations = 5
            else:
                self.iterations = iterations

            if n_sim != None:
                calibrate.n_sim = n_sim

            if n_processors != None:
                calibrate.n_processors = n_processors
            
            ## - Calibration data
            self.observed = observed
            self.input_parameters = input_parameters
            self.n_events = len(observed)
            self.cal_metric = cal_metric
            
            if 'dt' in self.input_parameters:
                calibrate.dt = self.input_parameters['dt']
            
            ## - Control variables
            self.error_min = calibrate.partial_data['error_min']
            self.n_iters = calibrate.partial_data['n_iters']
            self.n = calibrate.partial_data['n']
            self.best_performance = calibrate.partial_data['best_performance']
            self.points = calibrate.partial_data['points']
            self.sim_acc = calibrate.partial_data['sim_acc']
            self.sim_im = calibrate.partial_data['sim_im']
            self.sim_r_rup = calibrate.partial_data['sim_r_rup']
            self.error = calibrate.partial_data['error']
            self.bias_aggregated = calibrate.partial_data['bias_aggregated']
            self.bias_individual = calibrate.partial_data['bias_individual']
            self.n_not_improv = calibrate.partial_data['n_not_improv']
            
            self.gaussian_model = calibrate.partial_data['gaussian_model']
            self.xp,self.yp = calibrate.partial_data['xp'],calibrate.partial_data['yp'] 
            self.id = calibrate.partial_data['id']
            self.tried_variables = calibrate.partial_data['tried_variables'] 
            self.distances = calibrate.partial_data['distances'] 

            
            ## - Load reference data
            self.metric_reference = calibrate.partial_data['metric_reference']
            self.IM_reference,self.ref_plot = calibrate.partial_data['IM_reference'],calibrate.partial_data['ref_plot']
            
            ## - Print reference informations
            print('###########################################')
            print('### - Bayesian Optimization Algorithm - ###')
            print('####### - Luis Alvarez - 2019 - ###########')
            print('#------- - Resuming computations - -------#')
            print('Project = ' + str(id))
            print('Variables = ' + str(variable))
            print('Calibration component = ' + str(self.input_parameters['component']))
            print('Calibration metric = ' + str(cal_metric))
            print('Number of simulations = ' + str(n_sim))
            print('Number of iterations = ' + str(iterations))
            print('Number of processors = ' + str(calibrate.n_processors))
            print('###########################################')
            
        # --------------------------------- #
  
###############################################################################################################################################################
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
#### ---------------------------------------------------------------------------------------------------------------------------- ####

    def sample_next_hyperparameter(self,acquisition_func, gaussian_process, evaluated_loss, maximize=False,
                                bounds=(0, 10), n_restarts=25):
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
#### ---------------------------------------------------------------------------------------------------------------------------- ####
    def bayesian_optimisation(self,n_iters_max, sample_loss, bounds, x0=None, n_pre_samples=5,
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
        import pickle
        def save_partial():
            # - Save partial results
            calibrate.partial_data['gaussian_model'] = self.gaussian_model
            calibrate.partial_data['xp'],calibrate.partial_data['yp'] = self.xp,self.yp
            calibrate.partial_data['n_iters'] = self.n_iters
            calibrate.partial_data['n'],calibrate.partial_data['id'] = self.n,self.id
            calibrate.partial_data['tried_variables'] = self.tried_variables
            calibrate.partial_data['points'] = self.points
            calibrate.partial_data['sim_acc'] = self.sim_acc
            calibrate.partial_data['sim_im'] = self.sim_im
            calibrate.partial_data['sim_r_rup'] = self.sim_r_rup
            calibrate.partial_data['error'] = self.error
            calibrate.partial_data['bias_aggregated'] = self.bias_aggregated
            calibrate.partial_data['bias_individual'] = self.bias_individual
            calibrate.partial_data['error_min'] = self.error_min
            calibrate.partial_data['best_performance'] = self.best_performance
            calibrate.partial_data['distances'] = self.distances
            calibrate.partial_data['n_not_improv'] = self.n_not_improv
            with open(self.id + '//' + 'resume_file.pkl', 'wb') as f:  # Python 3: open(..., 'wb')
                pickle.dump(calibrate.partial_data, f)
            f.close()
 
        # - Find the best result so far
        def best_point(maximize,x,y):
            if maximize == True:
                y_best = max(y)
            else:
                y_best = min(y)
            
            index = np.where(y == y_best)
            return x[index],y_best
        # --------------------------------- #
        if calibrate.resume_boolean == False:
            
            self.n_iters = 0
            x_list = []
            y_list = []
        
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
        
            # Create the GP
            if gp_params is not None:
                model = gp.GaussianProcessRegressor(**gp_params)
            else:
                kernel =  1.0 * gp.kernels.Matern(length_scale=1.0, length_scale_bounds=(1e-3, 1e3),nu=self.nu)
                model = gp.GaussianProcessRegressor(kernel=kernel,
                                                    alpha=alpha,
                                                    n_restarts_optimizer=10,
                                                    normalize_y=True)                  
            self.gaussian_model.append(model)
            # - Save partial results
            save_partial()
            
            # - Run iterations 
            print('### - Running iterations - ###')
            while self.n_iters <= n_iters_max: 
                x_best,y_best = best_point(maximize,self.xp,self.yp)
                ## - Fit the GP to the data
                model.fit(self.xp, self.yp)
                self.gaussian_model.append(model)
                
                ## - Sample next hyperparameter
                if random_search:
                    x_random = np.random.uniform(bounds[:, 0], bounds[:, 1], size=(random_search, n_params))
                    ei = -1 * self.expected_improvement(x_random, model, self.yp, maximize=maximize, n_params=n_params)
                    next_sample = x_random[np.argmax(ei), :]
                else:
                    next_sample = self.sample_next_hyperparameter(self.expected_improvement, model, self.yp, maximize=maximize, bounds=bounds, n_restarts=100)
        
                ## - Duplicates will break the GP. In case of a duplicate, we will randomly sample a next query point.
                if np.any(np.abs(next_sample - self.xp) <= epsilon):
                    next_sample = np.random.uniform(bounds[:, 0], bounds[:, 1], bounds.shape[0])

                ## - Sample loss for new set of parameters
                cv_score = sample_loss(next_sample)
        
                ## - Update listsgeometry
                x_list.append(next_sample)
                y_list.append(cv_score)
        
                ## - Update xp and yp
                self.xp = np.array(x_list)
                self.yp = np.array(y_list)
                print('Iteration = ' + str(self.n_iters))
                self.n_iters += 1
                if self.n_not_improv >= calibrate.n_not_improv_max:
                    self.n_iters = self.n_iters*1e8
                    
                # - Save partial results
                save_partial()

        # --------------------------------- #
        else:
            x_list = list(self.xp)
            y_list = list(self.yp)
            
            # Create the GP
            if gp_params is not None:
                model = gp.GaussianProcessRegressor(**gp_params)
            else:
                kernel = 1.0 * gp.kernels.Matern(length_scale=1.0, length_scale_bounds=(1e-3, 1e3),nu=self.nu)
                model = gp.GaussianProcessRegressor(kernel=kernel,
                                                    alpha=alpha,
                                                    n_restarts_optimizer=10,
                                                    normalize_y=True)
            print('### - Running iterations - ###')
            while self.n_iters <= n_iters_max: 
                x_best,y_best = best_point(maximize,self.xp,self.yp)
                ## - Fit the GP to the data
                model.fit(self.xp, self.yp)
        
                ## - Sample next hyperparameter
                if random_search:
                    x_random = np.random.uniform(bounds[:, 0], bounds[:, 1], size=(random_search, n_params))
                    ei = -1 * self.expected_improvement(x_random, model, self.yp, maximize=maximize, n_params=n_params)
                    next_sample = x_random[np.argmax(ei), :]
                else:
                    next_sample = self.sample_next_hyperparameter(self.expected_improvement, model, self.yp, maximize=maximize, bounds=bounds, n_restarts=100)
        
                ## - Duplicates will break the GP. In case of a duplicate, we will randomly sample a next query point.
                if np.any(np.abs(next_sample - self.xp) <= epsilon):
                    next_sample = np.random.uniform(bounds[:, 0], bounds[:, 1], bounds.shape[0])

                ## - Sample loss for new set of parameters
                cv_score = sample_loss(next_sample)
        
                ## - Update lists
                x_list.append(next_sample)
                y_list.append(cv_score)
        
                ## - Update xp and yp
                self.xp = np.array(x_list)
                self.yp = np.array(y_list)
                print('Iteration = ' + str(self.n_iters))
                self.n_iters += 1
                if self.n_not_improv >= calibrate.n_not_improv_max:
                    self.n_iters = self.n_iters*1e8
            
                # - Save partial results
                save_partial()

###############################################################################################################################################################
    def avg_IM(self,list_standarized):
        list_standarized = np.array(list_standarized)
        return list_standarized.mean(axis=0)
###############################################################################################################################################################
    def reference_metric(self):
        if self.cal_metric == 'spectrum':
            self.IM_reference = []
            for p in calibrate.periods:
                sa = np.interp(p,self.observed['periods'],self.observed['sa'])
                self.IM_reference.append(sa)
            
            self.IM_reference = np.array(self.IM_reference)

        elif self.cal_metric == 'fas':
            self.metric_reference = {}
            im_reference  = []
            for key_event in self.observed:
                fas_observed = [np.interp(f,self.observed[key_event]['freqs'],self.observed[key_event]['fas']) for f in calibrate.freqs]
                self.metric_reference[key_event] = fas_observed
                im_reference.append(fas_observed)

            # - Get the mean  reference IM 
            self.IM_reference = self.avg_IM(im_reference)

        elif (self.cal_metric == 'spectra' or self.cal_metric == 'EXSIM_spectra'):
            self.metric_reference = {}
            im_reference  = []
            for key_event in self.observed:
                sa_observed = np.interp(calibrate.periods, self.observed[key_event]['periods'], self.observed[key_event]['sa'])
                self.metric_reference[key_event] = sa_observed
                im_reference.append(sa_observed)

            # - Get the mean  reference IM 
            self.IM_reference = self.avg_IM(im_reference)
                
            # - Save partial results
            calibrate.partial_data['IM_reference'] = self.IM_reference
            calibrate.partial_data['metric_reference'] = self.metric_reference
 
 ##############################################################################################           
    def fas_multiple(self,X):
        from Otarola_Parallel import EQ
        t0 = time.time()
        # --------------------------------- #
        self.n += 1
        self.output.process_log(self.id,message='#### - Iteration #' + str(self.n) + ' - ###')
        # --------------------------------- #   
        # - Assign the trial value to the parameters
        info = ''
        for parameter,x in zip(self.opt_param,X) :
            self.tried_variables[parameter].append(x)
            info = info + parameter + ' = ' + "%.2f" % x + ', '
            print(parameter + ' = ' + str(x))
            self.input_parameters[parameter] = x
            
        # --------------------------------- #   
        # - Define abcissa based on calobration metric
        IM_sim_list = []
        abcissa = calibrate.freqs
        xlegend = 'Frequencies [Hz]'
        ylegend = 'FAS [cm/s]'
            
        # ------------ SIMULATION AND INDIVIDUAL EVENT BIAS ---------------- #   
        Error, Bias, Sim_dist, Sim_acc, Sim_im = {},{},{},{},{}
        for key_event in self.observed:
            error = 0.0
            Bias[key_event] = {}
            
            # - Introduce event features
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
            self.input_parameters['site'] = self.observed[key_event]['site']
            
            # - Simulate and compute IM
            Mw, component = self.observed[key_event]['Mw'],self.input_parameters['component']
            simulation_acc, simulation_im, distance = sim_fas_multiple(Mw,self.input_parameters,calibrate.n_sim,component,
                                                         calibrate.dt,abcissa,calibrate.b,calibrate.delta_az,calibrate.pp)
            # - Compute biases
            im_sim = []
            for i in range(len(abcissa)):
                bias_temp,sim_temp = [],[]
                for j in range(len(simulation_im)):
                    sim_temp.append(simulation_im[j][i])
                    bias_temp.append(np.log(self.metric_reference[key_event][i]/simulation_im[j][i]))
                
                Bias[key_event][abcissa[i]] = bias_temp
                im_sim.append(np.mean(sim_temp))
                
                # - Error per event
                if calibrate.freqs[i] >= calibrate.freqs_calibration[0] and calibrate.freqs[i] <= calibrate.freqs_calibration[1]:
                    error += np.mean(bias_temp)**2

            # - Save event results
            IM_sim_list.append(im_sim)
            Sim_acc[key_event] = simulation_acc
            Sim_im[key_event] = simulation_im
            Sim_dist[key_event] = distance
            Error[key_event] = error
            
            # - Stats
            if calibrate.stats_boolean == True:
                print(key_event + ' - Distance: ' + str(distance) + ' - Error: ' + str(error))
        
        # ------------ EVENT ARRAY AGGREGATED BIAS AND ERROR  ---------------- #   
        ## - ERROR
        # - Distance bias
        distances, err = [], []
        for key_event in Error:
            distances.append(Sim_dist[key_event])
            err.append(Error[key_event])
            
        errors = [errors_i for _,errors_i in sorted(zip(distances,err))]
        distances.sort()
        
        x = np.array(distances).reshape((-1, 1))
        y = np.array(errors)
        model = LinearRegression().fit(x, y)
        slope_error = model.coef_[0]
        b_error = model.intercept_

        if calibrate.error_type == 1:
            """
            Squared root of the mean suquared of the errors
            """
            error_aggregated = 0.0
            for key in Error:
                error_aggregated += Error[key]**2.
            error_aggregated = (error_aggregated/len(Error))**0.5
        
        elif calibrate.error_type == 2:
            """
            Maximum error
            """
            error_aggregated = 0.0
            for key in Error:
                if Error[key] > error_aggregated:
                    error_aggregated = Error[key]
            error_aggregated = (error_aggregated)**0.5
        
        elif calibrate.error_type == 3:
            """
            Combined error.  Bias slope (distance vs error) and maximum error
            """
            # - Frequency related error
            freq_error = 0.0
            for key in Error:
                freq_error += Error[key]**2.
            freq_error = (freq_error/len(Error))**0.5
            
            # - Aggregate errors
            error_aggregated = freq_error*(1 + abs(slope_error))
            
            
        # - BIAS
        bias_per_abcissa = {}
        for f in abcissa:
            bias_per_abcissa[f] = {}
            temp,freq = [],[]
            for key in Bias:
                for i in range(len(Bias[key][f])):
                    temp.append(Bias[key][f][i])
                    freq.append(f)
            
            bias_per_abcissa[f] = {'abcissa':freq,'bias':temp,'bias_mean':np.mean(temp),'bias_std':np.std(temp)}

        # - Plot the bias for the iteration
        ref_bias_plot = self.output.bias_per_iteration(abcissa,bias_per_abcissa,xlegend,
                                       save_path=self.id + '//bias_' + ' iteration ' + str(self.n) + '.jpg')        
        
        # ------------ SAVE ITERATION RESULTS  ---------------- #
        self.points.append(X)
        self.sim_im.append(Sim_im)
        self.sim_r_rup.append(Sim_dist)
        self.sim_acc.append(Sim_acc)
        self.error.append(error_aggregated)
        self.bias_aggregated.append(bias_per_abcissa)
        self.bias_individual.append(Bias)

        
        # ------------ BEST SOLUTION   ---------------- #
        if error_aggregated <= self.error_min:
            self.n_not_improv = 0
            self.error_min = error_aggregated
            self.best_performance['sim_acc'] = Sim_acc
            self.best_performance['sim_im'] = Sim_im
            self.best_performance['sim_r_rup'] = Sim_dist
            self.best_performance['bias_info'] = bias_per_abcissa
            self.best_performance['bias_individual'] = Bias
            
            ### - Distance bias per freuqnecy of iterest
            for fi in calibrate.DistBias_freqs:
                dist, mean, std = [], [], []
                for key_event in self.observed:
                    y_obs = np.interp(fi,calibrate.freqs,self.metric_reference[key_event])
                    temp = []
                    for i in range(calibrate.n_sim):
                        y_sim = np.interp(fi,calibrate.freqs,simulation_im[i])
                        temp.append(np.log(y_obs/y_sim))
                    
                    dist.append(Sim_dist[key_event])
                    mean.append(np.mean(temp))
                    std.append(np.std(temp))
                    
                title = 'Distance bias - f:' + str(fi)
                self.output.frequency_distance_bias_plot(dist,mean,std,title=title,save_path= self.id + '//' + title + '.jpg')
                
            ### - Aggregated distance bias 
            title = 'Aggregated distance bias'
            self.output.distance_bias_plot(distances,errors,b_error,slope_error,title=title,save_path= self.id + '//' + title + '.jpg')
            


            ### - Plot the average IM for the best solution
            self.IM_sim = self.avg_IM(IM_sim_list)
            title = str(self.id) + ' iteration ' + str(self.n) + ' - IM comparison'
            self.output.avg_IM_comparison(abcissa,self.IM_sim,self.IM_reference,xlegend,ylegend,logscale=calibrate.logscale,
                                          title=title,save_path=self.id + '//' + title + '.jpg')

        else:
            self.n_not_improv += 1
            
        # ------------ VARIABLE EXPLORATION PLOT  ---------------- #
        trial = np.linspace(0,self.n,self.n+1)
        for key in self.tried_variables:
            # Save name check
            key_save = ''
            for l in key:
                if l == '_':
                    pass
                else:
                    key_save += l
            self.output.variable_exploration(key,calibrate.c_map,trial,self.tried_variables[key],self.error,save_path = self.id + '//' + key_save +'.jpg')
        
        # ------------ ITERATION PERFORMANCE INFO  ---------------- #
        t3 = time.time()    
        running_time = t3-t0
        
        if calibrate.stats_boolean == True:
            print('Iteration time = ' + str("%.3f" % round(running_time,3)) + 's')
            print('Error = ' + str(error_aggregated))
        
        # - Log up the process
        info = info + ' Error = ' + str(error_aggregated)
        self.output.process_log(self.id,message=info)
        return np.array(error_aggregated)
 ##############################################################################################           
    def sa_multiple(self,X):
        from Otarola_Parallel import EQ
        t0 = time.time()
        # --------------------------------- #
        self.n += 1
        self.output.process_log(self.id,message='#### - Iteration #' + str(self.n) + ' - ###')
        # --------------------------------- #   
        # - Assign the trial value to the parameters
        info = ''
        for parameter,x in zip(self.opt_param,X) :
            self.tried_variables[parameter].append(x)
            info = info + parameter + ' = ' + "%.2f" % x + ', '
            print(parameter + ' = ' + str(x))
            self.input_parameters[parameter] = x
            
        # --------------------------------- #   
        # - Define abcissa based on calobration metric
        IM_sim_list = []
        abcissa = calibrate.periods
        xlegend = 'Periods [s]'
        ylegend = 'Sa [cm/s/s]'
            
        # ------------ SIMULATION AND INDIVIDUAL EVENT BIAS ---------------- #   
        Error, Bias, Sim_dist, Sim_acc, Sim_im = {},{},{},{},{}
        for key_event in self.observed:
            error = 0.0
            Bias[key_event] = {}
            
            # - Introduce event features
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
            self.input_parameters['site'] = self.observed[key_event]['site']
            
            # - Simulate and compute IM
            Mw, component = self.observed[key_event]['Mw'], self.input_parameters['component']
            simulation_acc, simulation_im, distance = sim_sa_multiple(Mw, self.input_parameters, calibrate.n_sim, component,
                                                         calibrate.dt, abcissa, calibrate.damping)
            # - Compute biases
            im_sim = []
            for i in range(len(abcissa)):
                bias_temp,sim_temp = [],[]
                for j in range(len(simulation_im)):
                    sim_temp.append(simulation_im[j][i])
                    bias_temp.append(np.log(self.metric_reference[key_event][i]/simulation_im[j][i]))
                
                Bias[key_event][abcissa[i]] = bias_temp
                im_sim.append(np.mean(sim_temp))
                
                # - Error per event
                if calibrate.periods[i] >= calibrate.periods_calibration[0] and calibrate.periods[i] <= calibrate.periods_calibration[1]:
                    error += np.mean(bias_temp)**2

            # - Save event results
            IM_sim_list.append(im_sim)
            Sim_acc[key_event] = simulation_acc
            Sim_im[key_event] = simulation_im
            Sim_dist[key_event] = distance
            Error[key_event] = error
            
            # - Stats
            if calibrate.stats_boolean == True:
                print(key_event + ' - Distance: ' + str(distance) + ' - Error: ' + str(error))
        
        # ------------ EVENT ARRAY AGGREGATED BIAS AND ERROR  ---------------- #   
        ## - ERROR
        # - Distance bias
        distances, err = [], []
        for key_event in Error:
            distances.append(Sim_dist[key_event])
            err.append(Error[key_event])
            
        errors = [errors_i for _,errors_i in sorted(zip(distances,err))]
        distances.sort()
        
        x = np.array(distances).reshape((-1, 1))
        y = np.array(errors)
        model = LinearRegression().fit(x, y)
        slope_error = model.coef_[0]
        b_error = model.intercept_

        if calibrate.error_type == 1:
            """
            Squared root of the mean suquared of the errors
            """
            error_aggregated = 0.0
            for key in Error:
                error_aggregated += Error[key]**2.
            error_aggregated = (error_aggregated/len(Error))**0.5
        
        elif calibrate.error_type == 2:
            """
            Maximum error
            """
            error_aggregated = 0.0
            for key in Error:
                if Error[key] > error_aggregated:
                    error_aggregated = Error[key]
            error_aggregated = (error_aggregated)**0.5
        
        elif calibrate.error_type == 3:
            """
            Combined error.  Bias slope (distance vs error) and maximum error
            """
            # - Frequency related error
            freq_error = 0.0
            for key in Error:
                freq_error += Error[key]**2.
            freq_error = (freq_error/len(Error))**0.5
            
            # - Aggregate errors
            error_aggregated = freq_error*(1 + abs(slope_error))
            
            
        # - BIAS
        bias_per_abcissa = {}
        for f in abcissa:
            bias_per_abcissa[f] = {}
            temp,freq = [],[]
            for key in Bias:
                for i in range(len(Bias[key][f])):
                    temp.append(Bias[key][f][i])
                    freq.append(f)
            
            bias_per_abcissa[f] = {'abcissa':freq,'bias':temp,'bias_mean':np.mean(temp),'bias_std':np.std(temp)}

        # - Plot the bias for the iteration
        ref_bias_plot = self.output.bias_per_iteration(abcissa,bias_per_abcissa,xlegend,
                                       save_path=self.id + '//bias_' + ' iteration ' + str(self.n) + '.jpg')        
        
        # ------------ SAVE ITERATION RESULTS  ---------------- #
        self.points.append(X)
        self.sim_im.append(Sim_im)
        self.sim_r_rup.append(Sim_dist)
        self.sim_acc.append(Sim_acc)
        self.error.append(error_aggregated)
        self.bias_aggregated.append(bias_per_abcissa)
        self.bias_individual.append(Bias)

        
        # ------------ BEST SOLUTION   ---------------- #
        if error_aggregated <= self.error_min:
            self.n_not_improv = 0
            self.error_min = error_aggregated
            self.best_performance['sim_acc'] = Sim_acc
            self.best_performance['sim_im'] = Sim_im
            self.best_performance['sim_r_rup'] = Sim_dist
            self.best_performance['bias_info'] = bias_per_abcissa
            self.best_performance['bias_individual'] = Bias
                
            ### - Aggregated distance bias 
            title = 'Aggregated distance bias'
            self.output.distance_bias_plot(distances,errors,b_error,slope_error,title=title,save_path= self.id + '//' + title + '.jpg')
            


            ### - Plot the average IM for the best solution
            self.IM_sim = self.avg_IM(IM_sim_list)
            title = str(self.id) + ' iteration ' + str(self.n) + ' - IM comparison'
            self.output.avg_IM_comparison(abcissa,self.IM_sim,self.IM_reference,xlegend,ylegend,logscale=calibrate.logscale,
                                          title=title,save_path=self.id + '//' + title + '.jpg')
        else:
            self.n_not_improv += 1
            
        # ------------ VARIABLE EXPLORATION PLOT  ---------------- #
        trial = np.linspace(0,self.n,self.n+1)
        for key in self.tried_variables:
            # Save name check
            key_save = ''
            for l in key:
                if l == '_':
                    pass
                else:
                    key_save += l
            self.output.variable_exploration(key,calibrate.c_map,trial,self.tried_variables[key],self.error,save_path = self.id + '//' + key_save +'.jpg')
        
        # ------------ ITERATION PERFORMANCE INFO  ---------------- #
        t3 = time.time()    
        running_time = t3-t0
        
        if calibrate.stats_boolean == True:
            print('Iteration time = ' + str("%.3f" % round(running_time,3)) + 's')
            print('Error = ' + str(error_aggregated))
        
        # - Log up the process
        info = info + ' Error = ' + str(error_aggregated)
        self.output.process_log(self.id,message=info)
        return np.array(error_aggregated)
  ##############################################################################################           
    def gmpe_multiple(self,X):
        from Otarola_Parallel import EQ
        t0 = time.time()
        # --------------------------------- #
        self.n += 1
        self.output.process_log(self.id,message='#### - Iteration #' + str(self.n) + ' - ###')
        # --------------------------------- #   
        # - Assign the trial value to the parameters
        info = ''
        for parameter,x in zip(self.opt_param,X) :
            self.tried_variables[parameter].append(x)
            info = info + parameter + ' = ' + "%.2f" % x + ', '
            print(parameter + ' = ' + str(x))
            self.input_parameters[parameter] = x
            
        # --------------------------------- #   
        # - Define abcissa based on calobration metric
        IM_sim_list = []
        abcissa = calibrate.periods
        xlegend = 'Periods [s]'
        ylegend = 'Sa [cm/s/s]'
            
        # ------------ SIMULATION AND INDIVIDUAL EVENT BIAS ---------------- #   
        Error, Bias, Sim_dist, Sim_acc, Sim_im = {},{},{},{},{}
        for key_event in self.observed:
            error = 0.0
            Bias[key_event] = {}
            
            # - Introduce event features
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
            self.input_parameters['site'] = self.observed[key_event]['site']
            
            # - Simulate and compute IM
            Mw, component = self.observed[key_event]['Mw'], self.input_parameters['component']
            simulation_acc, simulation_im, distance = sim_sa_multiple(Mw, self.input_parameters, calibrate.n_sim, component,
                                                         calibrate.dt, abcissa, calibrate.damping)
            
            
            self.metric_reference = []
            for pi in calibrate.abcissa:
                sa, sigma = Zhao_2006(pi,self.observed[key_event]['Mw'], distance, self.input_parameters['hypocenter'][2],
                                      self.input_parameters['vs30'],0, 1, 0, 0)
                
                self.metric_reference.append(sa*981.)
                
            self.metric_reference = np.array(self.metric_reference)
                
            # - Compute biases
            im_sim = []
            for i in range(len(abcissa)):
                bias_temp,sim_temp = [],[]
                for j in range(len(simulation_im)):
                    sim_temp.append(simulation_im[j][i])
                    bias_temp.append(np.log(self.metric_reference[i]/simulation_im[j][i]))
                
                Bias[key_event][abcissa[i]] = bias_temp
                im_sim.append(np.mean(sim_temp))
                
                # - Error per event
                if calibrate.periods[i] >= calibrate.periods_calibration[0] and calibrate.periods[i] <= calibrate.periods_calibration[1]:
                    error += np.mean(bias_temp)**2

            # - Save event results
            IM_sim_list.append(im_sim)
            Sim_acc[key_event] = simulation_acc
            Sim_im[key_event] = simulation_im
            Sim_dist[key_event] = distance
            Error[key_event] = error
            
            # - Stats
            if calibrate.stats_boolean == True:
                print(key_event + ' - Distance: ' + str(distance) + ' - Error: ' + str(error))
        
        # ------------ EVENT ARRAY AGGREGATED BIAS AND ERROR  ---------------- #   
        ## - ERROR
        # - Distance bias
        distances, err = [], []
        for key_event in Error:
            distances.append(Sim_dist[key_event])
            err.append(Error[key_event])
            
        errors = [errors_i for _,errors_i in sorted(zip(distances,err))]
        distances.sort()
        
        x = np.array(distances).reshape((-1, 1))
        y = np.array(errors)
        model = LinearRegression().fit(x, y)
        slope_error = model.coef_[0]
        b_error = model.intercept_

        if calibrate.error_type == 1:
            """
            Squared root of the mean suquared of the errors
            """
            error_aggregated = 0.0
            for key in Error:
                error_aggregated += Error[key]**2.
            error_aggregated = (error_aggregated/len(Error))**0.5
        
        elif calibrate.error_type == 2:
            """
            Maximum error
            """
            error_aggregated = 0.0
            for key in Error:
                if Error[key] > error_aggregated:
                    error_aggregated = Error[key]
            error_aggregated = (error_aggregated)**0.5
        
        elif calibrate.error_type == 3:
            """
            Combined error.  Bias slope (distance vs error) and maximum error
            """
            # - Frequency related error
            freq_error = 0.0
            for key in Error:
                freq_error += Error[key]**2.
            freq_error = (freq_error/len(Error))**0.5
            
            # - Aggregate errors
            error_aggregated = freq_error*(1 + abs(slope_error))
            
            
        # - BIAS
        bias_per_abcissa = {}
        for f in abcissa:
            bias_per_abcissa[f] = {}
            temp,freq = [],[]
            for key in Bias:
                for i in range(len(Bias[key][f])):
                    temp.append(Bias[key][f][i])
                    freq.append(f)
            
            bias_per_abcissa[f] = {'abcissa':freq,'bias':temp,'bias_mean':np.mean(temp),'bias_std':np.std(temp)}

        # - Plot the bias for the iteration
        ref_bias_plot = self.output.bias_per_iteration(abcissa,bias_per_abcissa,xlegend,
                                       save_path=self.id + '//bias_' + ' iteration ' + str(self.n) + '.jpg')        
        
        # ------------ SAVE ITERATION RESULTS  ---------------- #
        self.points.append(X)
        self.sim_im.append(Sim_im)
        self.sim_r_rup.append(Sim_dist)
        self.sim_acc.append(Sim_acc)
        self.error.append(error_aggregated)
        self.bias_aggregated.append(bias_per_abcissa)
        self.bias_individual.append(Bias)

        
        # ------------ BEST SOLUTION   ---------------- #
        if error_aggregated <= self.error_min:
            self.n_not_improv = 0
            self.error_min = error_aggregated
            self.best_performance['sim_acc'] = Sim_acc
            self.best_performance['sim_im'] = Sim_im
            self.best_performance['sim_r_rup'] = Sim_dist
            self.best_performance['bias_info'] = bias_per_abcissa
            self.best_performance['bias_individual'] = Bias
                
            ### - Aggregated distance bias 
            title = 'Aggregated distance bias'
            self.output.distance_bias_plot(distances,errors,b_error,slope_error,title=title,save_path= self.id + '//' + title + '.jpg')
            


            ### - Plot the average IM for the best solution
            self.IM_sim = self.avg_IM(IM_sim_list)
            title = str(self.id) + ' iteration ' + str(self.n) + ' - IM comparison'
            self.output.avg_IM_comparison(abcissa,self.IM_sim,self.IM_reference,xlegend,ylegend,logscale=calibrate.logscale,
                                          title=title,save_path=self.id + '//' + title + '.jpg')
        else:
            self.n_not_improv += 1
            
        # ------------ VARIABLE EXPLORATION PLOT  ---------------- #
        trial = np.linspace(0,self.n,self.n+1)
        for key in self.tried_variables:
            # Save name check
            key_save = ''
            for l in key:
                if l == '_':
                    pass
                else:
                    key_save += l
            self.output.variable_exploration(key,calibrate.c_map,trial,self.tried_variables[key],self.error,save_path = self.id + '//' + key_save +'.jpg')
        
        # ------------ ITERATION PERFORMANCE INFO  ---------------- #
        t3 = time.time()    
        running_time = t3-t0
        
        if calibrate.stats_boolean == True:
            print('Iteration time = ' + str("%.3f" % round(running_time,3)) + 's')
            print('Error = ' + str(error_aggregated))
        
        # - Log up the process
        info = info + ' Error = ' + str(error_aggregated)
        self.output.process_log(self.id,message=info)
        return np.array(error_aggregated)
 ##############################################################################################           
    def sa_single(self,X):

        t0 = time.time()
        # --------------------------------- #
        self.n += 1
        self.output.process_log(self.id,message='#### - Iteration #' + str(self.n) + ' - ###')
        # --------------------------------- #   
        # - Assign the trial value to the parameters
        info = ''
        for parameter,x in zip(self.opt_param,X) :
            self.tried_variables[parameter].append(x)
            info = info + parameter + ' = ' + "%.2f" % x + ', '
            print(parameter + ' = ' + str(x))
            self.input_parameters[parameter] = x
            
        # --------------------------------- #   
        # - Define abcissa based on calobration metric
        Error = 0.0
        abcissa = calibrate.periods
        xlegend = 'Periods [s]'
        ylegend = 'Sa [cm/s/s]'
            
        # ------------ SIMULATION AND INDIVIDUAL EVENT BIAS ---------------- # 
        bias_per_abcissa = {}

        if 'Depth' in self.observed:
            self.input_parameters['Depth'] =  self.observed['Depth']
        if 'Alpha' in self.observed:
            self.input_parameters['Alpha'] =  self.observed['Alpha']   
        if 'Betha' in self.observed:
            self.input_parameters['Betha'] =  self.observed['Betha']
        if 'vs' in self.observed:
            self.input_parameters['vs'] =  self.observed['vs']   
        if 'vp' in self.observed:
            self.input_parameters['vp'] =  self.observed['vp']
        if 'hypocenter' in self.observed:
            self.input_parameters['hypocenter'] =  self.observed['hypocenter']
        self.input_parameters['site'] = self.observed['site']
        
        # - Simulate and compute IM    
        Mw, component = self.observed[key_event]['Mw'],self.input_parameters['component']
        simulation_acc,simulation_im,distance = simulation_acc,simulation_im,distance = sim_spectrum_single(Mw,self.input_parameters,calibrate.n_sim,component,
                                                                                calibrate.dt,abcissa,calibrate.damping,calibrate.b,
                                                                                calibrate.delta_az,calibrate.pp)


        # Compute biases
        im_sim = []
        for i in range(len(abcissa)):
            bias_temp,sim_temp,pp = [],[], []
            for j in range(calibrate.n_sim):
                sim_temp.append(metric_sim[j][i])
                bias_temp.append(np.log(self.IM_reference[i]/metric_sim[j][i]))
                pp.append(abcissa[i])
                
            bias_per_abcissa[abcissa[i]] = {'abcissa':pp,'bias':bias_temp,'bias_mean':np.mean(bias_temp),'bias_std':np.std(bias_temp)}

            im_sim.append(np.mean(sim_temp))
            
            # - Error
            if calibrate.periods[i] >= calibrate.periods_calibration[0] and calibrate.periods[i] <= calibrate.periods_calibration[1]:
                Error += np.mean(bias_temp)**2

        # - Plot the bias for the iteration
        ref_bias_plot = self.output.bias_per_iteration(abcissa, bias_per_abcissa,xlegend,
                                       save_path=self.id + '//bias_' + ' iteration ' + str(self.n) + '.jpg')        
        
        
        # ------------ SAVE ITERATION RESULTS  ---------------- #
        self.points.append(X)
        self.sim_im.append(simulation_im)
        self.sim_acc.append(simulation_acc)
        self.error.append(Error)
        self.bias_aggregated.append(bias_per_abcissa)
        self.bias_individual.append(bias_per_abcissa)

        # ------------ BEST SOLUTION  ---------------- #
        if Error <= self.error_min:
            self.n_not_improv = 0
            self.error_min = Error
            self.best_performance['sim_acc'] = simulation_acc
            self.best_performance['sim_im'] = simulation_acc
            self.best_performance['bias_info'] = bias_per_abcissa
            self.best_performance['bias_individual'] = bias_per_abcissa


            ### - Plot the average IM for the best solution
            self.IM_sim = self.avg_IM(metric_sim)
            title = str(self.id) + ' iteration ' + str(self.n) + ' - IM comparison'
            self.output.avg_IM_comparison(abcissa,self.IM_sim,self.IM_reference,xlegend,ylegend,logscale=calibrate.logscale,
                                          title=title,save_path=self.id + '//' + title + '.jpg')

        else:
            self.n_not_improv += 1
            
        # ------------ VARIABLE EXPLORATION PLOT  ---------------- #
        for key in self.tried_variables:
            # Safename check
            key_save = ''
            for l in key:
                if l != '_':
                    key_save += l
            self.output.variable_exploration(key,calibrate.c_map,self.tried_variables[key],self.error,save_path = self.id + '//' + key_save +'.jpg')
        
        # ------------ ITERATION PERFORMANCE INFO  ---------------- #
        t3 = time.time()    
        running_time = t3-t0
        
        if calibrate.stats_boolean == True:
            print('Iteration time = ' + str("%.3f" % round(running_time,3)) + 's')
            print('Error = ' + str(Error))
        
        # - Log up the process
        info = info + ' Error = ' + str(Error)
        self.output.process_log(self.id,message=info)
        return np.array(Error)
 
 ##############################################################################################
  ##############################################################################################           
    def EXSIM_sa_multiple(self,X):
        from Otarola_Parallel import EQ
        t0 = time.time()
        # --------------------------------- #
        self.n += 1
        self.output.process_log(self.id,message='#### - Iteration #' + str(self.n) + ' - ###')
        # --------------------------------- #   
        # - Assign the trial value to the parameters
        info = ''
        for parameter,x in zip(self.opt_param,X) :
            self.tried_variables[parameter].append(x)
            info = info + parameter + ' = ' + "%.2f" % x + ', '
            print(parameter + ' = ' + str(x))
            self.input_parameters[parameter] = x
            
        # --------------------------------- #   
        # - Define abcissa based on calobration metric
        IM_sim_list = []
        abcissa = calibrate.periods
        xlegend = 'Periods [s]'
        ylegend = 'Sa [cm/s/s]'
            
        # ------------ SIMULATION AND INDIVIDUAL EVENT BIAS ---------------- #   
        Error, Bias, Sim_dist, Sim_acc, Sim_im = {},{},{},{},{}
        for key_event in self.observed:
            error = 0.0
            Bias[key_event] = {}
            
            # - Introduce event features
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
            self.input_parameters['site'] = self.observed[key_event]['site']
            
            # - Simulate and compute IM
            Mw, component = self.observed[key_event]['Mw'], self.input_parameters['component']
            simulation_acc, simulation_im, distance = EXSIM_sa_multiple(Mw, self.input_parameters, calibrate.n_sim, component,
                                                         calibrate.dt, abcissa, calibrate.damping)
            # - Compute biases
            im_sim = []
            for i in range(len(abcissa)):
                bias_temp,sim_temp = [],[]
                for j in range(len(simulation_im)):
                    sim_temp.append(simulation_im[j][i])
                    bias_temp.append(np.log(self.metric_reference[key_event][i]/simulation_im[j][i]))
                
                Bias[key_event][abcissa[i]] = bias_temp
                im_sim.append(np.mean(sim_temp))
                
                # - Error per event
                if calibrate.periods[i] >= calibrate.periods_calibration[0] and calibrate.periods[i] <= calibrate.periods_calibration[1]:
                    error += np.mean(bias_temp)**2

            # - Save event results
            IM_sim_list.append(im_sim)
            Sim_acc[key_event] = simulation_acc
            Sim_im[key_event] = simulation_im
            Sim_dist[key_event] = distance
            Error[key_event] = error
            
            # - Stats
            if calibrate.stats_boolean == True:
                print(key_event + ' - Distance: ' + str(distance) + ' - Error: ' + str(error))
        
        # ------------ EVENT ARRAY AGGREGATED BIAS AND ERROR  ---------------- #   
        ## - ERROR
        # - Distance bias
        distances, err = [], []
        for key_event in Error:
            distances.append(Sim_dist[key_event])
            err.append(Error[key_event])
            
        errors = [errors_i for _,errors_i in sorted(zip(distances,err))]
        distances.sort()
        
        x = np.array(distances).reshape((-1, 1))
        y = np.array(errors)
        model = LinearRegression().fit(x, y)
        slope_error = model.coef_[0]
        b_error = model.intercept_

        if calibrate.error_type == 1:
            """
            Squared root of the mean suquared of the errors
            """
            error_aggregated = 0.0
            for key in Error:
                error_aggregated += Error[key]**2.
            error_aggregated = (error_aggregated/len(Error))**0.5
        
        elif calibrate.error_type == 2:
            """
            Maximum error
            """
            error_aggregated = 0.0
            for key in Error:
                if Error[key] > error_aggregated:
                    error_aggregated = Error[key]
            error_aggregated = (error_aggregated)**0.5
        
        elif calibrate.error_type == 3:
            """
            Combined error.  Bias slope (distance vs error) and maximum error
            """
            # - Frequency related error
            freq_error = 0.0
            for key in Error:
                freq_error += Error[key]**2.
            freq_error = (freq_error/len(Error))**0.5
            
            # - Aggregate errors
            error_aggregated = freq_error*(1 + abs(slope_error))
            
            
        # - BIAS
        bias_per_abcissa = {}
        for f in abcissa:
            bias_per_abcissa[f] = {}
            temp,freq = [],[]
            for key in Bias:
                for i in range(len(Bias[key][f])):
                    temp.append(Bias[key][f][i])
                    freq.append(f)
            
            bias_per_abcissa[f] = {'abcissa':freq,'bias':temp,'bias_mean':np.mean(temp),'bias_std':np.std(temp)}

        # - Plot the bias for the iteration
        ref_bias_plot = self.output.bias_per_iteration(abcissa,bias_per_abcissa,xlegend,
                                       save_path=self.id + '//bias_' + ' iteration ' + str(self.n) + '.jpg')        
        
        # ------------ SAVE ITERATION RESULTS  ---------------- #
        self.points.append(X)
        self.sim_im.append(Sim_im)
        self.sim_r_rup.append(Sim_dist)
        self.sim_acc.append(Sim_acc)
        self.error.append(error_aggregated)
        self.bias_aggregated.append(bias_per_abcissa)
        self.bias_individual.append(Bias)

        
        # ------------ BEST SOLUTION   ---------------- #
        if error_aggregated <= self.error_min:
            self.n_not_improv = 0
            self.error_min = error_aggregated
            self.best_performance['sim_acc'] = Sim_acc
            self.best_performance['sim_im'] = Sim_im
            self.best_performance['sim_r_rup'] = Sim_dist
            self.best_performance['bias_info'] = bias_per_abcissa
            self.best_performance['bias_individual'] = Bias
                
            ### - Aggregated distance bias 
            title = 'Aggregated distance bias'
            self.output.distance_bias_plot(distances,errors,b_error,slope_error,title=title,save_path= self.id + '//' + title + '.jpg')
            


            ### - Plot the average IM for the best solution
            self.IM_sim = self.avg_IM(IM_sim_list)
            title = str(self.id) + ' iteration ' + str(self.n) + ' - IM comparison'
            self.output.avg_IM_comparison(abcissa,self.IM_sim,self.IM_reference,xlegend,ylegend,logscale=calibrate.logscale,
                                          title=title,save_path=self.id + '//' + title + '.jpg')
        else:
            self.n_not_improv += 1
            
        # ------------ VARIABLE EXPLORATION PLOT  ---------------- #
        trial = np.linspace(0,self.n,self.n+1)
        for key in self.tried_variables:
            # Save name check
            key_save = ''
            for l in key:
                if l == '_':
                    pass
                else:
                    key_save += l
            self.output.variable_exploration(key,calibrate.c_map,trial,self.tried_variables[key],self.error,save_path = self.id + '//' + key_save +'.jpg')
        
        # ------------ ITERATION PERFORMANCE INFO  ---------------- #
        t3 = time.time()    
        running_time = t3-t0
        
        if calibrate.stats_boolean == True:
            print('Iteration time = ' + str("%.3f" % round(running_time,3)) + 's')
            print('Error = ' + str(error_aggregated))
        
        # - Log up the process
        info = info + ' Error = ' + str(error_aggregated)
        self.output.process_log(self.id,message=info)
        return np.array(error_aggregated)
##############################################################################################         
    def opt_resume(self):
        self.opt_minimize()
        
 ##############################################################################################         
    def opt_minimize(self):
        if self.cal_metric == 'fas':
            self.bayesian_optimisation(self.iterations, self.fas_multiple, self.bounds, x0=None, n_pre_samples=calibrate.n_pre_samples,gp_params=None, random_search=False, alpha=1e-5, epsilon=1e-7,maximize=False)
        elif self.cal_metric == 'spectra':
            self.bayesian_optimisation(self.iterations, self.sa_multiple, self.bounds, x0=None, n_pre_samples=calibrate.n_pre_samples,gp_params=None, random_search=False, alpha=1e-5, epsilon=1e-7,maximize=False)
        elif self.cal_metric == 'EXSIM_spectra':
            self.bayesian_optimisation(self.iterations, self.EXSIM_sa_multiple, self.bounds, x0=None, n_pre_samples=calibrate.n_pre_samples,gp_params=None, random_search=False, alpha=1e-5, epsilon=1e-7,maximize=False)
        elif self.cal_metric == 'gmpe':
            self.bayesian_optimisation(self.iterations, self.gmpe_multiple, self.bounds, x0=None, n_pre_samples=calibrate.n_pre_samples,gp_params=None, random_search=False, alpha=1e-5, epsilon=1e-7,maximize=False)
        elif self.cal_metric == 'spectrum':
            self.bayesian_optimisation(self.iterations, self.black_box_spectrum, self.bounds, x0=None, n_pre_samples=calibrate.n_pre_samples,gp_params=None, random_search=False, alpha=1e-5, epsilon=1e-7,maximize=False)
        
        # - Gather information of all iterations
        self.performances['points'] = self.points
        self.performances['sim_acc'] = self.sim_acc
        self.performances['error'] = self.error
        self.performances['bias_aggregated'] = self.bias_aggregated
        self.performances['bias_individual'] = self.bias_individual
        
        # - Plot trial - error evolution
        n = len(self.points)
        trial = np.arange(0,n,1)
        title = str(self.id) + ' Error evolution'
        self.output.error_evolution(trial,self.error,save_path=self.id + '//' + title + '.jpg',title=title)

        ## - Save results
        result = {'performances':self.performances,'best_performance':self.best_performance,'R_rup':self.distances}
        with open(self.id + '//' + 'results.pkl', 'wb') as f:  # Python 3: open(..., 'wb')
            pickle.dump(result, f)
        f.close() 
