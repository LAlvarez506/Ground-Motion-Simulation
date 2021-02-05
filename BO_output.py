import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from datetime import datetime
from mpl_toolkits.axes_grid.inset_locator import (inset_axes, InsetPosition,
                                                  mark_inset)

import numpy as np
class bo_output:
    def __init__(self):
        pass
# ------------------------------------------------------------------------
    def process_log(self,analysis_id,message=None,init=False,init_dict=None):
        file_name  = analysis_id + '//' + 'process_log.txt'
        if init is True:
            now = datetime.now()
            date_string = now.strftime("%d/%m/%Y %H:%M:%S")
            f = open(file_name, "w+")
            f.write('###########################################\n')
            f.write('### - Bayesian Optimization Algorithm - ###\n')
            f.write('####### - Luis Alvarez - 2019 - ###########\n')
            f.write('Project = ' + analysis_id +'\n')
            f.write(date_string +'\n')
            if init_dict is not None:
                for key in init_dict:
                    f.write(str(key) + ': ' + str(init_dict[key]) +'\n')
            f.write('###########################################\n')
            f.close()
        else:
            f = open(file_name, "a+")
            f.write(message + '\n')
            f.close()  
# ------------------------------------------------------------------------
    def avg_IM(self,x,IM_reference,im_reference,logscale,xlegend,ylegend):
        plt.ioff()
        fig = plt.figure()
        for i in range(len(im_reference)):
            plt.plot(x,im_reference[i],color='0.75',figure=fig)
        plt.plot(x,IM_reference,figure=fig)
        plt.xlabel(xlegend,figure=fig)
        plt.ylabel(ylegend,figure=fig)
        if logscale == True:
            plt.xscale("log")
            plt.yscale("log")
        plt.grid(figure=fig)
        plt.close(fig)
        ref_plot = fig
        return fig
# ------------------------------------------------------------------------
    def bias_per_iteration(self, abcissa, Bias, duration=None, arias=None, save_path=None):
        plt.ioff()
        Bias_Mean, Bias_Std = np.zeros(len(abcissa)), np.zeros(len(abcissa))
        i = 0
        for ab in abcissa:
            points = []
            for key in Bias:
                for j in range(len(Bias[key][ab])):
                    points.append(Bias[key][ab][j])
            Bias_Mean[i] = np.mean(points)
            Bias_Std[i] = np.std(points)
            i += 1
            
        if duration is None:
            plt.figure(1, figsize=(8,6))
            plt.plot(abcissa, Bias_Mean, color='darkorange', label='Sim - mean', zorder=100)
            plt.fill_between(abcissa, Bias_Mean, Bias_Mean + Bias_Std, color='0.75')
            plt.fill_between(abcissa, Bias_Mean, Bias_Mean - Bias_Std, color='0.75')
            plt.plot([abcissa.min(), abcissa.max()], [0., 0.], color='k', linestyle='dashed')
            plt.xscale('log')
            plt.grid()
            plt.xlabel(r"$\mathrm{Period}$" + ' ' +r"$\mathrm{[s]}$",fontsize=18)
            plt.ylabel(r"$\mathrm{Residual}$",fontsize=18)
            plt.tick_params(axis='x', labelsize=15)
            plt.tick_params(axis='y', labelsize=15)
            plt.xlim([abcissa.min(), abcissa.max()])
            plt.ylim(-3., 3.)
            plt.savefig(save_path)
            plt.close()
            
        else:
            temp_duration, temp_arias = [], []
            for key in duration:
                for i in range(len(duration[key])):
                    temp_duration.append(duration[key][i])
                    temp_arias.append(arias[key][i])
            
            D5_95 = {'mean':np.mean(temp_duration), 'std':np.std(temp_duration)}
            Arias = {'mean':np.mean(temp_arias), 'std':np.std(temp_arias)}
            y = [D5_95['mean'], Arias['mean']]
            y_err = [D5_95['std'], Arias['std']]
            
            fig, ax1 = plt.subplots(figsize=(8,6))
            ax1.plot(abcissa, Bias_Mean, color='darkorange', label='Sim - mean', zorder=100)
            ax1.fill_between(abcissa, Bias_Mean, Bias_Mean + Bias_Std, color='0.75')
            ax1.fill_between(abcissa, Bias_Mean, Bias_Mean - Bias_Std, color='0.75')
            ax1.plot([abcissa.min(), abcissa.max()], [0., 0.], color='k', linestyle='dashed')
            ax1.set_xscale('log')
            ax1.set_xlabel(r"$\mathrm{Period}$" + ' ' +r"$\mathrm{[s]}$",fontsize=18)
            ax1.set_ylabel(r"$\mathrm{Residual}$",fontsize=18)

            ax1.grid()
            ax1.set_xlim([abcissa.min(), abcissa.max()])
            ax1.set_ylim(-3., 3.)
            ax1.tick_params(axis='x', labelsize=15)
            ax1.tick_params(axis='y', labelsize=15)

            ax2 = plt.axes([0,0,1,1])
            # Manually set the position and relative size of the inset axes within ax1
            ip = InsetPosition(ax1, [0.7,0.7,0.25,0.25])
            ax2.set_axes_locator(ip)
            x2, labels = [1, 3], [r"$\mathrm{D_{5-95}}$", r"$\mathrm{Arias}$"]
            ax2.errorbar(x2, y, yerr=y_err, fmt='.k', ms=10)
            ax2.plot([0, 4.], [0., 0.], linewidth=2, color='k', linestyle='dashed')
            ax2.grid()
            # Some ad hoc tweaks.
            ax2.set_yticks(np.arange(-3,3,1.))
            ax2.set_xticks(x2)
            ax2.set_xticklabels(labels, fontsize=14)
            plt.savefig(save_path)
            plt.close()

        return Bias_Mean, Bias_Std

#---------------------------------------------------------------------------------------------------------------------------------
    def plot_individual_comp(self, reference, periods, acc, sa, dt, save_path=None):
        error = []
        for i in range(len(sa)):
            # - Because periods for both spectra are the same
            e = np.sum(abs(sa[i] - reference))
            error.append(e)
        
        error = np.array(error)
        lucky_index = np.where(error == error.min())[0][0]

        fig = plt.figure(figsize=(10,8))
        grid = plt.GridSpec(3, 2, wspace=0.25, hspace=0.3)
        ax1 = plt.subplot(grid[0, 0])
        ax2 = plt.subplot(grid[0, 1])
        ax3 = plt.subplot(grid[1:, 0:])

        ## - NS - ##
        acc_ns= acc[lucky_index][0]
        time = np.linspace(0., len(acc_ns)*dt, len(acc_ns))
        ax1.plot(time, acc_ns, color='darkorange', linewidth=0.75, label='NS-sim')
        ax1.set_ylabel(r"$\mathrm{acc}$" + ' ' + r"$\mathrm{[cm/s/s]}$", fontsize=16)
        ax1.legend(loc='upper center', bbox_to_anchor=(0.8, 1.00), prop={"size":14})
        ax1.tick_params(axis='x', labelsize=14)
        ax1.tick_params(axis='y', labelsize=14)

        ## - EW - ##
        acc_ew = acc[lucky_index][1]
        time = np.linspace(0., len(acc_ew)*dt, len(acc_ew))
        ax2.plot(time, acc_ew, color='darkorange', linewidth=0.75, label='EW-sim')
        ax2.set_ylabel(r"$\mathrm{acc}$" + ' ' + r"$\mathrm{[cm/s/s]}$", fontsize=16)
        ax2.legend(loc='upper center', bbox_to_anchor=(0.8, 1.00), prop={"size":14})
        ax2.tick_params(axis='x', labelsize=14)
        ax2.tick_params(axis='y', labelsize=14)

        ## - SPECTRA - ##
        ax3.plot(periods, reference, 'b-', label='GM - ref', zorder=100)
        ax3.plot(periods, np.mean(sa, axis=0), color='darkorange', label='GM - mean', zorder=100)
        for i in range(len(sa)):
                ax3.plot(periods, sa[i], color='0.75')

        ax3.set_xscale('log')
        ax3.set_yscale('log')
        ax3.legend(prop={"size":14})
        ax3.grid()
        ax3.tick_params(axis='x', labelsize=14)
        ax3.tick_params(axis='y', labelsize=14)
        ax3.set_xlabel(r"$\mathrm{Period}$" + ' ' +r"$\mathrm{[s]}$",fontsize=16)
        ax3.set_ylabel(r"$\mathrm{S_a}$" + ' ' +r"$\mathrm{[cm/s/s]}$",fontsize=16)
        ax3.set_xlim([0.01, 3.])
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()
# ------------------------------------------------------------------------
    def frequency_distance_bias_plot(self,dist,mean,std,title=None,save_path=None):
        plt.ioff()
        fig, ax1 =  plt.subplots(1,1)
        ax1.plot([0.,300.],[0.,0.], 'k--')
        ax1.errorbar(dist, mean, yerr=std,color='darkorange',fmt='o')
        ax1.set_xlabel(r"$\mathrm{Distance}$" + ' ' + r"$\mathrm{[km]}$",figure=fig,fontsize=16)
        ax1.set_ylabel(r"$\mathrm{Divergence}$",figure=fig,fontsize=16)
        ax1.set_xscale('log')
        ax1.set_ylim([-3.5,3.5])
        ax1.set_xticks([10.,100.,200.])
        ax1.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
        plt.savefig(save_path)
        plt.close(fig)

# ------------------------------------------------------------------------
    def distance_bias_plot(self,dist,error,intercept,slope,title=None,save_path=None):
        distances = np.array(dist)
        line = slope*distances + intercept

        plt.ioff()
        fig = plt.figure(figsize=(8, 6))
        plt.scatter(dist,error,marker='s',color='darkorange')
        plt.plot(distances,line,color='0.55',label=r"$\mathrm{y}$" + ' = '  + str("%.3f" % round(slope,3)) + r"$\mathrm{distance}$" + ' + '  + str("%.3f" % round(intercept,3))) 
        plt.xlabel(r"$\mathrm{Distance}$" + ' ' + r"$\mathrm{[km]}$",figure=fig,fontsize=16)
        plt.ylabel(r"$\mathrm{Divergence}$",figure=fig,fontsize=16)
        plt.legend(prop={"size":14})
        plt.grid(figure=fig)
        plt.savefig(save_path)
        plt.close(fig)
# ------------------------------------------------------------------------
    def centralFrequency_histogram_plot(self,cf,title=None,save_path=None,bin_width=None):
        if bin_width is None:
            bw = 10
        else:
            bw = bin_width
        plt.ioff()
        plt.figure()
        if title is not None:
            plt.title(title)
        n, bins, patches = plt.hist(cf,bw)
        plt.xlabel('log(cf_ref/cf_sim) [Hz]')
        plt.grid()
        plt.savefig(save_path)
        plt.close()        
# ------------------------------------------------------------------------
    def avg_IM_comparison(self,abcissa,IM_sim,IM_reference,xlegend,ylegend,logscale=None,title=None,save_path=None):
        plt.ioff()
        fig = plt.figure()
        if title is not None:
            plt.title(title,figure=fig)
        plt.plot(abcissa,IM_sim,label='Sim',figure=fig)
        plt.plot(abcissa,IM_reference,label='Ref',figure=fig)
        if logscale is True:
            plt.xscale("log")
            plt.yscale("log")
            plt.xlim([0.01, max(abcissa)])
        plt.xlabel(xlegend,figure=fig)
        plt.ylabel(ylegend,figure=fig)
        plt.grid(figure=fig)
        plt.legend(frameon=False, loc='lower center', ncol=2)
        plt.savefig(save_path)
        plt.close(fig)  
# ------------------------------------------------------------------------
    def variable_exploration(self,variable,c_map,trial,tried_variables,error,save_path=None):
        n = len(tried_variables)
        trial = np.arange(0,n,1)
        cm = plt.cm.get_cmap(c_map)
        plt.ioff()
        plt.figure()
        sc = plt.scatter(trial,tried_variables,c=error,vmin=0,vmax=max(error),cmap=cm)
        plt.ylabel(variable)
        plt.xlabel('Iteration')
        plt.margins(0.05)
        plt.grid()
        plt.colorbar(sc)
        plt.savefig(save_path)
        plt.close() 
# ------------------------------------------------------------------------
    def error_evolution(self,trial,error,save_path=None,title=None):
        plt.ioff()
        fig = plt.figure()
        if title is not None:
            plt.title(title,figure=fig)
        plt.plot(trial,error,figure=fig)
        plt.grid(figure=fig)
        plt.xlabel('Iteration')
        plt.ylabel('Error')
        plt.savefig(save_path)
        plt.close()
