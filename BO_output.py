import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from datetime import datetime
from matplotlib import rc
rc('text', usetex=True)
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
    def bias_per_iteration(self,abcissa,bias_per_abcissa,xlegend,save_path=None):
        plt.ioff()
        fig = plt.figure(figsize=(8, 6))
        for ab in abcissa:
            plt.errorbar(ab, bias_per_abcissa[ab]['bias_mean'], yerr=bias_per_abcissa[ab]['bias_std'],color='b',fmt='s',zorder=10)
            plt.scatter(bias_per_abcissa[ab]['abcissa'],bias_per_abcissa[ab]['bias'],color='0.75',zorder=1)

        plt.xlabel(xlegend,figure=fig,fontsize=16)
        plt.ylabel(r"$\mathrm{Divergence}$",figure=fig,fontsize=16)
        plt.xscale("log")
        plt.grid(figure=fig)
        plt.savefig(save_path)
        plt.close(fig)
        ref_bias_plot = fig

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
