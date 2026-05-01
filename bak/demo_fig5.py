import numpy as np
import matplotlib.pyplot as plt
import os
from os import mkdir
from os.path import isdir

######### Folder default for data storage. Can be overriden by optional argument to script #########
folder = "data/fig5/"
folder_thermal = folder + "data_thermal/"
folder_depolar = folder + "data_depolar/"
mappings = ("JW","BK","Parity")
file_structure_thermal = folder_thermal + "H2_6-31G_molecule_ShadowBernstein_method_{}_beta_{}.txt"
file_structure_depolar = folder_depolar + "energy_abs_diffs_{}_p_{}.txt"
####################################################################################################
def pretty_print_exponents(pre,exp):
    out = r"$"
    if pre is None:
        if exp is None:
            out += "1$"
        else:
            out += "10^{"+str(exp)+"}$"
        return out
    else:
        out +="{:.2f}".format(pre)
    if exp is None:
        out += "$"
    else:
        out += "\cdot 10^{"+str(exp)+"}$"
    return out


if __name__ == "__main__":
    # create temporary folder for storing outputs
    if not isdir("generated_figures"):
        mkdir("generated_figures")
        
    # get depolar data
    files = [file for file in os.listdir(folder_depolar) if file.find("H2_") >= 0 and file.find("energy") < 0]
    p = np.round(np.arange(0,1.1,0.1),1)
    E_GS = 2
    offset = 1
    p_plot = np.linspace(0,1,100)
        
    # get thermal data
    files = [file for file in os.listdir(folder_thermal) if file.find("H2_") >= 0 and file.find("energies") < 0 and file.find("beta")>=0]
    betas = []
    betas_print = []
    for file in files:
        betas.append(float(file[:-4].split("_")[-1]))
    betas = np.unique(betas)
    for beta in betas:
        exponent = int(np.floor(np.log10(beta)))
        prefactor = None if abs(10**exponent - beta) < 1e-3 else np.round(beta*10**(-exponent),2)
        if exponent == 0:
            exponent = None
        betas_print.append((prefactor,exponent))
    
    beta_cont, Evals_cont = np.loadtxt(folder_thermal+"plot_data_continous.txt",unpack=True,skiprows=2)
    beta_samp, Evals_samp = np.loadtxt(folder_thermal+"plot_data_sampled.txt",unpack=True,skiprows=2)
    with open(folder_thermal+"plot_data_continous.txt","r") as f:
        E_GS = float(f.readline().strip().split(" ")[-1])
        offset = float(f.readline().strip().split(" ")[-1])
    
    plt.figure(figsize=(16,8))

    # add lineplot for thermal energies and from where sampling took place
    ax = plt.subplot(241)
    plt.semilogx(beta_cont,Evals_cont+offset)
    plt.hlines(offset,1e-4,1,colors="black",linestyles="dotted")
    plt.hlines(E_GS,1,1e3,colors="black",linestyles="dotted")
    plt.xlim(0.6*beta_cont[0],beta_cont[-1])
    plt.xlabel(r"Inverse temperature $\beta$",fontsize="xx-large")
    plt.xticks(np.logspace(-4,2,4),fontsize="x-large")
    plt.ylabel(r"$E = \mathrm{Tr}[H\rho]$ [Ha]",fontsize="xx-large")
    #ax.yaxis.set_label_position("right")
    #ax.yaxis.set_ticks_position("right")
    plt.yticks(np.arange(-1,1.1),["-1","0","1"],fontsize="x-large")
    for x,y,s in zip((7,3e-1),(offset,E_GS),("offset",r"$E_\mathrm{GS}$")):
        plt.text(x,y,s,fontsize="x-large",horizontalalignment="center",verticalalignment="center")
    # add sampled data points to line
    for beta,E in zip(beta_samp,Evals_samp):
        plt.plot(beta,E+offset,marker="*",markersize=10)
    plt.grid()

    for i,map_name in enumerate(mappings):
        if i==0:
            ax0 = plt.subplot(242+i)
            ax = ax0
        else:
            ax = plt.subplot(242+i,sharey=ax)
        plt.loglog([],[],label=r"$\beta =$",linewidth=0)
        for beta,beta_pr in zip(betas,betas_print):
            N,rmse,std, _ = np.loadtxt(file_structure_thermal.format(map_name,beta),skiprows=1,unpack=True)
            plt.errorbar(N,rmse,yerr=std,label=pretty_print_exponents(*beta_pr))
        plt.text(0.075,0.9,map_name,transform=ax.transAxes,fontsize="xx-large",bbox=dict(facecolor='white', edgecolor="white", alpha=0.2))
        plt.ylim(1e-3,1e-1)
        plt.xlim(0.95e4,1.05e5)

        if i==2:
            ax.yaxis.set_label_position("right")
            ax.yaxis.set_ticks_position("right")
            plt.ylabel(r"$\vert \hat E - E \vert$ [Ha]",fontsize="xx-large")
            #ax.yaxis.set_label_coords(-0.25,1.1)
            plt.yticks(fontsize="x-large")
        else:
            ax.yaxis.set_tick_params(length=0,width=0,labelsize=0)
            ax.yaxis.set_ticks_position("right")
        plt.xticks(fontsize="x-large")
        plt.grid(which="both",axis="x")
        plt.grid(which="both",axis="y")

    # add lineplot for dep. energies and from where sampling took place
    ax = plt.subplot(245)
    plt.plot(p_plot,(1-p_plot)*E_GS+p_plot*offset)
    plt.xlim(-0.05,1.05)
    plt.xlabel(r"Depolariz. parameter $p$",fontsize="xx-large")
    plt.xticks(fontsize="x-large")
    #plt.ylabel(r"$E_p = \mathrm{Tr}[H\rho_p]$",fontsize="xx-large")
    plt.yticks([offset,E_GS],[r"$E(p=1)$",r"$E(p=0)$"],fontsize="x-large")
    plt.ylim(offset - 0.5*(E_GS - offset), 1.15*E_GS)
    # add sampled data points to line
    for pval in p:
        plt.plot(pval,(1-pval)*E_GS+pval*offset,marker="*",markersize=10)
    plt.text(0.5,offset - 0.25*(E_GS - offset),r"$\rho_p = (1-p) \vert \psi \rangle \langle \psi \vert + \frac{p}{d}\ 1$",
             fontsize="xx-large",horizontalalignment="center",verticalalignment="center",bbox=dict(facecolor='white', edgecolor="white", alpha=0.1))
    plt.grid()
    for i,map_name in enumerate(mappings):
        if i==0:
            ax0 = plt.subplot(246+i)
            ax = ax0
        else:
            ax = plt.subplot(246+i,sharey=ax)
        plt.loglog()
        for pval in p:
            N,rmse,std, _ = np.loadtxt(file_structure_depolar.format(map_name,pval),unpack=True)
            plt.errorbar(N,rmse,yerr=std)
        plt.text(0.075,0.9,map_name,transform=ax.transAxes,fontsize="xx-large",bbox=dict(facecolor='white', edgecolor="white", alpha=0.2))
        plt.ylim(1e-3,1e-1)
        plt.xlim(0.95e4,1.05e5)

        if i==1:
            plt.xlabel("Number of measurements [log.]",fontsize="xx-large")
            ax.yaxis.set_tick_params(length=0,width=0,labelsize=0)
            ax.yaxis.set_ticks_position("right")
        if i==2:
            ax.yaxis.set_label_position("right")
            ax.yaxis.set_ticks_position("right")
            plt.ylabel(r"$\vert \hat E - E \vert$ [Ha]",fontsize="xx-large")
            plt.yticks(fontsize="x-large")
        else:
            ax.yaxis.set_tick_params(length=0,width=0,labelsize=0)
            ax.yaxis.set_ticks_position("right")
        plt.xticks(fontsize="x-large")
        plt.grid(which="both",axis="x")
        plt.grid(which="both",axis="y")

    plt.subplots_adjust(hspace=0.25,wspace=0.1)
    plt.savefig("generated_figures/fig5_demo.png",bbox_inches="tight")
    plt.show()