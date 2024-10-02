import numpy as np
import matplotlib.pyplot as plt
from shadowgrouping.measurement_schemes import N_delta
from shadowgrouping.molecules import available_molecules
from os import mkdir
from os.path import isdir
import json

##################
folder = "data/benchmarks/"
delta = 0.02

# plot specifics for bar plots
top_threshold = 1e8
width = 0.8
align = "center"

def load_json(filename):
    with open(filename,"r") as f:
        data = json.load(f)
    return data
methods = list(load_json(folder+"H2_6-31g.json")["H2_6-31g"]["JW"]["empirical"].keys())
width_single = width / (len(methods) + 1)
available_molecules = available_molecules[1:] # do not show data for H2 molecule comprised of 4 qubits
bar_pos = np.arange(len(available_molecules))[np.newaxis,:] + width_single*(np.arange(len(methods)+1)+-0.5*len(methods))[:,np.newaxis]
####################################################################
########### style params etc #######################################
####################################################################

threshold = int(np.ceil(N_delta(delta)))

def get_cmap(n, name='hsv'):
    '''Returns a function that maps each index in 0, 1, ..., n-1 to a distinct 
    RGB color; the keyword argument name must be a standard mpl colormap name.'''
    return plt.cm.get_cmap(name, n)

def fit_func(N,A,coeff):
    return A*N**(-coeff)

def get_N_estimates(A,coeff,epsilon=1.6e-3,deltaA=None,deltaCoeff=None):
    N_chem = (A/epsilon)**(1/coeff)
    exponent = int(np.log10(N_chem))
    if deltaA is not None or deltaCoeff is not None:
        logN   = np.log(N_chem)
        deltaN = 0.0
        deltaN += 0 if deltaA is None else np.exp(logN+np.log(deltaA)-np.log(A)-np.log(coeff))
        deltaN += 0 if deltaCoeff is None else np.exp(logN+np.log(deltaCoeff*logN/coeff))
        return N_chem, deltaN, exponent
    return N_chem, exponent

def insert_value_to_ticks(arr,val,threshold=0.1):
    """ Insert a given value <val> into a sorted array <arr>.
        Must lie within the range of the array, however, the first and last elements of <arr> are cut-off afterwards.
        When inserted, array values that are within <threshold> x array-range are thrown away.
        Returns the new filtered array with the value inserted at the last index.
        Also returns the corresponding indices used for sorting the new array.
    """
    assert np.allclose(arr,np.sort(arr)), "Please provide a sorted array."
    assert arr[0] <= val and arr[-1] >= val, "Please provide a value within the range of arr."
    # check whether values are close in range
    R = arr[-1] - arr[0]
    arr = arr[1:-1]
    kept = np.bitwise_or(arr >= val+threshold*R, arr <= val-threshold*R)
    new_arr = np.append(arr[kept],val)
    inds = np.argsort(new_arr)
    return new_arr, inds
####################################################################

if __name__ == "__main__":
    # create temporary folder for storing outputs
    if not isdir("generated_figures"):
        mkdir("generated_figures")
    
    benchmark_dict = load_json(folder+"NH3.json")
    Nrec = np.loadtxt(folder+"Nrec_NH3.txt")
    Nrec_OGM = np.loadtxt(folder+"Nrec_OGM_New.txt")
    fit_params = load_json(folder+"NH3_fit_params.json")
    
    threshold = 1e3
    x_max_log = 6
    y_min_log = -2

    filtered = Nrec >= threshold
    filtered_new = Nrec_OGM >= threshold
    N_plot = np.logspace(np.log10(threshold),x_max_log,100)

    plt.figure(figsize=(12,12))

    for j,(mapping,map_dict) in enumerate(benchmark_dict["NH3"].items()):
        keys = map_dict["provable"].keys()
        plt.subplot(331+j*3)
        plt.loglog()
        cmap = get_cmap(len(keys),"Accent")
        for i,method in enumerate(keys):
            label = method if j==1 else ""
            if method =="OverlappedGrouping":
                Nvals = Nrec_OGM
                filt_p = filtered_new
            else:
                Nvals = Nrec
                filt_p = filtered
            emp,std = map_dict["empirical"][method]
            emp,std = np.array(emp),np.array(std)
            popt,pcov = fit_params[mapping]["empirical"].get(method,([0,0], None))
            if popt[-1] > 0.05:
                print(method,":",popt[-1],"--> N_chem.acc.:", get_N_estimates(*popt)[1])
                temp = fit_func(N_plot,*popt)
                plt.plot(N_plot,temp,color=cmap(i),linestyle="dotted")

            plt.errorbar(Nvals,emp,yerr=std/10,label=label,ecolor=cmap(i),linewidth=0,elinewidth=5)
        plt.ylim(10**y_min_log,1)
        if j==1:
            plt.legend(fontsize="large")
            plt.ylabel("Root-mean-squared error [Ha]",fontsize="xx-large")
        plt.grid()
        plt.xlim(100,N_plot[-1])
        plt.yticks(fontsize="x-large")
        if j==2:
            plt.xlabel("No. meas. rounds $N$",fontsize="xx-large")
            plt.xticks(fontsize="x-large")
        else:
            plt.xticks([])
    
    for j,mapping in enumerate(benchmark_dict["NH3"].keys()):
        # load fit_params from file into arrays
        data   = np.full((len(methods)+1,len(available_molecules)),1e5)
        data_c = np.zeros((len(methods)+1,len(available_molecules)))
        data_Delta   = np.zeros_like(data)
        data_Delta_c = np.zeros_like(data_c)

        for i,method in enumerate(methods):
            for k,molecule in enumerate(available_molecules):
                fit_params  = load_json(folder+molecule+"_fit_params.json")
                temp = fit_params[mapping]["empirical"].get(method,None)
                if temp is not None:
                    data[i,k], data_Delta[i,k]     = get_N_estimates(*temp[0],1.6e-3,*np.sqrt(np.diag(temp[1])))[:2]
                    data_c[i,k], data_Delta_c[i,k] = temp[0][1], np.sqrt(temp[1][-1][-1])
                else:
                    data[i,k], data_c[i,k], data_Delta[i,k], data_Delta_c[i,k] = 4*[-1]


        plt.subplot(332+j*3)
        for c,deltaC,pos,method in zip(data_c,data_Delta_c,bar_pos,methods):
            if method == "Derandomization":
                continue
            plt.bar(pos, c, yerr=deltaC, color = cmap(np.argmax([method==key for key in keys])), width = width_single, capsize=5,
                edgecolor ='grey', label =method, zorder=3, align=align)

        if j==1:
            plt.text(2,0.6,'Fit exponent c', fontsize = "xx-large",horizontalalignment="center")
        if j==2:
            plt.xticks(np.arange(len(available_molecules)),[mol.split("_")[0] for mol in available_molecules],fontsize="x-large")
        else:
            plt.xticks([])
        plt.grid(axis="y",zorder=1)
        plt.ylim(0,0.7)
        plt.yticks([0.1,0.3,0.5],[])
            
        plt.subplot(333+j*3)
        for d,deltaD,pos,method in zip(data,data_Delta,bar_pos,methods):
            filtered = np.bitwise_and(d >= 0, d < top_threshold)
            if np.sum(filtered) == 0:
                continue
            plt.bar(pos[filtered], d[filtered], yerr=deltaD[filtered], color = cmap(np.argmax([method==key for key in keys])), width = width_single,
                    edgecolor ='grey', label =method, zorder=3, align=align, capsize=5)

        plt.yscale('log')
        plt.gca().yaxis.tick_right()
        plt.gca().yaxis.set_label_position("right")
        if j==1:
            plt.ylabel('No. meas. rounds to provably reach chem. acc.', fontsize = "xx-large")
        if j==2:
            plt.xticks(np.arange(len(available_molecules)),[mol.split("_")[0] for mol in available_molecules],fontsize="x-large")
        else:
            plt.xticks([])
        plt.grid(axis="y",zorder=1)
        plt.ylim(1e5,top_threshold)
        plt.yticks(fontsize="x-large")
        plt.text(0.5,5e7,mapping,fontsize="xx-large",horizontalalignment="center")

    plt.subplots_adjust(wspace=0.1,hspace=0.1)
    plt.savefig("generated_figures/supp_fig2_demo.png",bbox_inches="tight")
    plt.show()
