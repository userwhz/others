import numpy as np
import matplotlib.pyplot as plt
from shadowgrouping.measurement_schemes import N_delta
from shadowgrouping.molecules import available_molecules
from os import mkdir
from os.path import isdir, isfile
import json

##################
folder = "data/benchmarks/"
folder_chem_acc = "data/chem_acc/"
delta = 0.02

# plot specifics for bar plots
top_threshold = 1e8
methods  = ['ShadowBernstein', 'AdaptivePaulis', 'AEQuO-naive', 'OverlappedGroupingNew','RandomPaulis']
map_names = ["JW","BK","Parity"]
width = 0.8
align = "center"
width_single = width / (len(methods) + 1)
available_molecules = available_molecules[1:] # do not show data for H2 molecule comprised of 4 qubits
bar_pos = np.arange(len(available_molecules))[np.newaxis,:] + width_single*(np.arange(len(methods)+1)+-0.5*len(methods))[:,np.newaxis]
####################################################################
########### style params etc #######################################
####################################################################

chem_acc = 1.6e-3

threshold = int(np.ceil(N_delta(delta)))

shots_dict = {
    "H2_6-31g": int(2e6),
    "LiH": int(1e6),
    "BeH2": int(6e6),
    "H2O": int(3e7),
    "NH3": int(3e7)
}

def load_json(filename):
    with open(filename,"r") as f:
        data = json.load(f)
    return data

def get_cmap(n, name='hsv'):
    '''Returns a function that maps each index in 0, 1, ..., n-1 to a distinct 
    RGB color; the keyword argument name must be a standard mpl colormap name.'''
    return plt.cm.get_cmap(name, n)

def fit_func(N,A,coeff):
    return A*N**(-coeff)

def pretty_print_exponents(pre,exp):
    out = r"$"
    if pre is None:
        if exp is None:
            out += "1$"
        else:
            out += "10^{"+str(exp)+"}$"
        return out
    else:
        out +="{}".format(pre)
    if exp is None:
        out += "$"
    else:
        out += "\cdot 10^{"+str(exp)+"}$"
    return out

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
    # load fit params for all molecules
    data   = np.full((len(methods)+1,len(available_molecules)),1e5)
    data_c = np.zeros((len(methods)+1,len(available_molecules)))
    data_Delta   = np.zeros_like(data)
    data_Delta_c = np.zeros_like(data_c)

    keys = benchmark_dict["NH3"]["JW"]["provable"].keys()
    cmap = get_cmap(len(keys),"Accent")
    for i,method in enumerate(methods):
        for k,molecule in enumerate(available_molecules):
            fit_params  = load_json(folder+molecule+"_fit_params.json")
            temp = fit_params["JW"]["empirical"].get(method,None)
            if temp is not None:
                data[i,k], data_Delta[i,k]     = get_N_estimates(*temp[0],1.6e-3,*np.sqrt(np.diag(temp[1])))[:2]
                data_c[i,k], data_Delta_c[i,k] = temp[0][1], np.sqrt(temp[1][-1][-1])
            else:
                data[i,k], data_c[i,k], data_Delta[i,k], data_Delta_c[i,k] = 4*[-1]

    plt.figure(figsize=(12,6))
    plt.subplot(121)
    for d,deltaD,pos,method in zip(data,data_Delta,bar_pos,methods):
        filtered = np.bitwise_and(d >= 0, d < top_threshold)
        if np.sum(filtered) == 0:
            continue
        plt.bar(pos[filtered], d[filtered], yerr=deltaD[filtered], color = cmap(np.argmax([method==key for key in keys])), width = width_single,
                edgecolor ='grey', label =method, zorder=3, align=align, capsize=5)

    plt.yscale('log')
    plt.legend(fontsize="x-large")
    plt.ylabel('No. meas. rounds to reach chem. acc.', fontsize = "xx-large")
    plt.xticks(np.arange(len(available_molecules)),[mol.split("_")[0] for mol in available_molecules],fontsize="x-large")
    plt.grid(axis="y",zorder=1)
    plt.ylim(1e5,top_threshold)
    plt.yticks(fontsize="x-large")
    
    # get ShadowGrouping data for reaching chem. acc.
    errors, stds = [], []
    for molecule in available_molecules:
        for mapping in map_names:
            filename = folder_chem_acc+"{}_molecule_ShadowBernstein_method_{}_longest_results.txt".format(molecule,mapping)
            if not isfile(filename):
                errors.append(-1)
                stds.append(0)
                continue
            # load energies and times
            params = {}
            with open(filename,"r") as f:
                line = f.readline().strip()
                while line.find("=") != -1:
                    parts = line.split(" ")
                    try:
                        params[parts[0]] = int(parts[2])
                    except:
                        params[parts[0]] = float(parts[2])
                    line = f.readline().strip()
            _, estimates = np.loadtxt(filename,skiprows=len(params),unpack=True)
            estimates = estimates[:100]
            estimates = np.abs(estimates-params["E_GS"])
            errors.append(np.mean(estimates))
            stds.append(np.std(estimates))
    errors, stds = np.array(errors), np.array(stds)
    
    # get extrapolation data for comparison
    epsilon_extrapolation = np.zeros(len(available_molecules)*3)
    delta_extrapolation   = np.zeros_like(epsilon_extrapolation)

    for molecule_ind, molecule_name in enumerate(available_molecules):
        N = shots_dict[molecule_name]
        with open(folder+molecule_name+"_fit_params.json","r") as f:
            data = json.load(f)
        for i,map_name in enumerate(map_names):
            popt,pcov = data[map_name]["empirical"]["ShadowBernstein"]
            dA,dC = np.sqrt(np.diag(pcov))
            epsilon_extrapolation[3*molecule_ind+i] = fit_func(N,*popt) / chem_acc
            delta_extrapolation[3*molecule_ind+i]   = fit_func(N,dA,popt[-1]) / chem_acc + epsilon_extrapolation[3*molecule_ind+i]*np.log(N)*dC
    # group data from same molecule together
    offset = [0.5,0,-0.5]
    
    plt.subplot(122)
    for i,(val,err,off) in enumerate(zip(epsilon_extrapolation,delta_extrapolation,offset*5)):
        plt.fill_between([i+off-0.25,i+off+0.25],2*[val-err],2*[val+err],color="gray",alpha=0.4)#,marker="*",markersize=8)
    for i in range(0,15,3):
        plt.errorbar(np.arange(i,i+3)+np.array(offset),errors[i:i+3]/chem_acc,yerr=stds[i:i+3]/chem_acc/10,linestyle="none",elinewidth=5)
        plt.plot(np.arange(i,i+3)+np.array(offset),errors[i:i+3]/chem_acc,"k.")
        N = shots_dict[available_molecules[i//3]]
        exp = int(np.floor(np.log10(N)))
        pre = int(np.rint(N*10**(-exp)))
        plt.text(i,0.1,"("+pretty_print_exponents(pre,exp)+")",fontsize="x-large",verticalalignment="center",bbox=dict(facecolor='white', edgecolor="white", alpha=0.3))
    plt.hlines(1,-1,len(errors)+1,colors="black",linestyles="dashed")
    plt.ylim(0,1.1)
    plt.xlim(-1,len(errors))
    plt.grid()
    plt.xticks(np.arange(1,16,3),[mol.split("_")[0] for mol in available_molecules],fontsize="x-large")
    plt.xlabel("(mapping order: JW/BK/Parity)",fontsize="x-large")
    plt.gca().yaxis.tick_right()
    plt.gca().yaxis.set_label_position("right")
    plt.ylabel(r"$\vert \hat E - E \vert / \epsilon_\mathrm{acc.}^\mathrm{chem.}$",fontsize="xx-large")
    plt.yticks(fontsize="x-large")
    
    plt.subplots_adjust(wspace=0.1)
    plt.savefig("generated_figures/fig4_demo.png",bbox_inches="tight")
    plt.show()
