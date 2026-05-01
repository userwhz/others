# standard modules
import numpy as np
import pandas as pd
from IPython.display import Image, display
from os import mkdir
from os.path import isdir, isfile
# custom package

# load the measurement schemes from the Benchmark
from shadowgrouping.measurement_schemes import Shadow_Grouping, Derandomization, AdaptiveShadows, L1_sampler,  hit_by
from shadowgrouping.measurement_schemes import SettingSampler as Overlapped_Grouping
from shadowgrouping.AEQuO import AEQuO
from shadowgrouping.weight_functions import Inconfidence_bound, Bernstein_bound
# wrapper class to combine the measurement scheme with the respective outcomes
from shadowgrouping.energy_estimator import Energy_estimator, StateSampler, Sign_estimator
# helper functions to load Hamiltonian decompositions
from shadowgrouping.measurement_schemes import setting_to_str
from shadowgrouping.hamiltonian import get_pauli_list, get_groundstate, char_to_int, int_to_char, mappings, load_pauli_list
from shadowgrouping.benchmark import track_method_epsilon, save_to_json
folder_Hamiltonians = "Hamiltonians/"
folder_OGM_settings = "OGM_probabilities/OGM_{}_{}{}.txt" # format string to fill in {molecule}x{qubit_number}x{mapping}
savepath = "generated_data/"
savename = "_molecule_{}_{}" # insert {mapping_name,method}

# create temporary folder for storing outputs
if not isdir(savepath):
    mkdir(savepath)

molecule_name = "H2" # choose one out of the molecules above
mapping_name = "JW" # choose one out of ["JW","BK","Parity"]
basis_set = "sto3g" # choose one out of ["sto3g","6-31g"] - the latter only for H2 molecule
savename = molecule_name + savename

observables, w, offset, E_GS, state = load_pauli_list(folder_Hamiltonians,molecule_name,basis_set,mapping_name)
print("w", w)
print("=====================================")
# wrap ground state <state> into StateSampler in order to retrieve samples in arbitrary
state_sampler = StateSampler(state)
# fill in format string for Overlapped Grouping probabilities
folder_OGM_settings = folder_OGM_settings.format(molecule_name,observables.shape[-1],mapping_name.lower())

# hyperparameters for ShadowGrouping, see eq. (48) in manuscript
alpha = np.max(np.abs(w))/np.min(np.abs(w)) + np.min(np.abs(w))

eps = 0.1 # accuracy in Hartree -- irrelevant for the benchmark below
N_START = 100 # number of total measurement settings
N_STOP = 1000
N_runs = 50 # number of independent repetitions for the energy estimation
N_plot = 10 # number of data points tracked
delta = 0.02 # see caption of Figure 3 in manuscript
methods = {}
# methods["ShadowGrouping"]= Shadow_Grouping(observables,w,eps,Bernstein_bound(alpha=alpha)())

# methods["ShadowGrouping-truncated"] = Shadow_Grouping(observables,w,eps,Bernstein_bound(alpha=alpha)())
# methods["Derandomization"] = Derandomization(observables,w,np.sqrt(0.9),use_one_norm=True)
# methods["RandomPaulis"] = Derandomization(observables,w,eps,delta=1) # delta controls the randomness
# methods["AdaptivePaulis"] = AdaptiveShadows(observables,w)
# methods["AEQuO"] = AEQuO(observables,w,offset,adaptiveness_L=2,interval_skewness_l=4,budget=N_STOP)

# file = folder_OGM_settings.format(molecule_name,observables.shape[1],mapping_name.lower())
file = "OGM_probabilities/OGM_stabilizer_422.txt"
if isfile(file):
    methods["OverlappedGrouping"] = Overlapped_Grouping(observables,w,file)

# all details can be found in benchmark.py in track_method_epsilon() to generate the benchmark data
eps_dict = {}
benchmark_file = savepath+savename.format(mapping_name,"benchmark.txt")
print("benchamrk", benchmark_file)
# look whether a benchmark.txt file already exists and load the data
# afterwards, check whether all methods in <methods> have been benchmarked
# If not, benchmark them directly

if isfile(benchmark_file):
    with open(benchmark_file,"r") as f:
        columns = f.readline().strip().split()
    data = np.loadtxt(benchmark_file,skiprows=1).T
    for column,row in zip(columns,data):
        eps_dict[column] = row
        if column.find("-prov")>-1:
            print("Data for label <{}> loaded from file.".format(column[:-5]))

for label,method in methods.items():
    if eps_dict.get(label+"-emp",None) is None:
        print("Benchmarking method " + label,"...")
        params = {"Nshots": N_STOP, "Nsteps": N_plot, "Nreps": N_runs, "Nstart": N_START}
        if label.find("truncate") >= 0:
            params["truncate"] = True
            label = label[:label.find("-trun")]
            print(3)
        estimator = Energy_estimator(method,StateSampler(state),offset=offset)
        if params.get("truncate",False):
            # if the method truncates, track_method_epsilon() returns the truncated and the untruncated data
            N_steps, eps_SG, eps_SG_emp, eps_SG_std, E_emp, eps_trunc, eps_trunc_emp, eps_trunc_std, E_trunc = track_method_epsilon(estimator,E_GS,delta,params)
            filename = savename.format(mapping_name,label) + "-truncated_energies.txt"
            np.savetxt(savepath+filename,E_trunc,comments="",header=str(E_GS))
            eps_dict["Nsteps"] = N_steps
            eps_dict[label+"-truncated-emp"] = eps_trunc_emp
            eps_dict[label+"-truncated-STD"] = eps_trunc_std
            eps_dict[label+"-truncated-prov"] = eps_trunc
            print(len(estimator.settings_dict))
            print("Data for label <{}-truncated> generated.".format(label))
            print(1)
        else:
            # 入口
            N_steps, eps_SG, eps_SG_emp, eps_SG_std, E_emp = track_method_epsilon(estimator,E_GS,delta,params, label = label)
            print(2)
            eps_dict["Nsteps"] = N_steps

        eps_dict[label+"-emp"] = eps_SG_emp
        eps_dict[label+"-STD"] = eps_SG_std
        eps_dict[label+"-prov"] = eps_SG
        filename = savename.format(mapping_name,label) + "_energies.txt"
        np.savetxt(savepath+filename,E_emp,comments="",header=str(E_GS))
        print(label,estimator.settings_dict)
        print(len(label))
        print(len(estimator.settings_dict))
        print("Data for label <{}> generated.".format(label))

# saving all data into one file
df = pd.DataFrame.from_dict(eps_dict)
vals = df.to_numpy()
header = ""
for key in df.columns:
    header += "{}\t".format(key)
np.savetxt(savepath+savename.format(mapping_name,"benchmark.txt"),vals,header=header,comments="")

print("All methods' benchmark data generated / loaded.")
