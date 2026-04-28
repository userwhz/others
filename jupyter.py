# standard modules
import numpy as np
import pandas as pd
from os import mkdir
from os.path import isdir, isfile
from time import perf_counter
# custom package

# load the measurement schemes from the Benchmark
from shadowgrouping.measurement_schemes import Shadow_Grouping, Derandomization, AdaptiveShadows, L1_sampler,  hit_by, GraphColoringGrouping
from shadowgrouping.measurement_schemes import SettingSampler as Overlapped_Grouping
from shadowgrouping.AEQuO import AEQuO
from shadowgrouping.weight_functions import Inconfidence_bound, Bernstein_bound
# wrapper class to combine the measurement scheme with the respective outcomes
from shadowgrouping.energy_estimator import Energy_estimator, StateSampler, Sign_estimator
# helper functions to load Hamiltonian decompositions
from shadowgrouping.measurement_schemes import setting_to_str
from shadowgrouping.hamiltonian import get_pauli_list, get_groundstate, char_to_int, int_to_char, mappings, \
    load_pauli_list, load_pauli_list1, load_pauli_list2, load_pauli_list3, load_pauli_list4, load_pauli_list5
from shadowgrouping.benchmark import track_method_epsilon, save_to_json
from shadowgrouping.ogm_fc import optimize_ogm_fc_distribution, save_group_distribution

def f(IndexNum):

    folder_Hamiltonians = "haozhaowu/"
    folder_OGM_settings = "OGM_probabilities/OGM_{}_{}{}.txt" # format string to fill in {molecule}x{qubit_number}x{mapping}
    savepath = "setting/h2/"
    savename = "_molecule_{}_{}" # insert {mapping_name,method}

    # create temporary folder for storing outputs
    if not isdir(savepath):
        mkdir(savepath)

    molecule_name = "H2" # choose one out of the molecules above
    mapping_name = "JW" # choose one out of ["JW","BK","Parity"]
    basis_set = "sto3g" # choose one out of ["sto3g","6-31g"] - the latter only for H2 molecule
    savename = f"{IndexNum}_{basis_set}"

    observables, w, offset, E_GS, state = load_pauli_list(folder_Hamiltonians,molecule_name,basis_set,mapping_name, suiji = IndexNum)
    # print("w", w)
    print("len(w)", len(w))
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
    # methods["GC"] = GraphColoringGrouping(observables, w, eps, commutation_mode="fc")
    # methods["ShadowGrouping"]= Shadow_Grouping(observables,w,eps,Bernstein_bound(alpha=alpha)())
    #
    methods["ShadowGrouping"] = Shadow_Grouping(
        observables, w, eps, Bernstein_bound(alpha=alpha)(), commutation_mode="fc"
    )
    methods["Derandomization_fc"] = Derandomization(
        observables, w, np.sqrt(0.9), use_one_norm=True, commutation_mode="fc"
    )
    methods["RandomPaulis_fc"] = Derandomization(observables, w, eps, delta=1, commutation_mode="fc")
    #
    # # methods["ShadowGrouping-truncated"] = Shadow_Grouping(observables,w,eps,Bernstein_bound(alpha=alpha)())
    methods["Derandomization"] = Derandomization(observables, w, np.sqrt(0.9), use_one_norm=True)
    methods["RandomPaulis"] = Derandomization(observables, w, eps, delta=1) # delta controls the randomness
    # # methods["ShadowGrouping-truncated"] = Shadow_Grouping(observables,w,eps,Bernstein_bound(alpha=alpha)())
    # methods["Derandomization"] = Derandomization(observables,w,np.sqrt(0.9),use_one_norm=True)
    # methods["RandomPaulis"] = Derandomization(observables,w,eps,delta=1) # delta controls the randomness
    # # methods["AdaptivePaulis"] = AdaptiveShadows(observables,w)
    # methods["AEQuO"] = AEQuO(observables,w,offset,adaptiveness_L=2,interval_skewness_l=4,budget=N_STOP)

    # file = folder_OGM_settings.format(molecule_name,observables.shape[1],mapping_name.lower())

    file = "haozhaowu/H2/ogm_outputs/OGM_ogm_H2_"+ f'{IndexNum}' + ".txt"
    # if isfile(file):
        # methods["OverlappedGrouping"] = Overlapped_Grouping(observables,w,file)
    #     methods["OverlappedGrouping"] = Overlapped_Grouping(observables, w, file, commutation_mode="fc")
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
        print("label", label)
        if eps_dict.get(label+"-emp",None) is None:
            print("Benchmarking method " + label,"...")
            params = {"Nshots": N_STOP, "Nsteps": N_plot, "Nreps": N_runs, "Nstart": N_START}
            if label.find("truncate") >= 0:
                params["truncate"] = True
                label = label[:label.find("-trun")]
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
            else:
                # 入口
                N_steps, eps_SG, eps_SG_emp, eps_SG_std, E_emp = track_method_epsilon(estimator,E_GS,delta,params,label = label)
                eps_dict["Nsteps"] = N_steps
                file_path = "H2_output.txt"

                # 构造要写入的字符串
                # 注意：print会自动换行，但写入文件需要手动在末尾加 '\n'
                content = f" {IndexNum} {len(estimator.settings_dict)}\n"

                # 使用 'a' 模式打开文件，表示追加 (append)
                # encoding='utf-8' 防止中文乱码
                with open(file_path, "a", encoding="utf-8") as f:
                    f.write(content)

            eps_dict[label+"-emp"] = eps_SG_emp
            eps_dict[label+"-STD"] = eps_SG_std
            eps_dict[label+"-prov"] = eps_SG
            filename = f"{savename}_{label}_energies.txt"

            np.savetxt(savepath+filename,E_emp,comments="",header=str(E_GS))
            # with open("settings.txt", "a", encoding="utf-8") as f:  # 'a' 模式追加内容
            #     print("setting number", len(estimator.settings_dict), file=f)
            # with open("settings.txt", "a", encoding="utf-8") as f:  # 'a' 模式追加内容
            #     print("shot number", 262144, file=f)
            # with open("settings.txt", "a", encoding="utf-8") as f:  # 'a' 模式追加内容
            #     print(label,estimator.settings_dict, file=f)

            # print(len(label))
            print("Data for label <{}> generated.".format(label))

    # saving all data into one file
    df = pd.DataFrame.from_dict(eps_dict)
    vals = df.to_numpy()
    header = ""
    for key in df.columns:
        header += "{}\t".format(key)
    # np.savetxt(savepath+savename.format(mapping_name,"benchmark.txt"),vals,header=header,comments="")

    print("All methods' benchmark data generated / loaded.")

def liH(IndexNum):

    folder_Hamiltonians = "haozhaowu/"
    folder_OGM_settings = "OGM_probabilities/OGM_{}_{}{}.txt" # format string to fill in {molecule}x{qubit_number}x{mapping}
    savepath = "setting/test/"
    savename = "_molecule_{}_{}" # insert {mapping_name,method}

    # create temporary folder for storing outputs
    if not isdir(savepath):
        mkdir(savepath)

    molecule_name = "H2" # choose one out of the molecules above
    mapping_name = "JW" # choose one out of ["JW","BK","Parity"]
    basis_set = "LiH" # choose one out of ["sto3g","6-31g"] - the latter only for H2 molecule
    savename = f"{IndexNum}_{basis_set}"

    observables, w, offset, E_GS, state = load_pauli_list3(folder_Hamiltonians,molecule_name,basis_set,mapping_name, suiji = IndexNum)
    # print("w", w)
    print("len(w)", len(w))
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
    # methods["GC"] = GraphColoringGrouping(observables, w, eps, commutation_mode="fc")
    # methods["ShadowGrouping"]= Shadow_Grouping(observables,w,eps,Bernstein_bound(alpha=alpha)())
    # methods["ShadowGrouping"] = Shadow_Grouping(
    #     observables, w, eps, Bernstein_bound(alpha=alpha)(), commutation_mode="fc"
    # )
    methods["RandomPaulis_fc"] = Derandomization(observables,w,eps,delta=1, commutation_mode="fc")
    #
    # methods["ShadowGrouping-truncated"] = Shadow_Grouping(observables,w,eps,Bernstein_bound(alpha=alpha)())
    # methods["Derandomization"] = Derandomization(observables,w,np.sqrt(0.9),use_one_norm=True)
    methods["RandomPaulis"] = Derandomization(observables,w,eps,delta=1) # delta controls the randomness
    # # methods["ShadowGrouping-truncated"] = Shadow_Grouping(observables,w,eps,Bernstein_bound(alpha=alpha)())
    # methods["Derandomization"] = Derandomization(observables,w,np.sqrt(0.9),use_one_norm=True)
    # methods["RandomPaulis"] = Derandomization(observables,w,eps,delta=1) # delta controls the randomness
    # # methods["AdaptivePaulis"] = AdaptiveShadows(observables,w)
    # methods["AEQuO"] = AEQuO(observables,w,offset,adaptiveness_L=2,interval_skewness_l=4,budget=N_STOP)

    # file = folder_OGM_settings.format(molecule_name,observables.shape[1],mapping_name.lower())

    file = "haozhaowu/LiH/hamil_class/ogm_outputs/OGM_ogm_LiH_"+ f'{IndexNum}' + ".txt"
    if isfile(file):
        methods["OverlappedGrouping"] = Overlapped_Grouping(observables, w, file, commutation_mode="fc")

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
        print("label", label)
        if eps_dict.get(label+"-emp",None) is None:
            print("Benchmarking method " + label,"...")
            params = {"Nshots": N_STOP, "Nsteps": N_plot, "Nreps": N_runs, "Nstart": N_START}
            if label.find("truncate") >= 0:
                params["truncate"] = True
                label = label[:label.find("-trun")]
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
            else:
                # 入口
                N_steps, eps_SG, eps_SG_emp, eps_SG_std, E_emp = track_method_epsilon(estimator,E_GS,delta,params,label = label)
                eps_dict["Nsteps"] = N_steps
                file_path = "LiH_output.txt"

                # 构造要写入的字符串
                # 注意：print会自动换行，但写入文件需要手动在末尾加 '\n'
                content = f" {IndexNum} {len(estimator.settings_dict)}\n"

                # 使用 'a' 模式打开文件，表示追加 (append)
                # encoding='utf-8' 防止中文乱码
                with open(file_path, "a", encoding="utf-8") as f:
                    f.write(content)


            eps_dict[label+"-emp"] = eps_SG_emp
            eps_dict[label+"-STD"] = eps_SG_std
            eps_dict[label+"-prov"] = eps_SG
            filename = f"{savename}_{label}_energies.txt"

            np.savetxt(savepath+filename,E_emp,comments="",header=str(E_GS))
            # with open("settings.txt", "a", encoding="utf-8") as f:  # 'a' 模式追加内容
            #     print("setting number", len(estimator.settings_dict), file=f)
            # with open("settings.txt", "a", encoding="utf-8") as f:  # 'a' 模式追加内容
            #     print("shot number", 262144, file=f)
            # with open("settings.txt", "a", encoding="utf-8") as f:  # 'a' 模式追加内容
            #     print(label,estimator.settings_dict, file=f)

            # print(len(label))
            print("Data for label <{}> generated.".format(label))

    # saving all data into one file
    df = pd.DataFrame.from_dict(eps_dict)
    vals = df.to_numpy()
    header = ""
    for key in df.columns:
        header += "{}\t".format(key)
    # np.savetxt(savepath+savename.format(mapping_name,"benchmark.txt"),vals,header=header,comments="")

    print("All methods' benchmark data generated / loaded.")


def random1(type, IndexNum):

    folder_Hamiltonians = "haozhaowu/random/"
    folder_OGM_settings = "OGM_probabilities/OGM_{}_{}{}.txt" # format string to fill in {molecule}x{qubit_number}x{mapping}
    savepath = f"setting/{type}/"
    savename = "_molecule_{}_{}" # insert {mapping_name,method}

    # create temporary folder for storing outputs
    if not isdir(savepath):
        mkdir(savepath)

    molecule_name = "H2" # choose one out of the molecules above
    mapping_name = "JW" # choose one out of ["JW","BK","Parity"]
    basis_set = "sto3g" # choose one out of ["sto3g","6-31g"] - the latter only for H2 molecule
    savename = f"{IndexNum}_random"

    observables, w, offset, E_GS, state = load_pauli_list1(folder_Hamiltonians,molecule_name,basis_set,mapping_name, suiji = IndexNum,type = type)
    # print("w", w)
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
    # methods["GC"] = GraphColoringGrouping(observables, w, eps, commutation_mode="fc")
    # methods["ShadowGrouping"]= Shadow_Grouping(observables,w,eps,Bernstein_bound(alpha=alpha)())
    methods["ShadowGrouping"] = Shadow_Grouping(
        observables, w, eps, Bernstein_bound(alpha=alpha)(), commutation_mode="fc"
    )
    methods["Derandomization"] = Derandomization(observables,w,np.sqrt(0.9),use_one_norm=True)
    methods["RandomPaulis"] = Derandomization(observables,w,eps,delta=1) # delta controls the randomness
    # methods["AdaptivePaulis"] = AdaptiveShadows(observables,w)
    # methods["AEQuO"] = AEQuO(observables,w,offset,adaptiveness_L=2,interval_skewness_l=4,budget=N_STOP)
    # methods["Derandomization_fc"] = Derandomization(
    #     observables, w, np.sqrt(0.9), use_one_norm=True, commutation_mode="fc"
    # )
    # methods["RandomPaulis_fc"] = Derandomization(observables, w, eps, delta=1, commutation_mode="fc")

    # file = folder_OGM_settings.format(molecule_name,observables.shape[1],mapping_name.lower())

    file = "haozhaowu/random/hamil_class/ogm_outputs/OGM_ogm_hamiltonian_" + f'{type}' + "_"  + f'{IndexNum}' + ".txt"
    print("file", file)
    if isfile(file):
        # methods["OverlappedGrouping"] = Overlapped_Grouping(observables,w,file)
        methods["OverlappedGrouping"] = Overlapped_Grouping(observables, w, file, commutation_mode="fc")
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
        print("label", label)
        if eps_dict.get(label+"-emp",None) is None:
            print("Benchmarking method " + label,"...")
            params = {"Nshots": N_STOP, "Nsteps": N_plot, "Nreps": N_runs, "Nstart": N_START}
            if label.find("truncate") >= 0:
                params["truncate"] = True
                label = label[:label.find("-trun")]
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
                print("method", method, "setting", len(estimator.settings_dict))
            else:
                # 入口
                N_steps, eps_SG, eps_SG_emp, eps_SG_std, E_emp = track_method_epsilon(estimator,E_GS,delta,params,label = label)
                eps_dict["Nsteps"] = N_steps
                print("type",type, "num", IndexNum, " method",method, "setting", len(estimator.settings_dict))
                file_path = "sparse_output.txt"

                # 构造要写入的字符串
                # 注意：print会自动换行，但写入文件需要手动在末尾加 '\n'
                content = f" {len(estimator.settings_dict)}\n"

                # 使用 'a' 模式打开文件，表示追加 (append)
                # encoding='utf-8' 防止中文乱码
                with open(file_path, "a", encoding="utf-8") as f:
                    f.write(content)

            eps_dict[label+"-emp"] = eps_SG_emp
            eps_dict[label+"-STD"] = eps_SG_std
            eps_dict[label+"-prov"] = eps_SG
            filename = f"{savename}_{label}_energies.txt"

            np.savetxt(savepath+filename,E_emp,comments="",header=str(E_GS))
            # with open("settings.txt", "a", encoding="utf-8") as f:  # 'a' 模式追加内容
            #     print("setting number", len(estimator.settings_dict), file=f)
            # with open("settings.txt", "a", encoding="utf-8") as f:  # 'a' 模式追加内容
            #     print("shot number", 262144, file=f)
            # with open("settings.txt", "a", encoding="utf-8") as f:  # 'a' 模式追加内容
            #     print(label,estimator.settings_dict, file=f)

            # print(len(label))
            print("Data for label <{}> generated.".format(label))

    # saving all data into one file
    df = pd.DataFrame.from_dict(eps_dict)
    vals = df.to_numpy()
    header = ""
    for key in df.columns:
        header += "{}\t".format(key)
    # np.savetxt(savepath+savename.format(mapping_name,"benchmark.txt"),vals,header=header,comments="")

    print("All methods' benchmark data generated / loaded.")


def heisenberg(IndexNum):

    folder_Hamiltonians = "haozhaowu/"
    folder_OGM_settings = "OGM_probabilities/OGM_{}_{}{}.txt"  # format string to fill in {molecule}x{qubit_number}x{mapping}
    savepath = "setting/heisenberg/"
    savename = "_molecule_{}_{}"  # insert {mapping_name,method}

    # create temporary folder for storing outputs
    if not isdir(savepath):
        mkdir(savepath)

    molecule_name = "H2"  # choose one out of the molecules above
    mapping_name = "JW"  # choose one out of ["JW","BK","Parity"]
    basis_set = "sto3g"  # choose one out of ["sto3g","6-31g"] - the latter only for H2 molecule
    savename = f"{IndexNum}_heisenberg"

    observables, w, offset, E_GS, state = load_pauli_list2(folder_Hamiltonians, molecule_name, basis_set,
                                                          mapping_name, suiji=IndexNum)
    # print("w", w)
    print("len(w)", len(w))
    print("=====================================")
    # wrap ground state <state> into StateSampler in order to retrieve samples in arbitrary
    state_sampler = StateSampler(state)
    # fill in format string for Overlapped Grouping probabilities
    folder_OGM_settings = folder_OGM_settings.format(molecule_name, observables.shape[-1], mapping_name.lower())

    # hyperparameters for ShadowGrouping, see eq. (48) in manuscript
    alpha = np.max(np.abs(w)) / np.min(np.abs(w)) + np.min(np.abs(w))

    eps = 0.1  # accuracy in Hartree -- irrelevant for the benchmark below
    N_START = 100  # number of total measurement settings
    N_STOP = 1000
    N_runs = 50  # number of independent repetitions for the energy estimation
    N_plot = 10  # number of data points tracked
    delta = 0.02  # see caption of Figure 3 in manuscript
    methods = {}
    methods["GC"] = GraphColoringGrouping(observables, w, eps, commutation_mode="fc")
    methods["ShadowGrouping"] = Shadow_Grouping(observables, w, eps, Bernstein_bound(alpha=alpha)())
    methods["ShadowGrouping_fc"] = Shadow_Grouping(
        observables, w, eps, Bernstein_bound(alpha=alpha)(), commutation_mode="fc"
    )
    #
    # # methods["ShadowGrouping-truncated"] = Shadow_Grouping(observables,w,eps,Bernstein_bound(alpha=alpha)())
    methods["Derandomization"] = Derandomization(observables, w, np.sqrt(0.9), use_one_norm=True)
    methods["Derandomization_fc"] = Derandomization(
        observables, w, np.sqrt(0.9), use_one_norm=True, commutation_mode="fc"
    )
    methods["RandomPaulis"] = Derandomization(observables, w, eps, delta=1) # delta controls the randomness
    methods["RandomPaulis_fc"] = Derandomization(observables, w, eps, delta=1, commutation_mode="fc")
    # # methods["AdaptivePaulis"] = AdaptiveShadows(observables,w)
    # methods["AEQuO"] = AEQuO(observables,w,offset,adaptiveness_L=2,interval_skewness_l=4,budget=N_STOP)

    # file = folder_OGM_settings.format(molecule_name,observables.shape[1],mapping_name.lower())

    file = "haozhaowu/heisenberg/hamil_class/ogm_outputs/OGM_ogm_hamiltonian_heisenberg_" + f'{IndexNum}' + ".txt"
    if isfile(file):
        methods["OverlappedGrouping"] = Overlapped_Grouping(observables, w, file, commutation_mode="fc")

    # all details can be found in benchmark.py in track_method_epsilon() to generate the benchmark data
    eps_dict = {}
    benchmark_file = savepath + savename.format(mapping_name, "benchmark.txt")
    print("benchamrk", benchmark_file)
    # look whether a benchmark.txt file already exists and load the data
    # afterwards, check whether all methods in <methods> have been benchmarked
    # If not, benchmark them directly

    if isfile(benchmark_file):
        with open(benchmark_file, "r") as f:
            columns = f.readline().strip().split()
        data = np.loadtxt(benchmark_file, skiprows=1).T
        for column, row in zip(columns, data):
            eps_dict[column] = row
            if column.find("-prov") > -1:
                print("Data for label <{}> loaded from file.".format(column[:-5]))

    for label, method in methods.items():
        print("label", label)
        if eps_dict.get(label + "-emp", None) is None:
            print("Benchmarking method " + label, "...")
            params = {"Nshots": N_STOP, "Nsteps": N_plot, "Nreps": N_runs, "Nstart": N_START}
            if label.find("truncate") >= 0:
                params["truncate"] = True
                label = label[:label.find("-trun")]
            estimator = Energy_estimator(method, StateSampler(state), offset=offset)
            if params.get("truncate", False):
                # if the method truncates, track_method_epsilon() returns the truncated and the untruncated data
                N_steps, eps_SG, eps_SG_emp, eps_SG_std, E_emp, eps_trunc, eps_trunc_emp, eps_trunc_std, E_trunc = track_method_epsilon(
                    estimator, E_GS, delta, params)
                filename = savename.format(mapping_name, label) + "-truncated_energies.txt"
                np.savetxt(savepath + filename, E_trunc, comments="", header=str(E_GS))
                eps_dict["Nsteps"] = N_steps
                eps_dict[label + "-truncated-emp"] = eps_trunc_emp
                eps_dict[label + "-truncated-STD"] = eps_trunc_std
                eps_dict[label + "-truncated-prov"] = eps_trunc
                print(len(estimator.settings_dict))
                print("Data for label <{}-truncated> generated.".format(label))
            else:
                # 入口
                N_steps, eps_SG, eps_SG_emp, eps_SG_std, E_emp = track_method_epsilon(estimator, E_GS, delta,
                                                                                      params, label=label)
                eps_dict["Nsteps"] = N_steps

            eps_dict[label + "-emp"] = eps_SG_emp
            eps_dict[label + "-STD"] = eps_SG_std
            eps_dict[label + "-prov"] = eps_SG
            filename = f"{savename}_{label}_energies.txt"

            np.savetxt(savepath + filename, E_emp, comments="", header=str(E_GS))
            # with open("settings.txt", "a", encoding="utf-8") as f:  # 'a' 模式追加内容
            #     print("setting number", len(estimator.settings_dict), file=f)
            # with open("settings.txt", "a", encoding="utf-8") as f:  # 'a' 模式追加内容
            #     print("shot number", 262144, file=f)
            # with open("settings.txt", "a", encoding="utf-8") as f:  # 'a' 模式追加内容
            #     print(label,estimator.settings_dict, file=f)

            # print(len(label))
            print("Data for label <{}> generated.".format(label))

    # saving all data into one file
    df = pd.DataFrame.from_dict(eps_dict)
    vals = df.to_numpy()
    header = ""
    for key in df.columns:
        header += "{}\t".format(key)
    # np.savetxt(savepath+savename.format(mapping_name,"benchmark.txt"),vals,header=header,comments="")

    print("All methods' benchmark data generated / loaded.")

def klocal():
    folder_Hamiltonians = "haozhaowu/"
    folder_OGM_settings = "OGM_probabilities/OGM_{}_{}{}.txt"  # format string to fill in {molecule}x{qubit_number}x{mapping}
    savepath = f"setting/klocal/"
    savename = "_molecule_{}_{}"  # insert {mapping_name,method}

    # create temporary folder for storing outputs
    if not isdir(savepath):
        mkdir(savepath)

    molecule_name = "H2"  # choose one out of the molecules above
    mapping_name = "JW"  # choose one out of ["JW","BK","Parity"]
    basis_set = "sto3g"  # choose one out of ["sto3g","6-31g"] - the latter only for H2 molecule
    savename = f"klocal"

    observables, w, offset, E_GS, state = load_pauli_list4(folder_Hamiltonians, molecule_name, basis_set,
                                                           mapping_name)
    # print("w", w)
    print("=====================================")
    # wrap ground state <state> into StateSampler in order to retrieve samples in arbitrary
    state_sampler = StateSampler(state)
    # fill in format string for Overlapped Grouping probabilities
    folder_OGM_settings = folder_OGM_settings.format(molecule_name, observables.shape[-1], mapping_name.lower())

    # hyperparameters for ShadowGrouping, see eq. (48) in manuscript
    alpha = np.max(np.abs(w)) / np.min(np.abs(w)) + np.min(np.abs(w))

    eps = 0.1  # accuracy in Hartree -- irrelevant for the benchmark below
    N_START = 100  # number of total measurement settings
    N_STOP = 1000
    N_runs = 50  # number of independent repetitions for the energy estimation
    N_plot = 10  # number of data points tracked
    delta = 0.02  # see caption of Figure 3 in manuscript
    methods = {}
    # methods["ShadowGrouping"]= Shadow_Grouping(observables,w,eps,Bernstein_bound(alpha=alpha)())
    methods["ShadowGrouping"] = Shadow_Grouping(
        observables, w, eps, Bernstein_bound(alpha=alpha)(), commutation_mode="fc"
    )
    methods["Derandomization"] = Derandomization(observables, w, np.sqrt(0.9), use_one_norm=True)
    methods["RandomPaulis"] = Derandomization(observables, w, eps, delta=1)  # delta controls the randomness
    # methods["AdaptivePaulis"] = AdaptiveShadows(observables,w)
    # methods["AEQuO"] = AEQuO(observables,w,offset,adaptiveness_L=2,interval_skewness_l=4,budget=N_STOP)


    # OGM (FC): build overlapped commuting groups + optimize group probabilities in Python,
    # then load as SettingSampler(commutation_mode="fc") which triggers joint-measurement in Energy_estimator.
    ogm_fc_file = savepath + "OGM_groups_klocal.txt"
    dist = optimize_ogm_fc_distribution(observables, w, T=100000)
    save_group_distribution(ogm_fc_file, dist)
    methods["OverlappedGrouping_fc"] = Overlapped_Grouping(observables, w, ogm_fc_file, commutation_mode="fc")
    # all details can be found in benchmark.py in track_method_epsilon() to generate the benchmark data
    eps_dict = {}
    benchmark_file = savepath + savename.format(mapping_name, "benchmark.txt")
    print("benchamrk", benchmark_file)

    if isfile(benchmark_file):
        with open(benchmark_file, "r") as f:
            columns = f.readline().strip().split()
        data = np.loadtxt(benchmark_file, skiprows=1).T
        for column, row in zip(columns, data):
            eps_dict[column] = row
            if column.find("-prov") > -1:
                print("Data for label <{}> loaded from file.".format(column[:-5]))

    for label, method in methods.items():
        print("label", label)
        if eps_dict.get(label + "-emp", None) is None:
            print("Benchmarking method " + label, "...")
            params = {"Nshots": N_STOP, "Nsteps": N_plot, "Nreps": N_runs, "Nstart": N_START}
            if label.find("truncate") >= 0:
                params["truncate"] = True
                label = label[:label.find("-trun")]
            estimator = Energy_estimator(method, StateSampler(state), offset=offset)
            if params.get("truncate", False):
                # if the method truncates, track_method_epsilon() returns the truncated and the untruncated data
                N_steps, eps_SG, eps_SG_emp, eps_SG_std, E_emp, eps_trunc, eps_trunc_emp, eps_trunc_std, E_trunc = track_method_epsilon(
                    estimator, E_GS, delta, params)
                filename = savename.format(mapping_name, label) + "-truncated_energies.txt"
                np.savetxt(savepath + filename, E_trunc, comments="", header=str(E_GS))
                eps_dict["Nsteps"] = N_steps
                eps_dict[label + "-truncated-emp"] = eps_trunc_emp
                eps_dict[label + "-truncated-STD"] = eps_trunc_std
                eps_dict[label + "-truncated-prov"] = eps_trunc
                print("method", method, "setting", len(estimator.settings_dict))
            else:
                # 入口
                N_steps, eps_SG, eps_SG_emp, eps_SG_std, E_emp = track_method_epsilon(estimator, E_GS, delta,
                                                                                      params, label=label)
                eps_dict["Nsteps"] = N_steps

                # 构造要写入的字符串
                content = f" {len(estimator.settings_dict)}\n"


            eps_dict[label + "-emp"] = eps_SG_emp
            eps_dict[label + "-STD"] = eps_SG_std
            eps_dict[label + "-prov"] = eps_SG
            filename = f"{savename}_{label}_energies.txt"

            np.savetxt(savepath + filename, E_emp, comments="", header=str(E_GS))
            print("Data for label <{}> generated.".format(label))

    # saving all data into one file
    df = pd.DataFrame.from_dict(eps_dict)
    vals = df.to_numpy()
    header = ""
    for key in df.columns:
        header += "{}\t".format(key)

    print("All methods' benchmark data generated / loaded.")


def H2O():
    folder_Hamiltonians = "haozhaowu/"
    folder_OGM_settings = "OGM_probabilities/OGM_{}_{}{}.txt"  # format string to fill in {molecule}x{qubit_number}x{mapping}
    savepath = f"setting/H2O/"
    savename = "_molecule_{}_{}"  # insert {mapping_name,method}

    # create temporary folder for storing outputs
    if not isdir(savepath):
        mkdir(savepath)

    molecule_name = "H2O"  # choose one out of the molecules above
    mapping_name = "JW"  # choose one out of ["JW","BK","Parity"]
    basis_set = "sto3g"  # choose one out of ["sto3g","6-31g"] - the latter only for H2 molecule
    savename = f"H2O"

    observables, w, offset, E_GS, state = load_pauli_list5(folder_Hamiltonians, molecule_name, basis_set,
                                                           mapping_name)
    # print("w", w)
    print("=====================================")
    # wrap ground state <state> into StateSampler in order to retrieve samples in arbitrary
    state_sampler = StateSampler(state)
    # fill in format string for Overlapped Grouping probabilities
    folder_OGM_settings = folder_OGM_settings.format(molecule_name, observables.shape[-1], mapping_name.lower())

    # hyperparameters for ShadowGrouping, see eq. (48) in manuscript
    alpha = np.max(np.abs(w)) / np.min(np.abs(w)) + np.min(np.abs(w))
    # FC joint-measurement uses explicit 2^k x 2^k unitaries; k must be >= max term locality.
    max_locality = int(np.max(np.sum(observables != 0, axis=1)))
    max_support_qubits = max(8, max_locality)

    eps = 0.1  # accuracy in Hartree -- irrelevant for the benchmark below
    N_START = 100  # number of total measurement settings
    N_STOP = 1000
    N_runs = 50  # number of independent repetitions for the energy estimation
    N_plot = 10  # number of data points tracked
    delta = 0.02  # see caption of Figure 3 in manuscript
    methods = {}
    # methods["ShadowGrouping"]= Shadow_Grouping(observables,w,eps,Bernstein_bound(alpha=alpha)())
    methods["ShadowGrouping"] = Shadow_Grouping(
        observables,
        w,
        eps,
        Bernstein_bound(alpha=alpha)(),
        commutation_mode="fc",
        max_support_qubits=max_support_qubits,
    )
    methods["Derandomization"] = Derandomization(observables, w, np.sqrt(0.9), use_one_norm=True)
    methods["RandomPaulis"] = Derandomization(observables, w, eps, delta=1)  # delta controls the randomness
    # methods["AdaptivePaulis"] = AdaptiveShadows(observables,w)
    # methods["AEQuO"] = AEQuO(observables,w,offset,adaptiveness_L=2,interval_skewness_l=4,budget=N_STOP)


    # OGM (FC): build overlapped commuting groups + optimize group probabilities in Python,
    # then load as SettingSampler(commutation_mode="fc") which triggers joint-measurement in Energy_estimator.
    ogm_fc_file = savepath + "OGM_groups_H2O.txt"
    dist = optimize_ogm_fc_distribution(observables, w, T=100000, max_support_qubits=max_support_qubits)
    save_group_distribution(ogm_fc_file, dist)
    methods["OverlappedGrouping_fc"] = Overlapped_Grouping(
        observables,
        w,
        ogm_fc_file,
        commutation_mode="fc",
        max_support_qubits=max_support_qubits,
    )
    # all details can be found in benchmark.py in track_method_epsilon() to generate the benchmark data
    eps_dict = {}
    benchmark_file = savepath + savename.format(mapping_name, "benchmark.txt")
    print("benchamrk", benchmark_file)

    if isfile(benchmark_file):
        with open(benchmark_file, "r") as f:
            columns = f.readline().strip().split()
        data = np.loadtxt(benchmark_file, skiprows=1).T
        for column, row in zip(columns, data):
            eps_dict[column] = row
            if column.find("-prov") > -1:
                print("Data for label <{}> loaded from file.".format(column[:-5]))

    for label, method in methods.items():
        print("label", label)
        if eps_dict.get(label + "-emp", None) is None:
            print("Benchmarking method " + label, "...")
            params = {"Nshots": N_STOP, "Nsteps": N_plot, "Nreps": N_runs, "Nstart": N_START}
            if label.find("truncate") >= 0:
                params["truncate"] = True
                label = label[:label.find("-trun")]
            estimator = Energy_estimator(method, StateSampler(state), offset=offset)
            if params.get("truncate", False):
                # if the method truncates, track_method_epsilon() returns the truncated and the untruncated data
                N_steps, eps_SG, eps_SG_emp, eps_SG_std, E_emp, eps_trunc, eps_trunc_emp, eps_trunc_std, E_trunc = track_method_epsilon(
                    estimator, E_GS, delta, params)
                filename = savename.format(mapping_name, label) + "-truncated_energies.txt"
                np.savetxt(savepath + filename, E_trunc, comments="", header=str(E_GS))
                eps_dict["Nsteps"] = N_steps
                eps_dict[label + "-truncated-emp"] = eps_trunc_emp
                eps_dict[label + "-truncated-STD"] = eps_trunc_std
                eps_dict[label + "-truncated-prov"] = eps_trunc
                print("method", method, "setting", len(estimator.settings_dict))
            else:
                # 入口
                N_steps, eps_SG, eps_SG_emp, eps_SG_std, E_emp = track_method_epsilon(estimator, E_GS, delta,
                                                                                      params, label=label)
                eps_dict["Nsteps"] = N_steps

                # 构造要写入的字符串
                content = f" {len(estimator.settings_dict)}\n"


            eps_dict[label + "-emp"] = eps_SG_emp
            eps_dict[label + "-STD"] = eps_SG_std
            eps_dict[label + "-prov"] = eps_SG
            filename = f"{savename}_{label}_energies.txt"

            np.savetxt(savepath + filename, E_emp, comments="", header=str(E_GS))
            print("Data for label <{}> generated.".format(label))

    # saving all data into one file
    df = pd.DataFrame.from_dict(eps_dict)
    vals = df.to_numpy()
    header = ""
    for key in df.columns:
        header += "{}\t".format(key)

    print("All methods' benchmark data generated / loaded.")



def BeH2():
    folder_Hamiltonians = "haozhaowu/"
    folder_OGM_settings = "OGM_probabilities/OGM_{}_{}{}.txt"  # format string to fill in {molecule}x{qubit_number}x{mapping}
    savepath = f"setting/BeH2/"
    savename = "_molecule_{}_{}"  # insert {mapping_name,method}

    # create temporary folder for storing outputs
    if not isdir(savepath):
        mkdir(savepath)

    molecule_name = "BeH2"  # choose one out of the molecules above
    mapping_name = "JW"  # choose one out of ["JW","BK","Parity"]
    basis_set = "sto3g"  # choose one out of ["sto3g","6-31g"] - the latter only for H2 molecule
    savename = f"BeH2"

    observables, w, offset, E_GS, state = load_pauli_list5(folder_Hamiltonians, molecule_name, basis_set,
                                                           mapping_name)
    # print("w", w)
    print("=====================================")
    # wrap ground state <state> into StateSampler in order to retrieve samples in arbitrary
    state_sampler = StateSampler(state)
    # fill in format string for Overlapped Grouping probabilities
    folder_OGM_settings = folder_OGM_settings.format(molecule_name, observables.shape[-1], mapping_name.lower())

    # hyperparameters for ShadowGrouping, see eq. (48) in manuscript
    alpha = np.max(np.abs(w)) / np.min(np.abs(w)) + np.min(np.abs(w))
    # FC joint-measurement uses explicit 2^k x 2^k unitaries; k must be >= max term locality.
    max_locality = int(np.max(np.sum(observables != 0, axis=1)))
    max_support_qubits = max(8, max_locality)

    eps = 0.1  # accuracy in Hartree -- irrelevant for the benchmark below
    N_START = 100  # number of total measurement settings
    N_STOP = 1000
    N_runs = 50  # number of independent repetitions for the energy estimation
    N_plot = 10  # number of data points tracked
    delta = 0.02  # see caption of Figure 3 in manuscript
    methods = {}
    # methods["ShadowGrouping"]= Shadow_Grouping(observables,w,eps,Bernstein_bound(alpha=alpha)())
    methods["ShadowGrouping"] = Shadow_Grouping(
        observables,
        w,
        eps,
        Bernstein_bound(alpha=alpha)(),
        commutation_mode="fc",
        max_support_qubits=max_support_qubits,
    )
    methods["Derandomization"] = Derandomization(observables, w, np.sqrt(0.9), use_one_norm=True)
    methods["RandomPaulis"] = Derandomization(observables, w, eps, delta=1)  # delta controls the randomness
    # methods["AdaptivePaulis"] = AdaptiveShadows(observables,w)
    # methods["AEQuO"] = AEQuO(observables,w,offset,adaptiveness_L=2,interval_skewness_l=4,budget=N_STOP)


    # OGM (FC): build overlapped commuting groups + optimize group probabilities in Python,
    # then load as SettingSampler(commutation_mode="fc") which triggers joint-measurement in Energy_estimator.
    ogm_fc_file = savepath + "OGM_groups_BeH2.txt"
    dist = optimize_ogm_fc_distribution(observables, w, T=100000, max_support_qubits=max_support_qubits)
    save_group_distribution(ogm_fc_file, dist)
    methods["OverlappedGrouping_fc"] = Overlapped_Grouping(
        observables,
        w,
        ogm_fc_file,
        commutation_mode="fc",
        max_support_qubits=max_support_qubits,
    )
    # all details can be found in benchmark.py in track_method_epsilon() to generate the benchmark data
    eps_dict = {}
    benchmark_file = savepath + savename.format(mapping_name, "benchmark.txt")
    print("benchamrk", benchmark_file)

    if isfile(benchmark_file):
        with open(benchmark_file, "r") as f:
            columns = f.readline().strip().split()
        data = np.loadtxt(benchmark_file, skiprows=1).T
        for column, row in zip(columns, data):
            eps_dict[column] = row
            if column.find("-prov") > -1:
                print("Data for label <{}> loaded from file.".format(column[:-5]))

    for label, method in methods.items():
        print("label", label)
        if eps_dict.get(label + "-emp", None) is None:
            print("Benchmarking method " + label, "...")
            params = {"Nshots": N_STOP, "Nsteps": N_plot, "Nreps": N_runs, "Nstart": N_START}
            if label.find("truncate") >= 0:
                params["truncate"] = True
                label = label[:label.find("-trun")]
            estimator = Energy_estimator(method, StateSampler(state), offset=offset)
            if params.get("truncate", False):
                # if the method truncates, track_method_epsilon() returns the truncated and the untruncated data
                N_steps, eps_SG, eps_SG_emp, eps_SG_std, E_emp, eps_trunc, eps_trunc_emp, eps_trunc_std, E_trunc = track_method_epsilon(
                    estimator, E_GS, delta, params)
                filename = savename.format(mapping_name, label) + "-truncated_energies.txt"
                np.savetxt(savepath + filename, E_trunc, comments="", header=str(E_GS))
                eps_dict["Nsteps"] = N_steps
                eps_dict[label + "-truncated-emp"] = eps_trunc_emp
                eps_dict[label + "-truncated-STD"] = eps_trunc_std
                eps_dict[label + "-truncated-prov"] = eps_trunc
                print("method", method, "setting", len(estimator.settings_dict))
            else:
                # 入口
                N_steps, eps_SG, eps_SG_emp, eps_SG_std, E_emp = track_method_epsilon(estimator, E_GS, delta,
                                                                                      params, label=label)
                eps_dict["Nsteps"] = N_steps

                # 构造要写入的字符串
                content = f" {len(estimator.settings_dict)}\n"


            eps_dict[label + "-emp"] = eps_SG_emp
            eps_dict[label + "-STD"] = eps_SG_std
            eps_dict[label + "-prov"] = eps_SG
            filename = f"{savename}_{label}_energies.txt"

            np.savetxt(savepath + filename, E_emp, comments="", header=str(E_GS))
            print("Data for label <{}> generated.".format(label))

    # saving all data into one file
    df = pd.DataFrame.from_dict(eps_dict)
    vals = df.to_numpy()
    header = ""
    for key in df.columns:
        header += "{}\t".format(key)

    print("All methods' benchmark data generated / loaded.")

def LiH12():
    folder_Hamiltonians = "haozhaowu/"
    folder_OGM_settings = "OGM_probabilities/OGM_{}_{}{}.txt"  # format string to fill in {molecule}x{qubit_number}x{mapping}
    savepath = f"setting/LiH12/"
    savename = "_molecule_{}_{}"  # insert {mapping_name,method}

    # create temporary folder for storing outputs
    if not isdir(savepath):
        mkdir(savepath)

    molecule_name = "H2"  # choose one out of the molecules above
    mapping_name = "JW"  # choose one out of ["JW","BK","Parity"]
    basis_set = "sto3g"  # choose one out of ["sto3g","6-31g"] - the latter only for H2 molecule
    savename = f"LiH12"

    observables, w, offset, E_GS, state = load_pauli_list5(folder_Hamiltonians, molecule_name, basis_set,
                                                           mapping_name)
    # print("w", w)
    print("=====================================")
    # wrap ground state <state> into StateSampler in order to retrieve samples in arbitrary
    state_sampler = StateSampler(state)
    # fill in format string for Overlapped Grouping probabilities
    folder_OGM_settings = folder_OGM_settings.format(molecule_name, observables.shape[-1], mapping_name.lower())

    # hyperparameters for ShadowGrouping, see eq. (48) in manuscript
    alpha = np.max(np.abs(w)) / np.min(np.abs(w)) + np.min(np.abs(w))

    eps = 0.1  # accuracy in Hartree -- irrelevant for the benchmark below
    N_START = 100  # number of total measurement settings
    N_STOP = 1000
    N_runs = 50  # number of independent repetitions for the energy estimation
    N_plot = 10  # number of data points tracked
    delta = 0.02  # see caption of Figure 3 in manuscript
    methods = {}
    # methods["GC"] = GraphColoringGrouping(observables, w, eps, commutation_mode="fc")
    # methods["ShadowGrouping"]= Shadow_Grouping(observables,w,eps,Bernstein_bound(alpha=alpha)())
    # methods["ShadowGrouping"] = Shadow_Grouping(
    #     observables, w, eps, Bernstein_bound(alpha=alpha)(), commutation_mode="fc"
    # )
    # methods["Derandomization"] = Derandomization(observables, w, np.sqrt(0.9), use_one_norm=True)
    # methods["RandomPaulis"] = Derandomization(observables, w, eps, delta=1)  # delta controls the randomness
    # methods["AdaptivePaulis"] = AdaptiveShadows(observables,w)
    # methods["AEQuO"] = AEQuO(observables,w,offset,adaptiveness_L=2,interval_skewness_l=4,budget=N_STOP)
    # methods["Derandomization_fc"] = Derandomization(
    #     observables, w, np.sqrt(0.9), use_one_norm=True, commutation_mode="fc"
    # )
    # methods["RandomPaulis_fc"] = Derandomization(observables, w, eps, delta=1, commutation_mode="fc")

    # file = folder_OGM_settings.format(molecule_name,observables.shape[1],mapping_name.lower())

    file = "haozhaowu/LiH12/hamil_class/ogm_outputs/OGM_ogm_hamiltonian_LiH_sto3g_12jw.txt"
    print("file", file)
    if isfile(file):
        # methods["OverlappedGrouping"] = Overlapped_Grouping(observables,w,file)
        methods["OverlappedGrouping"] = Overlapped_Grouping(observables, w, file, commutation_mode="fc")
    # all details can be found in benchmark.py in track_method_epsilon() to generate the benchmark data
    eps_dict = {}
    benchmark_file = savepath + savename.format(mapping_name, "benchmark.txt")
    print("benchamrk", benchmark_file)
    # look whether a benchmark.txt file already exists and load the data
    # afterwards, check whether all methods in <methods> have been benchmarked
    # If not, benchmark them directly

    if isfile(benchmark_file):
        with open(benchmark_file, "r") as f:
            columns = f.readline().strip().split()
        data = np.loadtxt(benchmark_file, skiprows=1).T
        for column, row in zip(columns, data):
            eps_dict[column] = row
            if column.find("-prov") > -1:
                print("Data for label <{}> loaded from file.".format(column[:-5]))

    for label, method in methods.items():
        print("label", label)
        if eps_dict.get(label + "-emp", None) is None:
            print("Benchmarking method " + label, "...")
            params = {"Nshots": N_STOP, "Nsteps": N_plot, "Nreps": N_runs, "Nstart": N_START}
            if label.find("truncate") >= 0:
                params["truncate"] = True
                label = label[:label.find("-trun")]
            estimator = Energy_estimator(method, StateSampler(state), offset=offset)
            if params.get("truncate", False):
                # if the method truncates, track_method_epsilon() returns the truncated and the untruncated data
                N_steps, eps_SG, eps_SG_emp, eps_SG_std, E_emp, eps_trunc, eps_trunc_emp, eps_trunc_std, E_trunc = track_method_epsilon(
                    estimator, E_GS, delta, params)
                filename = savename.format(mapping_name, label) + "-truncated_energies.txt"
                np.savetxt(savepath + filename, E_trunc, comments="", header=str(E_GS))
                eps_dict["Nsteps"] = N_steps
                eps_dict[label + "-truncated-emp"] = eps_trunc_emp
                eps_dict[label + "-truncated-STD"] = eps_trunc_std
                eps_dict[label + "-truncated-prov"] = eps_trunc
                print("method", method, "setting", len(estimator.settings_dict))
            else:
                # 入口
                N_steps, eps_SG, eps_SG_emp, eps_SG_std, E_emp = track_method_epsilon(estimator, E_GS, delta,
                                                                                      params, label=label)
                eps_dict["Nsteps"] = N_steps

                # 构造要写入的字符串
                # 注意：print会自动换行，但写入文件需要手动在末尾加 '\n'
                content = f" {len(estimator.settings_dict)}\n"


            eps_dict[label + "-emp"] = eps_SG_emp
            eps_dict[label + "-STD"] = eps_SG_std
            eps_dict[label + "-prov"] = eps_SG
            filename = f"{savename}_{label}_energies.txt"

            np.savetxt(savepath + filename, E_emp, comments="", header=str(E_GS))
            # with open("settings.txt", "a", encoding="utf-8") as f:  # 'a' 模式追加内容
            #     print("setting number", len(estimator.settings_dict), file=f)
            # with open("settings.txt", "a", encoding="utf-8") as f:  # 'a' 模式追加内容
            #     print("shot number", 262144, file=f)
            # with open("settings.txt", "a", encoding="utf-8") as f:  # 'a' 模式追加内容
            #     print(label,estimator.settings_dict, file=f)

            # print(len(label))
            print("Data for label <{}> generated.".format(label))

    # saving all data into one file
    df = pd.DataFrame.from_dict(eps_dict)
    vals = df.to_numpy()
    header = ""
    for key in df.columns:
        header += "{}\t".format(key)
    # np.savetxt(savepath+savename.format(mapping_name,"benchmark.txt"),vals,header=header,comments="")

    print("All methods' benchmark data generated / loaded.")
if __name__ == "__main__":
    # h2
    # 设定起始值、结束值和步长
    # start_val = 0.4
    # end_val = 2.2
    # step = 0.3
    # current_val = start_val
    #
    # # 使用 while 循环
    # # 注意：加上一个微小的数 (0.001) 是为了防止浮点数精度导致漏掉最后一个 4.3
    # while current_val <= end_val + 0.001:
    #     # 【关键步骤】保留1位小数，解决浮点数精度问题
    #     # 这样传进去的就是 0.7 而不是 0.7000000001
    #     clean_val = round(current_val, 1)
    #
    #     print(f"--------------------------")
    #     print(f"正在处理: {clean_val}")
    #
    #     # 调用函数
    #     f(clean_val)
    #
    #     # 增加步长
    #     current_val += step
    # start_time = perf_counter()
    # # # random
    # random1("dense", 6)
    # print(f"总耗时: {perf_counter() - start_time:.2f} 秒")

    start_time = perf_counter()
    # H2O()
    BeH2()
    # # random
    # klocal()
    # LiH12()

    print(f"总耗时: {perf_counter() - start_time:.2f} 秒")

    # random1("dense", 5)
    # random1("dense", 6)
    # for i in range(3, 7):
    #     random1("sparse", i)
    # random1("dense", 3)
    # random1("sparse", 3)
    # random1("sparse", 4)
    # random1("sparse", 6)

    # random1("sparse", 7)
    # random1("dense", 7)

    # heisenberg(3)
    # heisenberg(4)
    # heisenberg(5)
    # heisenberg(6)
    # heisenberg(4)
    # heisenberg(6)

    # 1. 定义范围 0.8 到 2.0 (不包含2.1)，步长 0.2
    # range1 = np.arange(1.0, 1.3, 0.2)
    #
    # # 2. 定义额外点
    # range2 = []
    #
    # # 3. 合并并遍历
    # # np.concatenate 将两个数组拼接
    # all_vals = np.concatenate([range1, range2])
    #
    # for val in all_vals:
    #     # 【关键】四舍五入保留1位小数，防止出现 1.20000000002 这种情况
    #     val = round(val, 1)
    #
    #     # 调用你的函数
    #     liH(val)
