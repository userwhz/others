from OGM_meas_convert import compute_pauli_counts

# 有问题 不要用
# Example Usage

file_front_set = ["rand_H_","sym_rand_H_", "H_", "sym_H_"]
path = r"/Users/wuhaozhao/Downloads/python_project/new_datas_afterJun18/shadowgrouping-master/meas_randH1/"
# path = r"/Users/wuhaozhao/Downloads/python_project/new_datas_afterJun18/OGM_optimization/test"
#r"/Users/bujiaowu/Documents/program/symmetry_state_photons/new_datas_afterJun18/shadowgrouping-master/meas"
# AllNumSamples = [100+20*3**j for j in range(1,7)]
AllNumSamples = [12,45,160,572,2038,7259,25848]
for n in range(3,7):
    for file_front in file_front_set:
        for N in AllNumSamples:
            output_file = path + "/OGM_" + file_front + str(n)+ "_" + str(N) + ".txt"
            input_file = "CutSet/OGM_" + file_front + str(n)+ ".txt"  # Replace with actual file name
            compute_pauli_counts(input_file,output_file, N )
