% %author: Bujiao Wu
% 
% clear all;
% 
% 
% x = 1; %input(prompt); %%x in {1,2}
% 
% %f = input('Please input file (E.g.: H2_4jw/H2_4bk/H2_4parity/...)\n','s');
% %f_front_set = {'H_', 'sym_H_', 'DM_H_Squared_', 'DM_sym_H_Squared_'};
% %f_front_set = {'KT_H_', 'KT_sym_H_', 'KT_partial_sym_H_'};
% %f_front_set = {'KT_H_Squared_', 'KT_sym_H_Squared_'};
% % f_front_set = {'ogm_H2_0.4.txt'}; 
% 
% 
% T = 100000;
% 
% 
% % for n = 3 : 4
% % 	for j = 1: length(f_front_set)
% %         f_front = f_front_set{j};
% %         f = strcat(f_front, num2str(n));
% %         f = strcat(f,'.txt');
% %         fprintf('OGMV1 cutdown:\n');
% %         time = main_CutOGM(f, T);
% % 	end
% % end
% 
% % vals = 0.4 : 0.3 : 4.3;
% % folder_dir = 'ogm_inputs/'; 
% % 
% % % 循环遍历每一个键长
% % for i = 1 : length(vals)
% %     val = vals(i);
% %     
% %     % 1. 构造文件名
% %     % 使用 sprintf '%.1f' 强制保留1位小数，生成如 '0.4', '0.7' 等字符串
% %     val_str = sprintf('%.1f', val);
% %     
% %     % 拼接文件名: ogm_H2_0.4.txt
% %     filename = strcat('ogm_H2_', val_str, '.txt');
% %     
% %     
% %     fprintf('\n--------------------------------------\n');
% %     fprintf('正在处理键长: %s\n', val_str);
% %     fprintf('文件路径: %s\n', filename);
% %     
% %     % 3. 检查文件是否存在并运行
% %     if exist(filename, 'file') == 2    
% %         fprintf('OGMV1 cutdown Start:\n');
% %         % 调用核心函数
% %         time = main_CutOGM(filename, T);
% %     else
% %         fprintf('❌ 错误: 找不到文件 %s\n', filename);
% %         fprintf('   请检查 Python 转换脚本是否已运行，以及 folder_dir 路径是否正确。\n');
% %     end
% % end
% % ogm_LiH_0.8.txt
% % f_front_set = {'ogm_hamiltonian_2dgrid_'};
% % n = 6
% % for j = 1: length(f_front_set)
% %     f_front = f_front_set{j};
% %     f_front
% %     f = strcat(f_front, num2str(n));
% %     f = strcat(f,'.txt');
% %     fprintf('OGMV1 cutdown:\n');
% %     time = main_CutOGM(f, T);
% % end
% clear all; clc;
% 
% % ================= 配置区域 =================
% % 1. 定义文件所在的文件夹路径
% % (根据你上一步 Python 脚本的 output_dir，应该是这里)
% 
% % 2. 采样次数
% T = 100000;
% 
% % 3. 定义键长范围
% % range1: 0.8, 1.0, ..., 2.0
% range1 = 2.2 : 0.2 : 4.0;
% % range2: 2.5, 3.0
% range2 = [2.5, 3.0];
% % 合并成一个数组
% vals = [range1, range2];
% % ===========================================
% 
% fprintf('=== 开始批量处理 LiH OGM ===\n');
% 
% % 循环遍历每一个键长
% for i = 1 : length(vals)
%     val = vals(i);
%     
%     % 1. 构造文件名字符串
%     % sprintf '%.1f' 保证生成 '0.8', '1.0' 这种格式
%     val_str = sprintf('%.1f', val);
%     
%     % 拼接文件名: ogm_LiH_0.8.txt
%     filename = strcat('ogm_LiH_', val_str, '.txt');
%     
%     % 2. 拼接完整路径: hamil_class/ogm_inputs/ogm_LiH_0.8.txt
%     f =  filename;
%     
%     fprintf('正在处理: %s\n', filename);
%     
%     % 3. 检查文件并运行
%     if exist(f, 'file') == 2
%         fprintf('OGMV1 cutdown:\n');
%         % 调用核心函数
%         time = main_CutOGM(f, T);
%     else
%         fprintf('❌ 错误: 找不到文件 %s\n', f);
%         fprintf('   请检查路径 folder_dir 是否正确。\n');
%     end
%     fprintf('--------------------------------\n');
% end

clear all; 
clc;

% ================= 配置区域 =================
% 1. 指定目标文件名
% 注意：根据文件名 n7，这通常代表一个 7 比特的随机哈密顿量
% 如果你之后生成了 14 比特的水分子文件，只需修改此处的文件名即可
% f = 'ogm_hamiltonian_klocal_random_n7_k3_terms98.txt';
f = 'ogm_hamiltonian_klocal_random_n7_k3_terms400.txt';


% 2. 采样总次数 (T)
% 你可以根据需要调整这个数值，10^5 是你代码中的默认值
T = 100000; 

% ================= 执行区域 =================
fprintf('=== 开始处理特定哈密顿量文件 ===\n');
fprintf('目标文件: %s\n', f);
fprintf('采样次数 T: %d\n', T);
fprintf('--------------------------------\n');

% 检查文件是否存在
if exist(f, 'file') == 2
    fprintf('状态: 文件已找到。正在启动 OGMV1 cutdown 算法...\n');
    
    % 记录开始时间
    tic;
    
    % 调用核心算法函数
    % 该函数会处理对易分组逻辑并输出测量优化结果
    time_elapsed = main_CutOGM(f, T);
    
    toc;
    fprintf('任务完成！\n');
else
    fprintf('❌ 错误: 找不到文件 "%s"\n', f);
    fprintf('请确保该 .txt 文件位于 MATLAB 当前的工作路径下。\n');
end
fprintf('================================\n');

