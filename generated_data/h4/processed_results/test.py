import os


def batch_rename_files():
    # 获取当前脚本所在的目录
    current_directory = os.getcwd()

    # 获取目录下所有文件
    files = os.listdir(current_directory)

    count = 0
    print(f"正在扫描目录: {current_directory} ...\n")

    for filename in files:
        # 检查文件名是否以 "H4_" 开头
        if filename.startswith("H4_"):
            # 定义旧路径
            old_path = os.path.join(current_directory, filename)

            # 定义新文件名：利用切片 filename[3:] 去掉前3个字符 ("H4_")
            new_filename = filename[3:]

            # 或者使用 replace 方法，只替换第一个匹配项
            # new_filename = filename.replace("H4_", "", 1)

            # 定义新路径
            new_path = os.path.join(current_directory, new_filename)

            try:
                # 执行重命名
                os.rename(old_path, new_path)
                print(f"[成功] {filename} -> {new_filename}")
                count += 1
            except Exception as e:
                print(f"[失败] 无法重命名 {filename}. 原因: {e}")

    print(f"\n完成！共重命名了 {count} 个文件。")


if __name__ == "__main__":
    # 为了防止误操作，建议先备份文件
    # 确认执行
    user_input = input("即将把当前目录下所有 'H4_' 开头的文件重命名（去掉前缀），是否继续？(y/n): ")
    if user_input.lower() == 'y':
        batch_rename_files()
    else:
        print("操作已取消。")