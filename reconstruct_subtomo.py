import os
import subprocess
import multiprocessing
# 定义文件夹路径
folder_path = './subtomos'
def task(folder_path, filename, prefix):
    print(f"File: {filename}, Prefix: {prefix}")
    result = subprocess.run(['relion_reconstruct', '--i', os.path.join(folder_path, filename),
                             '--o', os.path.join(folder_path, prefix+'.mrc'), '--skip_gridding'],
                            capture_output=True, text=True)
    assert result.returncode == 0
    return result
# 遍历文件夹中的所有文件
with multiprocessing.Pool(processes=48) as pool:
    results = []
    for filename in os.listdir(folder_path):
        # 检查文件是否以 _ctf.star 结尾
        if filename.endswith('_subtomo.star'):
        # 提取前缀
            prefix = filename[:-len('_subtomo.star')]  # 去掉最后的 '_subtomo.star'
            # 调用外部命令
            result = pool.apply_async(task, args=(folder_path, filename, prefix))
            results.append(result)

    for result in results:
        result.get()

print("All tasks done!")

