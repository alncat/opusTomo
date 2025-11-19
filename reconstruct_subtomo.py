import os
import subprocess
import multiprocessing
# define output folder path
folder_path = './subtomos'
def task(folder_path, filename, prefix):
    print(f"File: {filename}, Prefix: {prefix}")
    result = subprocess.run(['relion_reconstruct', '--i', os.path.join(folder_path, filename),
                             '--o', os.path.join(folder_path, prefix+'.mrc'), '--skip_gridding'],
                            capture_output=True, text=True)
    if result.returncode != 0:
        print("reconstruct failed!")
        print("STDOUT:\n", result.stdout)
        print("STDERR:\n", result.stderr)
    assert result.returncode == 0
    return result
# loop through all files
with multiprocessing.Pool(processes=48) as pool:
    results = []
    for filename in os.listdir(folder_path):
        # check if the file ends with _ctf.star
        if filename.endswith('_subtomo.star'):
        # extract prefix
            prefix = filename[:-len('_subtomo.star')]  # remove '_subtomo.star' suffix
            # call external commands
            result = pool.apply_async(task, args=(folder_path, filename, prefix))
            results.append(result)

    for result in results:
        result.get()

print("All tasks done!")

