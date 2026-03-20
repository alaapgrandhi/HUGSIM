import subprocess
import time
import os


def launch(shell_path, cuda_id, output, checkpoint_path=None, image_size=None):
    os.makedirs(output, exist_ok=True)
    print(os.path.join(output, 'output.txt'))
    print(shell_path, cuda_id, output)
    with open(os.path.join(output, 'output.txt'), 'w') as f:
        if image_size is not None:
            image_size_str = ",".join(map(str, image_size))
            process = subprocess.Popen(
                ["bash", shell_path, cuda_id, output, checkpoint_path, image_size_str], stdout=f, stderr=f
            )
        else:
            process = subprocess.Popen(
                ["bash", shell_path, cuda_id, output, checkpoint_path], stdout=f, stderr=f
            )
    return process


def check_alive(process, tolerant=100):
    i = 0
    while i < tolerant:
        return_code = process.poll()
        if return_code is not None:
            print(f"The AD algorithm completed with return code {return_code}.")
            process.kill()
            return
        elif i % 5 == 0:
            print(f"The AD algorithm is still running, remaining tolerant {tolerant - i}.")
        time.sleep(1)
        i += 1
    process.kill()
    print("The AD algorithm process is killed.")