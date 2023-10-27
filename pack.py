import os
import sys

version = ""
if len(sys.argv) > 1:
    version = sys.argv[1]
include_list = []
exclude_list = ["pack.sh", "pack.py", "run_all.sh", "env_dev.sh", "tests"]
dir_name = "houmo-modelzoo"

# include项
include_cmd = ""
for file in include_list:
    include_cmd += " " + dir_name + "/" + file
if include_cmd == "":
    include_cmd = " " + dir_name

# exclude项
exclude_cmd = " --exclude=.git*"
for file in exclude_list:
    exclude_cmd += " --exclude=" + dir_name + "/" + file
  
# tar
postfix = ""
if version != "":
    postfix = "_" + version
tar_cmd = '''tar -czf ''' + dir_name + postfix + ".tar.gz " + exclude_cmd + include_cmd
print(tar_cmd)

os.system("git clone ssh://gerrit.houmo.ai:29418/toolchain/imodelzoo -b master " + dir_name)
os.system(tar_cmd)
print("tar done.")

