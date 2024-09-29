import os
import sys

version = ""
if len(sys.argv) > 1:
    version = sys.argv[1]
include_list = ["data", "hmassist", "hmodel", "models", "utils/common", "utils/tcim_perf", "release.cmake",
                "requirements.txt", "env.sh", "benchmark.yml", "README.md"]
exclude_list = ['models/backbone/vit']
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

from git import Repo
repo = Repo('.')
branch_name = repo.active_branch.name
cmd = f"git clone -b {branch_name} http://jenkinspublic:hmCI2%4022!@gerrit.houmo.ai/toolchain/imodelzoo {dir_name}"
print(cmd)
os.system(cmd)

# tar
postfix = ""
if version != "":
    postfix = "_" + version
cmd = 'tar -czf ' + dir_name + postfix + ".tar.gz " + exclude_cmd + include_cmd
print(cmd)

os.system(cmd)
print("tar done.")
