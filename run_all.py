import os
import argparse
import subprocess
import yaml


def parseArgs():
    parser = argparse.ArgumentParser()
    parser.add_argument("--type", help="print type name")
    parser.add_argument("--diff_file",help="diff_file path")
    parser.add_argument("--config",help="config path",default="./config/imodelExampleConfig.yaml")
    allArgs = parser.parse_args()
    return allArgs

def runAll():
    cmd = "bash run_all.sh"
    print("------------- run_all start ------------")
    result = os.system(cmd)
    if result == 0:
        print("------------- run_all success ------------")
    else:
        raise RuntimeError("run_all fail, error={}". format(result))

    # proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, shell=True,encoding='utf-8')
    # (out, _) = proc.communicate()
    # if proc.returncode != 0:
    #     msg = cmd + "\n" + out
    #     raise RuntimeError(msg)
    # else:
    #     print("------------- test all case success ------------")

def readFile(diff_file):
    try:
        with open(diff_file) as file:
            content = file.read()
            allLines = content.strip().split("\n")
            return allLines
    except FileNotFoundError:
        raise FileNotFoundError(f"File '{diff_file}' does not exist.")

def readWithYaml(config):
    try:
        with open(config,"r") as file:
            yamlData = yaml.load(file,Loader=yaml.FullLoader)
    except FileNotFoundError:
        raise FileNotFoundError(f"File '{config}' does not exist.")

    return yamlData

def getTestModules(line,yamlData):
    allModules = set()
    if yamlData == None:
        return allModules
    for key,value in yamlData.items():
        for keyin,valuein in value.items():
            if keyin != "include" or valuein == None:
                continue
            for include in valuein:
                if not line.startswith(include):
                    continue
                allModules.add(key)
    for key,value in yamlData.items():
        for keyin,valuein in value.items():
            if keyin != "exclude" or valuein == None:
                continue
            for exclude in valuein:
                if not line.startswith(exclude):
                    continue
                allModules.remove(key)

    return allModules

def getTestUnit(testModule,yamlData):
    unitTests = set()
    if yamlData == None:
        return None
    for key, value in yamlData.items():
        if key != testModule:
            continue
        unitTests.update(value.get("test"))

    return unitTests

def getUnitDict(testUnit,yamlData):
    if yamlData == None:
        return
    finalUnitDict = {}
    for key,value in yamlData.items():
        for keyin, valuein in value.items():
            if keyin != "example_case" or valuein == None:
                continue
            unitDict = {casekey: caseValue for case in valuein for casekey,caseValue in case.items() if casekey == testUnit }
            if len(unitDict) == 0:
                continue
            finalUnitDict.update(unitDict)
    return finalUnitDict

def runCase(allUnitDict):
    for caseName,case in allUnitDict.items():
        print("------------ begin test " + caseName + " ------------")
        script = case.get("script")
        args = case.get("args")
        if script == None or len(script) == 0:
            print("case " + caseName + "is not excute, " + caseName + " end")
            continue
        cmd = "bash " + script
        # if args == None or len(args) == 0:
        #     cmd = "bash " + script
        # else:
        #     argStr = ""
        #     for arg in args:
        #         argStr = argStr + arg + " "
        #     cmd = "bash " + script + " " + argStr

        print("------------- test " + caseName + " start ------------")
        result = os.system(cmd)
        if result == 0:
            print("------------- test " + caseName + " success ------------")
        else:
            raise RuntimeError("test " + caseName + " fail, error={}". format(result))

        # proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, shell=True,encoding='utf-8')
        # (out, _) = proc.communicate()
        # if proc.returncode != 0:
        #     msg = "command is:" + cmd + "\n" + out
        #     raise RuntimeError(msg)
        # else:
        #     print("------------- test " + caseName + " success ------------")


def runWithDiff(allArgs):
    allTestModules = set()
    allTestUnits = set()
    allUnitDict = {}
    allDiffFiles = readFile(allArgs.diff_file)
    yamlData = readWithYaml(allArgs.config)
    for line in allDiffFiles:
        testModules = getTestModules(line,yamlData)
        allTestModules.update(testModules)
    for testModule in allTestModules:
        testUnits = getTestUnit(testModule,yamlData)
        allTestUnits.update(testUnits)
    for testUnit in allTestUnits:
        unitDict = getUnitDict(testUnit,yamlData)
        allUnitDict.update(unitDict)
    os.system("pip3 install -r requirements.txt")
    print("test modules:", allTestModules)
    print("test units:", allTestUnits)
    print("test cases:", allUnitDict)
    runCase(allUnitDict)

def main(allArgs=None):
    if allArgs == None:
        allArgs = parseArgs()
    if allArgs.type == "diff_file":
        runWithDiff(allArgs)
    else:
        raise Exception("script type is not recognition, begin exit")

if __name__ == "__main__":
    os.environ['MODELZOO_URL'] = "http://10.10.1.53:8082/artifactory/toolchain/release"
    os.environ['MODEL_PATH'] = "/data02/modelzoo_ci/models"
    main()
