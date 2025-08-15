import os
import argparse
import subprocess
import logging
import yaml

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
)
logger = logging.getLogger("run_all.py")
HOUMO_TARGET = os.getenv('HOUMO_TARGET', 'houmo')
script_dir = os.path.dirname(os.path.abspath(__file__))


def parseArgs():
    parser = argparse.ArgumentParser()
    parser.add_argument("--type", help="print type name")
    parser.add_argument("--diff_file", help="diff_file path")
    parser.add_argument(
        "--config", help="config path", default="./config/imodelExampleConfig.yaml"
    )
    allArgs = parser.parse_args()
    return allArgs


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
        with open(config, "r") as file:
            yamlData = yaml.load(file, Loader=yaml.FullLoader)
    except FileNotFoundError:
        raise FileNotFoundError(f"File '{config}' does not exist.")

    return yamlData


def addCiMarker(test_type: str, filter_str: str):
    module_name = None
    case_dict = dict()
    for line in filter_str.split('\n'):
        line = line.strip()
        if "<Module" in line:
            module_name = line.rsplit(" ", 1)[-1][:-1].strip()
            case_dict[module_name] = list()
        elif "<Function" in line and module_name:
            func_name = line.rsplit(" ", 1)[-1][:-1].strip()
            case_dict[module_name].append(func_name)

    file_folder = script_dir + "/tests/" + test_type
    # add ci marker: imodelzoo
    for py_name, funcs in case_dict.items():
        file_path = file_folder + "/" + py_name
        logger.info(f"Add ci marker into file_path:{funcs}")

        func_indexes = list()
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            for idx, line in enumerate(lines):
                line = line.strip()
                for func in funcs:
                    func_str = "def " + func + "("
                    if func_str in line:
                        func_indexes.append(idx)
                        break
        for idx in reversed(func_indexes):
            logger.info(f"Add ci marker to testcase func:{lines[idx]}")
            lines.insert(idx, "@pytest.mark.imodelzoo\n")
        with open(file_path, 'w', encoding='utf-8') as f:
            f.writelines(lines)


def getTestModules(line, yamlData):
    allModules = set()
    if yamlData == None:
        return allModules
    for key, value in yamlData.items():
        for keyin, valuein in value.items():
            if keyin != "include" or valuein == None:
                continue
            for include in valuein:
                if not line.startswith(include):
                    continue
                allModules.add(key)
    for key, value in yamlData.items():
        for keyin, valuein in value.items():
            if keyin != "exclude" or valuein == None:
                continue
            for exclude in valuein:
                if not line.startswith(exclude):
                    continue
                allModules.remove(key)

    return allModules


def getTestUnit(testModule, yamlData):
    unitTests = set()
    if yamlData == None:
        return None
    for key, value in yamlData.items():
        if key != testModule:
            continue
        testcase = value.get("test", None)
        if testcase is not None:
            unitTests.update(value.get("test"))

    return unitTests


def getUnitDict(testUnit, yamlData):
    if yamlData == None:
        return
    finalUnitDict = {}
    for key, value in yamlData.items():
        for keyin, valuein in value.items():
            if keyin != "example_case" or valuein == None:
                continue
            unitDict = {
                casekey: caseValue
                for case in valuein
                for casekey, caseValue in case.items()
                if casekey == testUnit
            }
            if len(unitDict) == 0:
                continue
            finalUnitDict.update(unitDict)
    return finalUnitDict


def runCase(allUnitDict):
    pytest_folder = script_dir + "/tests"
    os.chdir(pytest_folder)
    logger.info(f"The path for executing pytest tests: {os.getcwd()}")
    valid_args = [
        "all",
        "quant",
        "compile",
        "demo",
        "compare",
        "eval",
        "perf",
    ]
    for caseName, case in allUnitDict.items():
        test_type = case.get("test_type", None)
        script = case.get("script", None)
        args = case.get("args", None)

        if script is None or len(script) == 0:
            logger.info(f"<=== case {caseName} is not execute, {caseName} end")
            continue
        logger.info(f"===> begin test: {test_type}/{caseName}")

        if ".sh" in script:
            cmd_list = ["bash", script]
        else:
            if test_type is None or args is None or args not in valid_args:
                logger.info(f"<=== case {caseName} is not execute, {caseName} end")
                continue
            cmd_list = [
                "pytest",
                "--no-header",
                "--collect-only",
                test_type,
                "-m",
                script,
            ]

        logger.info(f"---> test {caseName} start, test cmd: {cmd_list}")
        process = subprocess.Popen(
            cmd_list, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
        )
        stdout, stderr = process.communicate(timeout=7200)  # timeout: 2h
        # logger.info(f"[testcase log] stdout:\n {stdout}")
        # logger.info(f"[testcase log] stderr:\n {stderr}")
        if process.returncode == 0 and ".sh" in script:
            logger.info(f"<--- test {caseName} success")
        elif process.returncode == 0:
            addCiMarker(test_type, stdout)
            logger.info(f"<--- test {test_type}/{caseName} success")
        else:
            raise RuntimeError(
                f"<--- test {caseName} fail, error code: {process.returncode}"
            )


def runWithDiff(allArgs):
    allTestModules = set()
    allTestUnits = set()
    allUnitDict = {}
    allDiffFiles = readFile(allArgs.diff_file)
    yamlData = readWithYaml(allArgs.config)
    for line in allDiffFiles:
        testModules = getTestModules(line, yamlData)
        allTestModules.update(testModules)
    for testModule in allTestModules:
        testUnits = getTestUnit(testModule, yamlData)
        allTestUnits.update(testUnits)
    for testUnit in allTestUnits:
        unitDict = getUnitDict(testUnit, yamlData)
        allUnitDict.update(unitDict)
    if HOUMO_TARGET == "xh1":
        os.system("pip3 install -r requirements-xh1.txt")
    else:
        os.system("pip3 install -r requirements-xh2.txt")
    logger.info(f"test modules: {allTestModules}")
    logger.info(f"test units: {allTestUnits}")
    logger.info(f"test cases: {allUnitDict}")
    runCase(allUnitDict)

    pytest_folder = script_dir + "/tests"
    os.chdir(pytest_folder)
    logger.info(f"[execution log] test imodelzoo cases...")
    # "pytest --log-cli-level=INFO -s -m imodelzoo"
    result = os.system("pytest --collect-only -s -m imodelzoo")
    # logger.info(f"[execution log] ret: {result}")


def main(allArgs=None):
    if allArgs == None:
        allArgs = parseArgs()
    if allArgs.type == "diff_file":
        runWithDiff(allArgs)
    else:
        raise Exception("script type is not recognition, begin exit")


if __name__ == "__main__":
    os.environ['HOUMO_MODELZOO_URL'] = (
        "http://10.10.1.53:8082/artifactory/toolchain/release"
    )
    os.environ['HOUMO_MODEL_PATH'] = "/data02/modelzoo_ci/models"
    os.system("cd hmatc && chmod +x install.sh && ./install.sh")
    HOUMO_DATASETS_PATH = os.getenv("HOUMO_DATASETS_PATH")
    imagenet_dir = os.path.join(HOUMO_DATASETS_PATH, "imagenet")
    os.environ['HOUMO_EXAMPLES_PATH'] = f"{script_dir}/apis"
    os.system(f"cp data/datasets/imagenet/synset_1000.txt {imagenet_dir}")
    os.system(f"cp data/datasets/imagenet/val.txt {imagenet_dir}")
    # os.system("wget http://10.10.1.53:8082/artifactory/toolchain/support/xh2_extra_libs.zip")
    # os.system("unzip xh2_extra_libs.zip -d xh2_extra_libs")
    # os.system("cp xh2_extra_libs/* /usr/local/houmo/lib")

    os.environ['SKIP_INFER'] = "ON"
    model_dir = os.path.join(script_dir, "tests/model_results")
    os.makedirs(model_dir, exist_ok=True)
    # install pytest in release docker
    os.system("pip3 install pytest")
    os.system("pip3 install pytest-xdist")

    main()
