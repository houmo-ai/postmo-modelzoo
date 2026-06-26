import os
import argparse
import subprocess
import logging
import sys
import yaml

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("run_all.py")
HOUMO_TARGET = os.getenv("HOUMO_TARGET", "houmo")
script_dir = os.path.dirname(os.path.abspath(__file__))


def run_command(cmd, cwd=None, timeout=None, capture_output=False):
    logger.info(f"[command] run: {cmd}, cwd: {cwd or os.getcwd()}")
    return subprocess.run(
        cmd,
        cwd=cwd,
        timeout=timeout,
        check=True,
        text=True,
        capture_output=capture_output,
    )


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


def path_matches_rule(line: str, rule: str) -> bool:
    if not line or not rule:
        return False

    line_path = os.path.normpath(line.strip()).replace("\\", "/")
    rule_path = os.path.normpath(rule.strip()).replace("\\", "/")

    return line_path == rule_path or line_path.startswith(f"{rule_path}/")


def get_module_rules(module_config, rule_type: str):
    rules = module_config.get(rule_type)
    return rules or []


def has_matching_rule(line: str, rules) -> bool:
    return any(path_matches_rule(line, rule) for rule in rules)


def addCiMarker(test_type: str, filter_str: str):
    module_name = None
    case_dict = dict()
    for line in filter_str.split("\n"):
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
        with open(file_path, "r", encoding="utf-8") as f:
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
        with open(file_path, "w", encoding="utf-8") as f:
            f.writelines(lines)


def getTestModules(line, yamlData):
    allModules = set()
    if yamlData == None:
        return allModules

    for key, value in yamlData.items():
        include_rules = get_module_rules(value, "include")
        if has_matching_rule(line, include_rules):
            allModules.add(key)

    if not allModules:
        return allModules

    for key, value in yamlData.items():
        exclude_rules = get_module_rules(value, "exclude")
        if has_matching_rule(line, exclude_rules):
            allModules.discard(key)

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
    """Execute all test cases, if the script is .sh, execute with bash,
    else execute with pytest, and add ci marker for pytest cases.
    """
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
        try:
            result = run_command(
                cmd_list, timeout=3600, capture_output=True
            )  # timeout: 1h
        except subprocess.CalledProcessError as e:
            # pytest --collect-only commands should not raise exceptions
            if "pytest" in cmd_list and "--collect-only" in cmd_list:
                logger.info(
                    f"[testcase log] pytest --collect-only returned {e.returncode}, treating as success"
                )
            else:
                if e.stdout:
                    logger.info(f"[testcase log] stdout:\n {e.stdout}")
                if e.stderr:
                    logger.info(f"[testcase log] stderr:\n {e.stderr}")
                raise RuntimeError(
                    f"<--- test {caseName} fail, error code: {e.returncode}"
                ) from e

        if ".sh" in script:
            logger.info(f"<--- test {caseName} success")
        else:
            addCiMarker(test_type, result.stdout)
            logger.info(f"<--- test {test_type}/{caseName} success")


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
    run_command([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
    logger.info(f"test modules: {allTestModules}")
    logger.info(f"test units: {allTestUnits}")
    logger.info(f"test cases: {allUnitDict}")
    runCase(allUnitDict)

    pytest_folder = script_dir + "/tests"
    os.chdir(pytest_folder)
    logger.info(f"[execution log] test imodelzoo cases...")
    # "pytest --log-cli-level=INFO -s -m imodelzoo"
    run_command(["pytest", "--collect-only", "-s", "-m", "imodelzoo"])


def main(allArgs=None):
    if allArgs == None:
        allArgs = parseArgs()
    if allArgs.type == "diff_file":
        runWithDiff(allArgs)
    else:
        raise Exception("script type is not recognition, begin exit")


def setup_environment():
    HOUMO_EXAMPLES_PATH = os.path.abspath(os.getenv("HOUMO_EXAMPLES_PATH", "."))
    hmatc_dir = os.path.join(script_dir, "hmatc")
    llm_perf_dir = os.path.join(HOUMO_EXAMPLES_PATH, "tools", "llm_perf")
    llm_perf_bin = os.path.join(HOUMO_EXAMPLES_PATH, "tools", "bin", "llm_perf")
    hmeval_install = os.path.join(
        HOUMO_EXAMPLES_PATH, "tools", "hmeval", "scripts", "install.sh"
    )

    # install hmatc
    run_command(["chmod", "+x", "install.sh"], cwd=hmatc_dir)
    run_command(["./install.sh"], cwd=hmatc_dir)
    # install llm_perf
    run_command(["bash", "build_linux.sh"], cwd=llm_perf_dir)
    run_command(["cp", llm_perf_bin, os.path.join(HOUMO_PATH, "bin")])
    # install hmeval
    run_command([hmeval_install])

    # install pytest in release docker
    run_command([sys.executable, "-m", "pip", "install", "pytest"])
    run_command([sys.executable, "-m", "pip", "install", "pytest-xdist"])
    run_command([sys.executable, "-m", "pip", "install", "pytest-dependency"])


if __name__ == "__main__":
    os.environ["HOUMO_MODEL_PATH"] = "/data02/modelzoo_ci/models"
    HOUMO_PATH = os.getenv("HOUMO_PATH", "/usr/local/houmo")
    os.environ["SKIP_INFER"] = "ON"

    try:
        setup_environment()
        sys.exit(main() or 0)
    except subprocess.CalledProcessError as e:
        logger.error(f"[command] failed: {e.cmd}, return code: {e.returncode}")
        sys.exit(e.returncode)
    except subprocess.TimeoutExpired as e:
        logger.error(f"[command] timeout: {e.cmd}, timeout: {e.timeout}s")
        sys.exit(1)
    except OSError as e:
        logger.error(f"[command] execute failed: {e}")
        sys.exit(1)
