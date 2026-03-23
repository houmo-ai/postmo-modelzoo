# Copyright 2025 HOUMO AI
#
# File: tests_pyvenv_utils.py
# Description:
#   Python virtual environment utilities module.
#   This module provides utility functions for creating and managing Python virtual
#   environments specifically designed for testing iModelZoo applications.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SPDX-License-Identifier: Apache-2.0

import os
import sys
import subprocess
from pathlib import Path
import logging


logger = logging.getLogger(__name__)


VENV_NAME = "imodelzoo_test"


def get_imodelzoo_root() -> Path:
    """
    Get the repository root directory for iModelZoo.

    Returns:
        Path: Absolute path to the repository root directory
    """
    return Path(__file__).resolve().parents[2]


def get_xh_model_zoo_root() -> Path:
    """
    Get the parent directory that contains the `xh_model_zoo` package.

    Returns:
        Path: Absolute path to `hmodel/xh2`
    """
    return get_imodelzoo_root() / "hmodel" / "xh2"


def _join_pythonpath(*paths: str) -> str:
    """
    Join PYTHONPATH entries while skipping empty values.

    Args:
        *paths (str): PYTHONPATH entries to join

    Returns:
        str: Colon-separated PYTHONPATH value
    """
    return ":".join(path for path in paths if path)


def get_python_executable() -> str:
    """
    Get the absolute path of the python3 executable.

    Returns:
        str: Absolute path to the python3 executable

    Raises:
        RuntimeError: If python3 executable cannot be found
    """
    try:
        result = subprocess.check_output(
            ["bash", "-c", "command -v python3"], text=True, stderr=subprocess.STDOUT
        )
        return result.strip()
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"Python3 executable not found: {e.output}") from e


def get_site_packages(python_exe: str) -> str:
    """
    Get the site-packages directory path for a specified Python interpreter.

    Args:
        python_exe (str): Path to the Python executable

    Returns:
        str: Path to the site-packages directory

    Raises:
        RuntimeError: If getting site-packages fails
    """
    try:
        result = subprocess.check_output(
            [python_exe, "-c", "import site; print(site.getsitepackages()[0])"],
            text=True,
            stderr=subprocess.STDOUT,
        )
        return result.strip()
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"Failed to get site-packages: {e.output}") from e


def modify_activate_script(
    activate_path: str, venv_site: str, original_site: str
) -> None:
    """
    Modify the activate script to configure PYTHONPATH environment variable.

    Args:
        activate_path (str): Path to the activate script
        venv_site (str): Site-packages path of the virtual environment
        original_site (str): Original site-packages path

    Raises:
        RuntimeError: If modifying the activate script fails
    """
    try:
        with open(activate_path, "a", encoding="utf-8") as f:
            f.write("export ORIGINAL_PYTHONPATH=$PYTHONPATH\n")
            f.write(
                f"export PYTHONPATH={venv_site}:{original_site}:$ORIGINAL_PYTHONPATH\n"
            )
    except IOError as e:
        raise RuntimeError(f"Failed to modify activate script: {e}") from e


def modify_deactivate_script(deactivate_path: str) -> None:
    """
    Modify the deactivate script to restore original PYTHONPATH and clear temp variables.

    Args:
        deactivate_path (str): Path to the deactivate script

    Raises:
        RuntimeError: If modifying the deactivate script fails
    """
    try:
        with open(deactivate_path, "a", encoding="utf-8") as f:
            f.write("export PYTHONPATH=$ORIGINAL_PYTHONPATH\n")
            f.write("unset ORIGINAL_PYTHONPATH\n")
    except IOError as e:
        raise RuntimeError(f"Failed to modify deactivate script: {e}") from e


def modify_pyvenv_cfg(cfg_path: str) -> None:
    """
    Modify pyvenv.cfg to set include-system-site-packages to false.

    Args:
        cfg_path (str): Path to the pyvenv.cfg file

    Raises:
        RuntimeError: If modifying the pyvenv.cfg file fails
    """
    try:
        # Read file content and replace the specified line
        with open(cfg_path, "r", encoding="utf-8") as f:
            content = f.read()

        modified_content = content.replace(
            "include-system-site-packages = true",
            "include-system-site-packages = false",
        )

        with open(cfg_path, "w", encoding="utf-8") as f:
            f.write(modified_content)
    except IOError as e:
        raise RuntimeError(f"Failed to modify pyvenv.cfg: {e}") from e


def write_venv_pth(venv_site: str) -> None:
    """
    Write a `.pth` file so the virtualenv can always import project packages.

    This keeps imports working for commands like `imodelzoo_test/bin/python3 ptq.py`
    even after the current process has restored its original environment.

    Args:
        venv_site (str): Site-packages path of the virtual environment

    Raises:
        RuntimeError: If writing the `.pth` file fails
    """
    repo_root = get_imodelzoo_root()
    xh_model_zoo_root = get_xh_model_zoo_root()
    extra_paths = [
        str(repo_root),
        str(xh_model_zoo_root),
    ]

    missing_paths = [path for path in extra_paths if not Path(path).exists()]
    if missing_paths:
        raise RuntimeError(
            f"Failed to configure virtualenv import paths, missing: {missing_paths}"
        )

    pth_path = Path(venv_site) / "imodelzoo_paths.pth"

    try:
        with open(pth_path, "w", encoding="utf-8") as f:
            f.write("\n".join(extra_paths) + "\n")
    except IOError as e:
        raise RuntimeError(f"Failed to write virtualenv .pth file: {e}") from e


def create_distutils_symlink(dir_path: str) -> None:
    """
    Check and create distutils symbolic link.

    Args:
        dir_path (str): Path to the virtual environment directory

    Raises:
        RuntimeError: If creating the distutils symbolic link fails
    """
    # Define target directory and source symbolic link path
    distutils_dir = (
        Path(dir_path) / "lib" / "python3.12" / "site-packages" / "distutils"
    )
    symlink_source = (
        "/opt/venv/houmo/lib/python3.12/site-packages/setuptools/_distutils"
    )

    # Check if directory exists, create symlink if it doesn't
    if not distutils_dir.exists():
        try:
            # Create symlink (os.symlink works on Linux)
            os.symlink(symlink_source, str(distutils_dir))
            logger.info(
                f"Successfully created symlink: {distutils_dir} -> {symlink_source}"
            )
        except OSError as e:
            raise RuntimeError(f"Failed to create distutils symlink: {e}") from e
    else:
        logger.warning(
            f"Distutils directory already exists, no need to create symlink: {distutils_dir}"
        )


def create_virtualenv(python_exe: str, site_packages: str, dir_path: str) -> None:
    """
    Create a virtual environment with processing based on Python path.

    Args:
        python_exe (str): Path to the Python executable
        site_packages (str): Path to the original site-packages directory
        dir_path (str): Directory path where the virtual environment will be created

    Raises:
        subprocess.CalledProcessError: If virtual environment creation fails
    """
    dir_path_obj = Path(dir_path)
    dir_path_obj.mkdir(parents=True, exist_ok=True)

    # Check if Python interpreter is in /opt/venv path
    if "/opt/venv" in python_exe:
        # Create virtual environment with extra-search-dir
        subprocess.run(
            [
                "virtualenv",
                f"--python={python_exe}",
                f"--extra-search-dir={site_packages}",
                dir_path,
            ],
            check=True,
            text=True,
            stdout=sys.stdout,
            stderr=sys.stderr,
        )

        # Construct Python path and site-packages path within the virtual environment
        venv_python = str(dir_path_obj / "bin" / "python3")
        venv_site = get_site_packages(venv_python)

        # Persist extra import roots inside the virtualenv itself
        write_venv_pth(venv_site)

        # Modify activation/deactivation scripts
        activate_path = str(dir_path_obj / "bin" / "activate")
        deactivate_path = str(dir_path_obj / "bin" / "deactivate")
        modify_activate_script(activate_path, venv_site, site_packages)
        modify_deactivate_script(deactivate_path)

        # Modify pyvenv.cfg
        pyvenv_cfg_path = str(dir_path_obj / "pyvenv.cfg")
        modify_pyvenv_cfg(pyvenv_cfg_path)

        create_distutils_symlink(dir_path)
    else:
        # Create virtual environment with --system-site-packages
        subprocess.run(
            [
                "virtualenv",
                f"--python={python_exe}",
                "--system-site-packages",
                dir_path,
            ],
            check=True,
            text=True,
            stdout=sys.stdout,
            stderr=sys.stderr,
        )

        venv_python = str(dir_path_obj / "bin" / "python3")
        venv_site = get_site_packages(venv_python)
        write_venv_pth(venv_site)


def install_requirements(
    dir_path: str, requirements_path: str = "requirements.txt"
) -> None:
    """
    Activate virtual environment and install dependencies from requirements file.

    Args:
        dir_path (str): Path to the virtual environment directory
        requirements_path (str): Path to the requirements file (default: "requirements.txt")

    Raises:
        RuntimeError: If installing dependencies fails
    """
    venv_pip = str(Path(dir_path) / "bin" / "pip3")
    try:
        # Directly use pip from within the virtual environment, no need to manually source activate
        subprocess.run(
            [venv_pip, "install", "-r", requirements_path],
            check=True,
            text=True,
            stdout=sys.stdout,
            stderr=sys.stderr,
        )
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"Failed to install dependencies: {e.output}") from e


def activate_virtualenv(venv_dir: str, python_exe: str, site_packages: str) -> None:
    """
    Apply virtualenv activation effects to the current Python process.

    This mirrors the relevant behavior of `source <venv>/bin/activate` used in
    the shell script so that later subprocess calls that rely on `python3` or
    `pip3` see the virtual environment first in `PATH`.

    Args:
        venv_dir (str): Path to the virtual environment directory
        python_exe (str): Path to the original Python executable
        site_packages (str): Path to the original site-packages directory
    """
    venv_path = Path(venv_dir).resolve()
    venv_bin = str(venv_path / "bin")
    venv_python = str(venv_path / "bin" / "python3")

    os.environ["VIRTUAL_ENV"] = str(venv_path)
    os.environ.pop("PYTHONHOME", None)

    current_path = os.environ.get("PATH", "")
    path_entries = [entry for entry in current_path.split(":") if entry]
    if not path_entries or path_entries[0] != venv_bin:
        path_entries = [entry for entry in path_entries if entry != venv_bin]
        os.environ["PATH"] = ":".join([venv_bin, *path_entries])

    if "/opt/venv" in python_exe:
        venv_site = get_site_packages(venv_python)
        original_pythonpath = os.environ.get("PYTHONPATH", "")
        os.environ["ORIGINAL_PYTHONPATH"] = original_pythonpath
        os.environ["PYTHONPATH"] = _join_pythonpath(
            venv_site, site_packages, original_pythonpath
        )


def deactivate_virtualenv(
    original_cwd: str, original_env: dict[str, str | None]
) -> None:
    """
    Restore process state after virtualenv activation.

    Args:
        original_cwd (str): Working directory before activation
        original_env (dict[str, str | None]): Environment values to restore
    """
    os.chdir(original_cwd)

    for env_name, env_value in original_env.items():
        if env_value is None:
            os.environ.pop(env_name, None)
        else:
            os.environ[env_name] = env_value


def install_py_venv(env_dir: str, log_file: str, flow_type: str = "default") -> bool:
    """
    Main function to orchestrate complete virtual environment setup process.

    Args:
        env_dir (str): Directory where the virtual environment will be created
        log_file (str): Path to the log file
        flow_type (str): Type of workflow ('default' or 'quant') which determines
                        which requirements file to use

    Returns:
        bool: True if successful, False if no valid requirements file is found
    """
    rqmt_name = "requirements.txt"
    if flow_type == "quant":
        tmp_rqmt_path = os.path.join(env_dir, "requirements_ptq.txt")
        if os.path.exists(tmp_rqmt_path) and os.path.isfile(tmp_rqmt_path):
            rqmt_name = "requirements_ptq.txt"

    rqmt_path = os.path.join(env_dir, rqmt_name)
    if not os.path.exists(rqmt_path) or not os.path.isfile(rqmt_path):
        return False

    # 1. Get Python path and site-packages
    py_exe = get_python_executable()
    site_packages = get_site_packages(py_exe)
    venv_dir = os.path.join(env_dir, VENV_NAME)
    original_cwd = os.getcwd()
    original_env = {
        "PATH": os.environ.get("PATH"),
        "PYTHONPATH": os.environ.get("PYTHONPATH"),
        "ORIGINAL_PYTHONPATH": os.environ.get("ORIGINAL_PYTHONPATH"),
        "PYTHONHOME": os.environ.get("PYTHONHOME"),
        "VIRTUAL_ENV": os.environ.get("VIRTUAL_ENV"),
    }

    try:
        os.chdir(env_dir)

        # 2. Create and activate virtual environment
        create_virtualenv(py_exe, site_packages, venv_dir)
        activate_virtualenv(venv_dir, py_exe, site_packages)

        # 3. Install dependencies
        install_requirements(venv_dir, rqmt_path)
        logger.info(f"Virtual environment created successfully: {venv_dir}")

        return True
    finally:
        deactivate_virtualenv(original_cwd, original_env)
