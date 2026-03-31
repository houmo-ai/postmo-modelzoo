from pathlib import Path
from typing import List

from setuptools import find_packages, setup


ROOT = Path(__file__).resolve().parent


def read_requirements(path: Path) -> List[str]:
    if not path.exists():
        return []
    lines = [line.strip() for line in path.read_text(encoding="utf-8").splitlines()]
    return [
        line
        for line in lines
        if line and not line.startswith("#") and not line.startswith("-r")
    ]


long_description = ""
readme_file = ROOT / "README.md"
if readme_file.exists():
    long_description = readme_file.read_text(encoding="utf-8")


setup(
    name="hmeval",
    version="0.1.0",
    description="Command line tool for large model evaluation",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Houmo AI",
    license="Proprietary",
    keywords=["llm", "evaluation", "cli", "benchmark"],
    python_requires=">=3.8",
    packages=find_packages(exclude=("tests", "tests.*")),
    include_package_data=True,
    # Keep base install light; install evaluation dependencies via extras.
    install_requires=[],
    extras_require={
        "eval": read_requirements(ROOT / "requirements.txt"),
    },
    entry_points={
        "console_scripts": [
            "hmeval=hmeval.cli:main",
        ]
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: Other/Proprietary License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3 :: Only",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Environment :: Console",
        "Operating System :: POSIX :: Linux",
    ],
)
