# Copyright 2025 HOUMO AI
#
# File: generate_test_report.py
# Description:
#   Generate a comprehensive test report from XML test results.
#   This script parses pytest XML output files from both inference and
#   non-inference tests, merges the results, and generates a final report
#   in CSV format with status determination based on specific rules.
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
import glob
import argparse
import datetime
import pandas as pd
from lxml import etree


def parse_args() -> argparse.Namespace:
    """Parse command line arguments for the test report generation script."""
    parser = argparse.ArgumentParser(description="CD Tester")
    parser.add_argument(
        "-v",
        "--version",
        required=True,
        type=str,
        help="Houmo Dadao software version, example: 0.3.1, 2.4.2",
    )
    parser.add_argument(
        "-t",
        "--target",
        type=str,
        default="xh2",
        help="Houmo backend, support: xh1, xh2.",
    )
    parser.add_argument(
        "-id",
        "--image_id",
        type=str,
        required=True,
        help="Docker image id.",
    )
    parser.add_argument(
        "--release",
        type=str,
        default="off",
        help="use release models for testing, support: on, off.",
    )

    args = parser.parse_args()
    return args


def _parse_xml_res(xml_path: str, key_str: str) -> list:
    """Parse XML test results and extract test case information.

    Args:
        xml_path (str): Path to the XML file to parse
        key_str (str): String to identify the test type (e.g. 'infer', 'no_infer')

    Returns:
        list: List of dictionaries containing test case information
    """
    rows = list()
    parse_res = dict()

    # Parse the XML file into a tree structure
    tree = etree.parse(xml_path)
    root = tree.getroot()  # Get the root node

    # Find the testsuite element to get overall test statistics
    testsuite = root.xpath("//testsuite")
    test_num = int(testsuite[0].get("tests"))
    skipped_num = int(testsuite[0].get("skipped"))
    failure_num = int(testsuite[0].get("failures"))
    error_num = int(testsuite[0].get("errors"))
    passed_num = test_num - skipped_num - failure_num - error_num

    # Store test statistics in the result dictionary
    parse_res["test_time"] = testsuite[0].get("time")
    parse_res["test_num"] = test_num
    parse_res["passed_info"] = dict()
    parse_res["skipped_info"] = dict()
    parse_res["failure_info"] = dict()
    parse_res["error_info"] = dict()
    parse_res["skipped_info"]["skipped_num"] = skipped_num
    parse_res["failure_info"]["failure_num"] = failure_num
    parse_res["error_info"]["error_num"] = error_num
    parse_res["passed_info"]["passed_num"] = passed_num

    # Initialize lists to store different types of test cases
    passed_cases, skipped_cases, failure_cases, error_cases = (
        list(),
        list(),
        list(),
        list(),
    )

    # Find all testcase nodes in the XML
    testcases = root.xpath("//testcase")
    for testcase in testcases:
        # Extract test case information
        classname = testcase.get("classname")
        name = testcase.get("name")
        status = "unknown"

        # Determine the status based on child elements
        if testcase.find("skipped") is not None:
            skipped_cases.append(name)
            status = "skipped"
        elif testcase.find("failure") is not None:
            failure_cases.append(name)
            status = "failure"
        elif testcase.find("error") is not None:
            error_cases.append(name)
            status = "error"
        else:
            passed_cases.append(name)
            status = "passed"

        # Add the test case information to rows
        rows.append(
            {
                "test_type": classname,
                "test_name": name,
                f"{key_str}_status": status,
            }
        )

    # Store the categorized test cases in the result dictionary
    parse_res["skipped_info"]["skipped_cases"] = skipped_cases
    parse_res["failure_info"]["failure_cases"] = failure_cases
    parse_res["error_info"]["error_cases"] = error_cases
    parse_res["passed_info"]["passed_cases"] = passed_cases

    return rows


def _get_final_status(row):
    """Determine the final test status based on inference and non-inference results.

    Args:
        row: A pandas DataFrame row containing 'no_infer_status' and 'infer_status'

    Returns:
        str: Final status ('skipped', 'failure', 'error', 'passed', or 'unknown')
    """
    no_infer = row["no_infer_status"]
    infer = row["infer_status"]

    # Rule 1: Both are skipped
    if (no_infer == "skipped" or pd.isna(no_infer)) and infer == "skipped":
        return "skipped"
    # Rule 2: Either has failure
    elif no_infer == "failure" or infer == "failure":
        return "failure"
    # Rule 3: Either has error
    elif no_infer == "error" or infer == "error":
        return "error"
    # Rule 4: no_infer is skipped and infer is passed
    elif no_infer == "passed" or infer == "passed":
        return "passed"
    # Other cases
    else:
        return "unknown"


def _get_xml_files(folder_path: str, xml_type: str) -> list:
    """Find XML files matching a specific pattern in the given folder.

    Args:
        folder_path (str): Directory to search for XML files
        xml_type (str): Type of XML files to find ('infer' or 'no_infer')

    Returns:
        list: List of paths to matching XML files
    """
    xml_files = list()
    if not os.path.isdir(folder_path):
        return xml_files

    # Create pattern for finding XML files
    pattern = os.path.join(folder_path, f"pytest_results_{xml_type}_*.xml")
    xml_files = glob.glob(pattern)
    return xml_files


if __name__ == "__main__":
    args = parse_args()
    target = args.target
    version = args.version
    image_id = args.image_id

    # Get the script directory to find XML files
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # Find XML files for both inference and non-inference tests
    xml_infer_list = _get_xml_files(script_dir, "infer")
    xml_no_infer_list = _get_xml_files(script_dir, "no_infer")
    # Exit if no XML files are found
    if len(xml_infer_list) == 0 and len(xml_no_infer_list) == 0:
        print("*" * 10, "CD TEST RESULT: FAILED (NO RESULTS)", "*" * 10)
        exit(1)

    # Process inference test results
    infer_df = pd.DataFrame()
    for xml_path in xml_infer_list:
        if not os.path.exists(xml_path):
            continue
        print("xml path:", xml_path)
        res = _parse_xml_res(xml_path, "infer")
        res_df = pd.DataFrame(res)
        infer_df = pd.concat([infer_df, res_df], ignore_index=True)

    # Process non-inference test results
    no_infer_df = pd.DataFrame()
    for xml_path in xml_no_infer_list:
        if not os.path.exists(xml_path):
            continue
        print("xml path:", xml_path)
        res = _parse_xml_res(xml_path, "no_infer")
        res_df = pd.DataFrame(res)
        no_infer_df = pd.concat([no_infer_df, res_df], ignore_index=True)

    # Merge results based on available data
    if not infer_df.empty and not no_infer_df.empty:
        final_df = pd.merge(
            infer_df, no_infer_df, on=["test_type", "test_name"], how="outer"
        )
        print(final_df.columns)
        final_df["status"] = final_df.apply(_get_final_status, axis=1)
    elif not infer_df.empty:
        final_df = infer_df
        final_df["status"] = final_df["infer_status"]
    elif not no_infer_df.empty:
        final_df = no_infer_df
        final_df["status"] = final_df["no_infer_status"]
    else:
        print("*" * 10, "CD TEST RESULT: FAILED (EXTRACT RESULTS)", "*" * 10)
        exit(1)

    # Check for failed tests and report accordingly
    failed_df = final_df.query('status=="failure" | status=="error"')
    if not failed_df.empty:
        print("*" * 10, "CD TEST RESULT: FAILED", "*" * 10)
        print(failed_df)
    else:
        print("*" * 10, "CD TEST RESULT: SUCCESS", "*" * 10)
        # logger.info(final_df.query('status=="passed"'))

    # Generate current date string for the filename
    current_dt = datetime.date.today().strftime("%Y%m%d")
    final_df.to_csv(
        f"{script_dir}/[CD_Tester]{target}_v{version}_{image_id}_results_{current_dt}.csv",
        index=False,
    )
