import os
import glob
import argparse
import datetime
import pandas as pd
from lxml import etree


def parse_args():
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
    rows = list()
    parse_res = dict()

    # 解析为树形结构
    tree = etree.parse(xml_path)
    root = tree.getroot()  # 获取根节点

    testsuite = root.xpath("//testsuite")
    test_num = int(testsuite[0].get("tests"))
    skipped_num = int(testsuite[0].get("skipped"))
    failure_num = int(testsuite[0].get("failures"))
    error_num = int(testsuite[0].get("errors"))
    passed_num = test_num - skipped_num - failure_num - error_num

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

    passed_cases, skipped_cases, failure_cases, error_cases = (
        list(),
        list(),
        list(),
        list(),
    )

    # 查找所有 testcase 节点
    testcases = root.xpath("//testcase")
    for testcase in testcases:
        classname = testcase.get("classname")
        name = testcase.get("name")
        status = "unknown"
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
        rows.append(
            {
                'test_type': classname,
                'test_name': name,
                f'{key_str}_status': status,
            }
        )

    parse_res["skipped_info"]["skipped_cases"] = skipped_cases
    parse_res["failure_info"]["failure_cases"] = failure_cases
    parse_res["error_info"]["error_cases"] = error_cases
    parse_res["passed_info"]["passed_cases"] = passed_cases

    return rows


def _get_final_status(row):
    no_infer = row['no_infer_status']
    infer = row['infer_status']

    # 规则1：两者均为skipped
    if (no_infer == 'skipped' or pd.isna(no_infer)) and infer == 'skipped':
        return 'skipped'
    # 规则2：任意一个为failure
    elif no_infer == 'failure' or infer == 'failure':
        return 'failure'
    # 规则3：任意一个为error
    elif no_infer == 'error' or infer == 'error':
        return 'error'
    # 规则4：no_infer为skipped且infer为passed
    elif no_infer == 'passed' or infer == 'passed':
        return 'passed'
    # 其他情况
    else:
        return 'unknown'


def _get_xml_files(folder_path: str, xml_type: str) -> list:
    xml_files = list()
    if not os.path.isdir(folder_path):
        return xml_files

    pattern = os.path.join(folder_path, f"pytest_results_{xml_type}_*.xml")
    xml_files = glob.glob(pattern)
    return xml_files


if __name__ == "__main__":
    args = parse_args()
    target = args.target
    version = args.version
    image_id = args.image_id

    script_dir = os.path.dirname(os.path.abspath(__file__))
    xml_infer_list = _get_xml_files(script_dir, "infer")
    xml_no_infer_list = _get_xml_files(script_dir, "no_infer")
    if len(xml_infer_list) == 0 and len(xml_no_infer_list) == 0:
        print("*" * 10, "CD TEST RESULT: FAILED (NO RESULTS)", "*" * 10)
        exit(1)

    infer_df = pd.DataFrame()
    for xml_path in xml_infer_list:
        if not os.path.exists(xml_path):
            continue
        print("xml path:", xml_path)
        res = _parse_xml_res(xml_path, "infer")
        res_df = pd.DataFrame(res)
        infer_df = pd.concat([infer_df, res_df], ignore_index=True)

    no_infer_df = pd.DataFrame()
    for xml_path in xml_no_infer_list:
        if not os.path.exists(xml_path):
            continue
        print("xml path:", xml_path)
        res = _parse_xml_res(xml_path, "no_infer")
        res_df = pd.DataFrame(res)
        no_infer_df = pd.concat([no_infer_df, res_df], ignore_index=True)

    if not infer_df.empty and not no_infer_df.empty:
        final_df = pd.merge(
            infer_df, no_infer_df, on=['test_type', 'test_name'], how='outer'
        )
        print(final_df.columns)
        final_df['status'] = final_df.apply(_get_final_status, axis=1)
    elif not infer_df.empty:
        final_df = infer_df
        final_df['status'] = final_df['infer_status']
    elif not no_infer_df.empty:
        final_df = no_infer_df
        final_df['status'] = final_df['no_infer_status']
    else:
        print("*" * 10, "CD TEST RESULT: FAILED (EXTRACT RESULTS)", "*" * 10)
        exit(1)

    failed_df = final_df.query('status=="failure" | status=="error"')
    if not failed_df.empty:
        print("*" * 10, "CD TEST RESULT: FAILED", "*" * 10)
        print(failed_df)
    else:
        print("*" * 10, "CD TEST RESULT: SUCCESS", "*" * 10)
        # logger.info(final_df.query('status=="passed"'))
    current_dt = datetime.date.today().strftime("%Y%m%d")
    final_df.to_csv(
        f"{script_dir}/[CD_Tester]{target}_v{version}_{image_id}_results_{current_dt}.csv",
        index=False,
    )
