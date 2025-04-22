#!/usr/bin/env python

import os
import sys
import numpy as np
import argparse
import importlib
import logging
import time
from hmassist.utils import check
from hmassist.utils import logger
from hmassist.utils.glog_format import GLogFormatter
from hmassist.utils.parser import read_yaml_to_dict, dump_yaml
from hmassist.utils.dist_metrics import cosine_distance, euclid_distance
from hmassist.utils.utils import get_random_data
from hmassist.executors.xh1_exec import XH1Exec
from hmassist.executors.onnx_exec import OnnxExec
from hmassist.models.base_model import BaseModel
from hmassist.utils.utils import sanitize_name


HOUMO_TARGET_LIST=["houmo", "xh1"]


def save_data(data, dir, name):
    if not os.path.exists(dir):
        os.makedirs(dir)
    data.tofile(os.path.join(dir, f"{name}.bin"))
    data.tofile(os.path.join(dir, f"{name}.txt"), sep="\n")
    np.save(os.path.join(dir, f'{name}.npy'), data)


def get_executor(cfg):
    target = cfg["target"]
    if target == "onnx":
        backend = OnnxExec(cfg)
    elif target in ["houmo", "xh1"]:
        backend = XH1Exec(cfg)
    else:
        logger.error(f"Not support target -> {target}")
        exit(-1)
    backend.houmo_target_list = HOUMO_TARGET_LIST
    return backend


def get_model(cfg):
    model_impl_class = cfg["model"].get("impl_class", None)
    executor = get_executor(cfg)

    try:
        m = importlib.import_module("model")
        if hasattr(m, model_impl_class):
            # 实例化预处理对象
            model = getattr(m, model_impl_class)(
                executor=executor,
                dataset=None,
                # dtype=dtype,   # int8/fp32
            )
        else:
            logger.error(f"model.py has no class named {model_impl_class}, please check your config")
            exit(-1)
        del sys.modules["model"]
    except Exception as e:
        logger.warning(f"can not load impl class: {model_impl_class}, use default model will not support demo/eval: {e}")
        model = BaseModel(executor=executor, dataset=None)

    return model


def quantize(cfg):
    if not check.check_quant_config(cfg):
        exit(-1)
    model = get_model(cfg)
    model.executor.quantize(model.get_input_datas)
    del model


def build(cfg):
    if not check.check_build_config(cfg):
        exit(-1)
    model = get_model(cfg)
    model.executor.build()

    # compare golden data
    model.load()
    model.executor.print_input_info()
    model.executor.print_output_info()
    logger.info("start compare golden data...")
    if not os.path.exists(model.executor.build_dir):
        os.makedirs(model.executor.build_dir)
    inputs = model.executor.get_golden_inputs()
    if inputs is not None:
        # save input data for result check in tcim_perf
        for input_name, input_data in inputs.items():
            input_save_name = sanitize_name(input_name)
            input_data.tofile(os.path.join(model.executor.build_dir, f"{input_save_name}.bin"))
        model.executor.set_fixed_out(True)
        start = time.time()
        outputs = model.executor.infer(inputs)
        cost = time.time() - start
        model.executor.set_fixed_out(False)
        dequanted_outputs = model.executor.infer(inputs)
        logger.info(f"[infer] cost {cost * 1000:.3f} ms")
    logger.info(f"inputs saved in {model.executor.build_dir}")

    result_check = True
    for output_name, output_data in outputs.items():
        logger.info(f"{model.target} output[{output_name}] shape = {output_data.shape}, dtype = {output_data.dtype}")
        # save output data for result check in tcim_perf
        output_save_name = sanitize_name(output_name)
        save_data(output_data, model.executor.build_dir, output_save_name)
            
        golden_output_path = os.path.join(model.executor.golden_data_path, f'hmquant_{model.executor.model_name}_{output_save_name}_output.npy')
        if os.path.exists(golden_output_path):
            golden_output = np.load(golden_output_path)
            golden_output = np.concatenate([golden_output for i in range(model.executor.batch)], axis=0)
            logger.info(f"golden output[{output_name}] shape = {golden_output.shape}, dtype = {golden_output.dtype}")
            cosine_dist1 = cosine_distance(golden_output, output_data)
            is_match1 = (golden_output == output_data).all()
            logger.info(f"[compare] golden output [{output_name}] match={is_match1}, similarity={cosine_dist1:.6f}")
        else:
            logger.warning(f"compare canceled while golden output not found -> {golden_output_path}")
        dequanted_output_path = os.path.join(model.executor.golden_data_path, f'hmquant_{model.executor.model_name}_{output_save_name}_dequant_output.npy')
        if os.path.exists(dequanted_output_path):
            golden_dequant_output = np.load(dequanted_output_path)
            golden_dequant_output = np.concatenate([golden_dequant_output for i in range(model.executor.batch)], axis=0)
            logger.info(f"golden dequant output[{output_name}] shape = {golden_dequant_output.shape}, dtype = {golden_dequant_output.dtype}")
            cosine_dist2 = cosine_distance(golden_dequant_output, dequanted_outputs[output_name])
            is_match2 = (golden_dequant_output == dequanted_outputs[output_name]).all()
            logger.info(f"[compare] dequanted golden output [{output_name}] match={is_match2}, similarity={cosine_dist2:.6f}")
        else:
            logger.warning(f"dequanted compare canceled while golden output not found -> {golden_dequant_output}")
        
        if is_match1 and is_match2:
            continue
        if cosine_dist1 < 0.999 or cosine_dist2 < 0.999:
            result_check &= False
    logger.info(f"outputs saved in {model.executor.build_dir}")
    if not result_check:
        logger.error("result check failed.")
        exit(-1)
    logger.info(f"{model.executor.model_name} build completed.")
    del model


def test(cfg):
    if not check.check_test_config(cfg):
        exit(-1)
    model = get_model(cfg)
    model.load()
    model.executor.print_input_info()
    model.executor.print_output_info()
    data_path = cfg["test"].get("data_path")
    if data_path:
        dir, file = os.path.split(data_path)
        inputs = model.get_input_datas(dir, file)
    else:
        inputs = {}
        for _input in model.inputs:
            name = _input["name"]
            dtype = _input["dtype"]
            logger.warning(f"data[{name}] will use random data")
            inputs[name] = get_random_data(name, dtype, model.input_shape)
    inputs = model.executor._preprocess(inputs)
    model.executor.set_fixed_out(False)
    start = time.time()
    outputs = model.executor.infer(inputs)
    cost = time.time() - start
    logger.info(f"[infer] cost {cost * 1000:.3f} ms")

    # save datas
    save_dir = os.path.join(model.executor.test_dir)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    for input_name, input_data in inputs.items():
        save_data(input_data, save_dir, input_name + "_input")
    for output_name, output_data in outputs.items():
        logger.info(f"{model.target} output[{output_name}] shape = {output_data.shape}, dtype = {output_data.dtype}")
        output_save_name = sanitize_name(output_name)
        save_data(output_data, save_dir, output_save_name)
    logger.info(f"test outputs saved in {save_dir}")

    if model.target in HOUMO_TARGET_LIST:
        # compare to quant output and framework output
        for output_name, output_data in outputs.items():
            output_save_name = sanitize_name(output_name)
            hmquant_data_path = os.path.join(model.executor.golden_data_path, f'hmquant_{model.executor.model_name}_{output_save_name}_dequant_output.npy')
            raw_data_path = os.path.join(model.compare_dir, "test", output_save_name + '.npy')
            if os.path.exists(hmquant_data_path):
                hmquant_data = np.load(hmquant_data_path)
                logger.info(f"hmquant output[{output_name}] shape = {hmquant_data.shape}, dtype = {hmquant_data.dtype}")
            else:
                hmquant_data = None
                logger.warning(f"compare canceled while hmquant output not found -> {hmquant_data_path}")
            if os.path.exists(raw_data_path):
                raw_data = np.load(raw_data_path)
                logger.info(f"{model.framework} output[{output_name}] shape = {raw_data.shape}, dtype = {raw_data.dtype}")
            else:
                raw_data = None
                logger.warning(f"compare canceled while {model.framework} output not found -> {raw_data_path}")
            
            if hmquant_data is not None:
                if raw_data is not None:
                    cosine_dist = cosine_distance(hmquant_data, raw_data)
                    euclid_dist = euclid_distance(hmquant_data, raw_data)
                    logger.info(f"[compare] output [{output_name}] {model.framework} vs hmquant similarity={cosine_dist:.6f}, euclid_dist={euclid_dist:.6f}")
                cosine_dist = cosine_distance(output_data, hmquant_data)
                euclid_dist = euclid_distance(output_data, hmquant_data)
                logger.info(f"[compare] output [{output_name}] hmquant vs {model.target} similarity={cosine_dist:.6f}, euclid_dist={euclid_dist:.6f}")
            if raw_data is not None:
                cosine_dist = cosine_distance(output_data, raw_data)
                euclid_dist = euclid_distance(output_data, raw_data)
                logger.info(f"[compare] output [{output_name}] {model.framework} vs {model.target} similarity={cosine_dist:.6f}, euclid_dist={euclid_dist:.6f}")
            
    logger.info("test completed")
    del model


def demo(cfg):
    logger.info(cfg)
    if not check.check_demo_config(cfg):
        exit(-1)
    model = get_model(cfg)
    data_dir = cfg["demo"]["data_dir"]
    test_num = cfg["demo"]["test_num"]
    # if not os.environ.get("HDPL_PLATFORM") == "ASIC":
    #     if test_num > 10 or test_num == 0:
    #         test_num = 10
    #         logger.warning("test num set to 10 because HDPL_PLATFORM=ISIM may take a lot of time.")

    file_list = []
    if os.path.isfile(data_dir):
        _, ext = os.path.splitext(data_dir)
        if ext in [".jpg", ".JPEG", ".bmp", ".png", ".jpeg", ".BMP"]:
            file_list.append(data_dir)
        else:
            logger.error(f"file type not support -> {data_dir}")
    elif os.path.isdir(data_dir):
        for filename in os.listdir(data_dir):
            _, ext = os.path.splitext(filename)
            if ext in [".jpg", ".JPEG", ".bmp", ".png", ".jpeg", ".BMP"]:
                file_list.append(os.path.join(data_dir, filename))
                if len(file_list) == test_num:
                    break
    file_list.sort()
    model.load()

    for filepath in file_list:
        model.demo(filepath)
    logger.info(f"[infer] average cost {model.ave_latency_ms:.3f} ms")
    logger.info(f"[end2end] average cost: {model.end2end_latency_ms:.3f} ms")
    logger.info("demo completed")
    del model


def perf(cfg):
    if not check.check_perf_config(cfg):
        exit(-1)
    model = get_model(cfg)
    test_num = cfg["perf"]["test_num"]
    model.executor.perf(test_num)
    logger.info("perf test completed")
    del model


def eval(cfg):
    if not check.check_eval_config(cfg):
        exit(-1)
    model = get_model(cfg)
    dataset_class = cfg["eval"].get("dataset_class", None)
    data_dir = cfg["eval"].get("data_dir")
    model.test_num = cfg["eval"].get("test_num", 0)
    try:
        m = importlib.import_module("dataset")
        if hasattr(m, dataset_class):
            # 实例化预处理对象
            dataset = getattr(m, dataset_class)(data_dir, test_num=model.test_num)
        else:
            logger.error(f"dataset.py has no class named {dataset_class}, please check your config")
            exit(-1)
        del sys.modules["dataset"]
    except Exception as e:
        logger.error(f"can not find dataset.py, use default model will not support eval: {e}")
        return -1

    # if not os.environ.get("HDPL_PLATFORM") == "ASIC":
    #     if model.test_num > 10 or model.test_num == 0:
    #         model.test_num = 10
    #         logger.warning("test num set to 10 because HDPL_PLATFORM=ISIM may take a lot of time.")
    model.dataset = dataset
    model.load()

    res = model.evaluate()
    logger.info(f"[infer] average cost {model.ave_latency_ms:.3f} ms")
    logger.info(f"[end2end] average cost: {model.end2end_latency_ms:.3f} ms")
    logger.info(f"{res}")
    logger.info("eval test completed")

    with open('output/hmeval.txt', 'w') as file:
        file.write(f"{res}\n")
    del model
    return res


def run(args):
    # 补充自定义预处理文件所在目录，必须与配置文件同目录
    config_abspath = os.path.abspath(args.config)
    config_dir = os.path.dirname(config_abspath)
    sys.path.insert(0, config_dir)  # 自定义模块环境变量

    config = read_yaml_to_dict(config_abspath)
    config["target"] = args.target
    config['batch'] = args.batch
    config['thread_num'] = args.thread_num
    config['core_num'] = args.core_num
    
    # import pprint
    # pprint.pprint(config.dump())
    logger.info(f"config:\n{dump_yaml(config)}")

    if args.type == "quant":
        quantize(config)
    elif args.type == "build":
        build(config)
    elif args.type == "test":
        test(config)
    elif args.type == "demo":
        if args.test_num != -1:
            config['demo']['test_num'] = args.test_num
        demo(config)
    elif args.type == "perf":
        if args.test_num != -1:
            config['perf']['test_num'] = args.test_num
        config['perf']['infer_only'] = args.infer_only
        perf(config)
    elif args.type == "eval":
        if args.test_num != -1:
            config['eval']['test_num'] = args.test_num
        eval(config)
    elif args.type == "benchmark":
        benchmark(config)
    else:
        logger.error(f"Not support operation -> {args.type}")

    sys.path.remove(config_dir)


def benchmark(config):
    import csv
    from prettytable import PrettyTable

    header = ["ModelName", "Shape", "Dataset", "CoreNum", "Batch", "ThreadNum", "Accuracy(onnx)",
              f"Accuracy({config['target']})", "AccRelError", "Latency(ms)", "Throughput"]
    table = PrettyTable(header)
    t = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
    if not os.path.exists("reports"):
        os.mkdir("reports")
    report_file = os.path.abspath(f"reports/benchmark_{t}.csv")
    with open(report_file, "w") as f:
        f_csv = csv.writer(f)
        f_csv.writerow(header)

    root = os.getcwd()
    logger.info(config["models"])
    for model_name in config["models"]:
        batch = 1
        core_num = 1
        thread_num = 1
        if "batch" in config["models"][model_name]:
            batch = config["models"][model_name]["batch"]
        if "core_num" in config["models"][model_name]:
            core_num = config["models"][model_name]["core_num"]
        if "thread_num" in config["models"][model_name]:
            thread_num = config["models"][model_name]["thread_num"]
        perf_result = {}
        eval_result = {}
        onnx_result = {}
        os.chdir(config["models"][model_name]["location"])
        # get model
        os.system("python3 get_model.py")
        # quant
        if "quant" not in config["models"][model_name] or config["models"][model_name]["quant"]:
            os.system("hmquant.sh")
        # build
        os.system(f"hmbuild.sh --core_num {core_num} --batch {batch}")
        # perf
        if os.path.exists("output/hmperf.txt"):
            os.remove("output/hmperf.txt")
        os.system(f"hmperf.sh --batch {batch} --thread_num {thread_num} --infer_only")
        if os.path.exists("output/hmperf.txt"):
            perf_result = read_yaml_to_dict("output/hmperf.txt")
        # eval
        if "eval" not in config["models"][model_name] or config["models"][model_name]["eval"]:
            if os.path.exists("output/hmeval.txt"):
                os.remove("output/hmeval.txt")
            os.system("hmeval.sh")
            if os.path.exists("output/hmeval.txt"):
                eval_result = read_yaml_to_dict("output/hmeval.txt")
        if "onnx" not in config["models"][model_name] or config["models"][model_name]["onnx"]:
            if os.path.exists("output/hmeval.txt"):
                os.remove("output/hmeval.txt")
            os.system("hmeval.sh --target onnx")
            if os.path.exists("output/hmeval.txt"):
                onnx_result = read_yaml_to_dict("output/hmeval.txt")

        if "shape" in perf_result:
            shapes = perf_result["shape"]
        else:
            shapes = "NotTest"
        if "avg_latency" in perf_result:
            avg_latency = perf_result["avg_latency"]
        else:
            avg_latency = "NotTest"
        if "qps" in perf_result:
            throughput = perf_result["qps"]
        else:
            throughput = "NotTest"
        
        acc_result_onnx = ""
        acc_result_hdpl = ""
        acc_result_err = ""
        dataset = ""
        if "accuracy" in onnx_result:
            last = list(onnx_result["accuracy"])[-1]
            for acc in onnx_result["accuracy"]:
                acc_result_onnx += f"{acc}: {onnx_result['accuracy'][acc]:.3f}"
                if acc != last:
                    acc_result_onnx += "\n"
        else:
            acc_result_onnx = "NotTest"
        if "dataset" in onnx_result:
            dataset = onnx_result["dataset"]
        else:
            dataset = "NotTest"
        if "accuracy" in eval_result:
            last = list(eval_result["accuracy"])[-1]
            for acc in eval_result["accuracy"]:
                acc_result_hdpl += f"{acc}: {eval_result['accuracy'][acc]:.3f}"
                if acc != last:
                    acc_result_hdpl += "\n"
                if "accuracy" in onnx_result and onnx_result["accuracy"][acc] != 0:
                    acc_err = eval_result["accuracy"][acc] / onnx_result["accuracy"][acc] - 1
                    acc_result_err += f"{acc}: {acc_err:.3f}"
                else:
                    acc_result_err = "NotTest"
                if acc != last:
                    acc_result_err += "\n"
        else:
            acc_result_hdpl = "NotTest"
            acc_result_err = "NotTest"

        row = [model_name, shapes, dataset, core_num, batch, thread_num,
               acc_result_onnx, acc_result_hdpl, acc_result_err,
               f"{avg_latency:.3f}",
               f"{throughput:.2f}"]
        table.add_row(row)
        with open(report_file, "a") as f:
            f_csv = csv.writer(f)
            f_csv.writerow(row)
        logger.info(f"<=== Benchmark {model_name} completed.")
        os.chdir(root)
    logger.info(f"\n{table}")
    logger.info("benchmark completed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="HmAssist Tool")
    parser.add_argument("type", type=str,
                        choices=("quant", "build", "test", "demo", "perf", "eval", "benchmark"),
                        help="Specify an operation")
    parser.add_argument("--target", type=str, required=True,
                        choices=(HOUMO_TARGET_LIST + ["onnx"]),
                        help="Specify a chip target")
    parser.add_argument("--config", type=str, default="config.yml",
                        help="Specify a config file, default is config.yml")
    parser.add_argument("--batch", type=int, default=1,
                        help="Specify batch size in build, default is 1")
    parser.add_argument("--core_num", type=int, default=1,
                        help="Specify core number in build, default is 1")
    parser.add_argument("--thread_num", type=int, default=1,
                        help="Specify thread number in perf, default is 1")
    parser.add_argument("--test_num", type=int, default=-1,
                        help="Specify the test number in demo, default is the config in the config file")
    parser.add_argument("--infer_only", action='store_true', default=False,
                        help="Specify if only test infer while perfing, default is False")

    args = parser.parse_args()
    logger.info(args)
    if not check.check_file(args.config):
        exit(-1)

    # TODO: get version
    # VERSION = "v2.0.0"
    # logger.info("{} with HmAssist version: {}".format(args.type, VERSION))

    run(args)
