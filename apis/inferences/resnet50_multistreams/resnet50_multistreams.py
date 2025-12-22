import os
import sys
import threading
import queue
import numpy as np
import cv2
import argparse
from loguru import logger

import tcim_lite as tcim

HOUMO_TARGET = os.getenv("HOUMO_TARGET")
assert HOUMO_TARGET in ["xh2"], f"Unsupported HOUMO_TARGET: {HOUMO_TARGET}"


def get_args() -> argparse.Namespace:
    """Parse commandline."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-n',
        dest='device_num',
        type=int,
        default=1,
        help='device_num',
    )
    parser.add_argument(
        '-t',
        dest='thread_num',
        type=int,
        default=4,
        help='thread_num',
    )
    parser.add_argument(
        '-s',
        dest='sample_num',
        type=int,
        default=10,
        help='sample_num',
    )
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    sys.path.insert(0, "../../common/python")
    logger.info("===> resnet50_multistreams python example start...")
    logger.info(
        f"tcim runtime version: {tcim.runtime.get_version()}, houmo target: {HOUMO_TARGET}"
    )

    # set the parameters
    args = get_args()
    device_num = args.device_num
    thread_num = args.thread_num
    sample_num = args.sample_num
    model_path = "./resnet50_xh2_b1_1core.hmm"
    if not os.environ.get("HDPL_PLATFORM") == "ASIC":
        thread_num = 1
    logger.info(f"devices: {device_num}")
    logger.info(f"threads: {thread_num}")
    logger.info(f"samples: {sample_num}")
    logger.info(f"model: {model_path}")

    modules = []
    threads = []

    if not os.path.exists(model_path):
        logger.error("[error] could not find model: {}".format(model_path))
        exit(-1)

    # 1. input preprocess
    input_data = cv2.imread("../../data/snake.jpg")

    image_rgb = cv2.cvtColor(input_data, cv2.COLOR_BGR2RGB)
    image_rgb = cv2.resize(image_rgb, (224, 224))  # HWC uint8
    mean_arr = np.array([123.675, 116.28, 103.53])
    std_arr = np.array([58.395, 57.12, 57.375])
    image_norm = (image_rgb - mean_arr) / std_arr
    image_norm = np.transpose(image_norm, (2, 0, 1))  # CHW uint8
    image_norm = np.expand_dims(image_norm, axis=0)  # NCHW uint8
    input_data = image_norm.astype(np.float16)

    # 2. prepare input & output queue
    input_datas = []
    input_datas.append(input_data)

    qin = queue.Queue()
    qout = queue.Queue()
    for i in range(sample_num):
        qin.put((i, input_datas))

    # 3. define threads
    barrier = threading.Barrier(thread_num * device_num)
    thread_cnt = 0

    def thread_func(tid, did, module, qin, qout, barrier):
        count = 0
        # 3.1 prepare input
        input_infos = {}
        input_num = module.get_num_inputs()
        for id in range(0, input_num):
            input_name = module.get_input_name(id)
            input_info = module.get_input_info(input_name).ascontiguous()
            logger.info(
                "input[{}] shape = {}, dtype = {}, format = {}".format(
                    input_name,
                    input_info.shape,
                    input_info.dtype,
                    input_info.format.name,
                )
            )
            input_infos[input_name] = input_info
        # 3.2 prepare output
        output_infos = {}
        output_num = module.get_num_outputs()
        for id in range(0, output_num):
            output_name = module.get_output_name(id)
            output_info = module.get_output_info(output_name).ascontiguous()
            logger.info(
                "output[{}] shape = {}, dtype = {}, format = {}".format(
                    output_name,
                    output_info.shape,
                    output_info.dtype,
                    output_info.format.name,
                )
            )
            output_infos[output_name] = output_info
        # 3.3 wait until all threads ready
        barrier.wait()
        # 3.4 infer loop
        while not qin.empty():
            # 3.4.1 get data from the task queue
            req_id, input_datas = qin.get()

            # 3.4.2 set input to the module
            for input_name in input_infos:
                module.set_input(input_name, input_datas[0])

            # 3.4.3 run and sync
            module.run()
            module.sync()

            # 3.4.4 get output and push to the output queue
            output_datas = {}
            for output_name in output_infos:
                output_datas[output_name] = (
                    module.get_output(output_name, output_infos[output_name])
                    .astype(np.float32)
                    .numpy()
                )
            qout.put((req_id, output_datas))
            count += 1
            logger.info(
                "thread {} on device {} run sample {} end.".format(tid, did, req_id)
            )
        logger.info(
            "thread {} on device {} completed. {} sampels tested.".format(
                tid, did, count
            )
        )

    # 4.1 load models
    module_dict = {}
    for did in range(device_num):
        wm = tcim.runtime.WeightManager(did)
        option = tcim.runtime.Option(wm)
        module_dict[did] = []
        for i in range(thread_num):
            module = tcim.runtime.load(model_path, option=option)
            module_dict[did].append(module)
            logger.info(
                "thread {} on device {} load model {} loaded.".format(
                    i, did, model_path
                )
            )

    # 4.2 create threads
    tid = 0
    for did in range(device_num):
        for i in range(thread_num):
            threads.append(
                threading.Thread(
                    target=thread_func,
                    args=(tid, did, module_dict[did][i], qin, qout, barrier),
                )
            )
            tid += 1

    for thread in threads:
        thread.start()

    # 5. wait all threads done
    for thread in threads:
        thread.join()

    # 6. postprocess and check result
    while not qout.empty():
        req_id, output_datas = qout.get()
        for output_name in output_datas:
            from postprocess import softmax

            output_data = softmax(output_datas[output_name])
            pred = np.argsort(-output_data, axis=1, kind="quicksort").flatten()[0]
            prob_list = output_data.flatten()
            logger.info(
                "sample {} top1: predict cls = {}, prob = {:.6f}".format(
                    req_id, pred, prob_list[pred]
                )
            )
            # check result, modify it when you change model or data
            assert pred == 65

    logger.info("<=== resnet50_multistreams python example completed.\n")
