import os
import sys
import threading
import queue
import numpy as np
import cv2
import torch
import argparse

import tcim_lite as tcim


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
    sys.path.insert(0, "../common/python")
    print("\n===> resnet50_multistreams python example start...")
    print("tcim runtime version: {}".format(tcim.runtime.get_version()))

    # set the parameters
    args = get_args()
    device_num = args.device_num
    thread_num = args.thread_num
    sample_num = args.sample_num
    model_path = "resnet50.hmm"
    if not os.environ.get("HDPL_PLATFORM") == "ASIC":
        thread_num = 1
    print("devices:", device_num)
    print("threads:", thread_num)
    print("samples:", sample_num)
    print("model:", model_path)

    modules = []
    threads = []

    if not os.path.exists(model_path):
        print("[error] could not find model: {}".format(model_path))
        exit(-1)

    # 1. input preprocess
    input_data = cv2.imread("../data/snake.jpg")
    input_data = cv2.resize(input_data, (224, 224))  # HWC uint8
    input_data = np.transpose(input_data, (2, 0, 1))  # CHW uint8
    input_data = np.expand_dims(input_data, axis=0)  # NCHW uint8
    input_data = torch.tensor(input_data.astype(np.float32))  # NHWC float32
    input_data = torch.squeeze(input_data, 0)  # HWC float32
    from transform import BGR2YUV
    rgb2yuv_func = BGR2YUV(fmt='YUV420')
    input_data = torch.unsqueeze(rgb2yuv_func(input_data), 0).numpy()  # NHWC float32
    input_data = input_data.astype(np.uint8)

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
    def thread_func(tid, did, model_path, wm, qin, qout, barrier):
        count = 0
        # 3.1 load model, create a stream and set to the module
        option = tcim.runtime.Option(wm)
        module = tcim.runtime.load(model_path, option = option)
        print("thread {} on device {} load model {} loaded.".format(tid, did, model_path))
        stream = tcim.runtime.Stream()
        module.set_stream(stream)

        # 3.2 prepare input
        input_infos = {}
        input_num = module.get_num_inputs()
        for id in range(0, input_num):
            input_name = module.get_input_name(id)
            if input_name.endswith(".y"):
                input_name, _ = input_name.split(".y")
            elif input_name.endswith(".uv"):
                continue
            input_info = module.get_input_info(input_name)
            print("input[{}] shape = {}, dtype = {}, format = {}"
                  .format(input_name,
                          input_info.shape,
                          input_info.dtype,
                          input_info.format.name))
            input_infos[input_name] = input_info
        # 3.3 prepare output
        output_infos = {}
        output_num = module.get_num_outputs()
        for id in range(0, output_num):
            output_name = module.get_output_name(id)
            output_info = module.get_output_info(output_name)
            print("output[{}] shape = {}, dtype = {}, format = {}"
                  .format(output_name,
                          output_info.shape,
                          output_info.dtype,
                          output_info.format.name))
            output_infos[output_name] = output_info
        # 3.4 wait until all threads ready
        barrier.wait()
        # 3.5 infer loop
        while not qin.empty():
            # 3.5.1 get data from the task queue
            req_id, input_datas = qin.get()

            # 3.5.2 set input to the module
            for input_name in input_infos:
                module.set_input(input_name, tcim.runtime.Tensor(input_infos[input_name], input_datas[0]))

            # 3.5.3 run and sync
            module.run()
            module.sync()

            # 3.5.4 get output and push to the output queue
            output_datas = {}
            for output_name in output_infos:
                output_datas[output_name] = module.get_output(output_name, output_infos[output_name]).cast(np.float32).numpy()
            qout.put((req_id, output_datas))
            count += 1
            print("thread {} on device {} run sample {} end.".format(tid, did, req_id))
        print("thread {} on device {} completed. {} sampels tested.".format(tid, did, count))

    # 4. create threads
    tid = 0
    for did in range(device_num):
        wm = tcim.runtime.WeightManager(did)
        for i in range(thread_num):
            threads.append(threading.Thread(target=thread_func, args=(tid, did, model_path, wm, qin, qout, barrier)))
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
            print("sample {} top1: predict cls = {}, prob = {:.6f}".format(req_id, pred, prob_list[pred]))
            assert(pred == 65)

    print("<=== resnet50_multistreams python example completed.\n")
