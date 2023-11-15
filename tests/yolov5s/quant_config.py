inputs_cfg = {
    "ALL":dict(
        data_format="RGB",
        first_layer_weight_denorm_mean=[0.485, 0.456, 0.406],
        first_layer_weight_denorm_std=[0.229, 0.224, 0.225],
        resizer_crop={"top": 0, "left": 0, "height": 0, "width": 0},
        resizer_resize={
            "height": 384,
            "width": 640,
            "align_corners": False,
            "method": "bilinear"}
        )}

graph_opt_cfg = dict(
    auto_quant_flag = True
)

base_o_observer = dict(
    type="kl",
    observer_param_dict=dict(percent=0.99999)
)

op_cfg = dict(
    global_cfg = dict(o_observer=base_o_observer),
)

# python demo/ptq/yolov5/yolov5.py --onnx_path dev/model_zoo2/unkown/yolov5s.onnx 
# --config_path configs/yolov5/yolov5_384x640_rgb2yuv_minmax.py --input_shape 384 640 