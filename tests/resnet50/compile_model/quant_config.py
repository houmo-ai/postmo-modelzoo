inputs_cfg = {
    "ALL": dict(
        data_format = "RGB",
        first_layer_weight_denorm_mean = [0.485, 0.456, 0.406],
        first_layer_weight_denorm_std = [0.229, 0.224, 0.225],
        resizer_crop = {"top": 0, "left": 0, "height": 224, "width": 224},
        resizer_resize = {
            "height": 224,
            "width": 224,
            "align_corners": False,
            "method": "bilinear"},
        toYUV_format = 'YUV422',
        )
}

graph_opt_cfg = dict()