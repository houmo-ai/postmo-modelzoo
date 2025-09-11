from hmatc.utils.utils import get_file_from_jfrog


build_path = "models/yolov5s/yolov5s_clip_xh1_b1_1roi_1core_O2_dynamic_v2_v2.5.0.tar.xz"
get_file_from_jfrog(build_path, "./", "./")
