import argparse
import os
import sys
import shutil
import glob
import re

QUANT_MODELS_URL = (
    "http://10.10.1.53:8082/artifactory/toolchain/release/models_outdated/"
)


def parse_args():
    parser = argparse.ArgumentParser(description="Normalize Quant Folder")
    parser.add_argument(
        "--model_name",
        required=True,
        type=str,
        help="(required) model name, example: qwen3",
    )
    parser.add_argument(
        "--backend",
        required=True,
        type=str,
        help="(required) houmo backend, example: xh1, xh2",
    )
    parser.add_argument(
        "--quant_folder",
        required=True,
        type=str,
        help="(required) model name, example: qwen3",
    )
    parser.add_argument(
        "--result_folder",
        required=True,
        type=str,
        help="The path for storing the results.",
    )
    parser.add_argument(
        "--zipped_name",
        default=None,
        type=str,
        help="The name of the compressed package. If provided, it will be uploaded to Jfrog. Example: hmquant_xh2_qwen3_8b_2k_20250812",
    )

    args = parser.parse_args()
    return args


def find_file_recursive(root_dir, pattern, excludes=list()):
    """递归查找符合模式的文件"""
    matches = []
    for dirpath, _, filenames in os.walk(root_dir):
        for filename in filenames:
            filename_new = filename.lower()
            if glob.fnmatch.fnmatch(filename_new, pattern):
                flag = True
                for ex_str in excludes:
                    if ex_str in filename:
                        flag = False
                        break
                if flag is True:
                    matches.append(os.path.join(dirpath, filename))
    return matches


def _process_model_files_xh2(
    backend: str, quant_dir: str, result_dir: str, model_name: str, model_type: str
):
    try:
        # 1. 创建目标文件夹结构
        model_res_dir = os.path.join(result_dir, model_type)
        if model_type == "decode":
            model_res_dir = os.path.join(result_dir, "decoder")
        elif model_type == "vision":
            model_res_dir = os.path.join(result_dir, "visual")
        os.makedirs(model_res_dir, exist_ok=True)
        print(f"已创建目标文件夹结构: {model_res_dir}")

        if model_name == "qwen3-vl" and model_type == "vision":
            model_type = "visual"

        # 2. 处理 hmquant_*_with_act.onnx
        if model_type == "encoder":
            onnx_files = find_file_recursive(
                quant_dir,
                f"hmquant_*with_act.onnx",
                excludes=["decode", "prefill", "vision"],
            )
        else:
            onnx_files = find_file_recursive(
                quant_dir, f"hmquant_*{model_type}*with_act.onnx"
            )
        if len(onnx_files) == 0:
            return False
        # 取第一个匹配的文件
        onnx_src = onnx_files[0]
        onnx_dst = os.path.join(model_res_dir, f"hmquant_{model_name}_with_act.onnx")
        shutil.copy(onnx_src, onnx_dst)
        print(f"已重命名并移动: {onnx_src} -> {onnx_dst}")

        # 3. 处理 *_external_data文件
        if model_type == "encoder":
            external_files = find_file_recursive(
                quant_dir,
                "*external_data",
                excludes=["decode", "prefill", "vision"],
            )
        else:
            external_files = find_file_recursive(
                quant_dir, f"*{model_type}*external_data"
            )
        if len(external_files) == 0:
            return False
        for external_file in external_files:
            # 检查该文件是否与找到的onnx文件在同一目录
            onnx_dir = os.path.dirname(onnx_files[0]) if onnx_files else None
            external_dir = os.path.dirname(external_file)

            # 如果找到了onnx文件，只移动同目录下的external_data文件
            if onnx_dir and external_dir == onnx_dir:
                file_name = os.path.basename(external_file)
                external_dst = os.path.join(model_res_dir, file_name)

                shutil.copy(external_file, external_dst)
                print(f"已移动: {external_file} -> {external_dst}")

        # 4. 处理golden数据
        if model_type == "encoder":
            golden = find_file_recursive(
                quant_dir,
                "hmquant_*_input.npy",
                excludes=["decode", "prefill", "vision"],
            )
            opt_golden = find_file_recursive(
                quant_dir,
                "hmquant_*_output.npy",
                excludes=["decode", "prefill", "vision"],
            )
            golden += opt_golden
        else:
            golden = find_file_recursive(quant_dir, f"hmquant_*{model_type}*_input.npy")
            opt_golden = find_file_recursive(
                quant_dir, f"hmquant_*{model_type}*_output.npy"
            )
            golden += opt_golden

        name_pattern = r"(?<=hmquant_).*?_decode"
        if model_type == "prefill":
            name_pattern = r"(?<=hmquant_).*?_prefill"
        elif model_type == "vision":
            if model_name == "qwen2.5-vl":
                name_pattern = r'(hmquant_)(qwen2\.5-vl-7b-insturct-vision_xh2a_)(.*?)(_batch_image_)(.*?)(\.npy)'
            elif model_name == "qwen3-vl":
                name_pattern = (
                    r'(hmquant_)(qwen3_vl_instruct)(_vision)(_config_)(.*?)(\.npy)'
                )
            elif model_name == "minicpmo":
                name_pattern = (
                    r'(hmquant_)(minicpmo_vision_)(7b_xh2a_)(.*?k_)(.*?)(\.npy)'
                )
        elif model_type == "encoder":
            name_pattern = r'(hmquant_)(whisper_meduim_xh2a_w8a8_sefp_)(.*?)(\.npy)'

        if model_type in ["decode", "prefill", "encoder"] and model_name == "whisper":
            name_pattern = r"(?<=hmquant_).*?_sefp"

        for data_file in golden:
            file_name = os.path.basename(data_file)
            if "image_embeds" in file_name:
                data_dst = os.path.join(model_res_dir, "image_embeds.npy")
                shutil.copy(data_file, data_dst)
                print(f"已移动: {data_file} -> {data_dst}")
                continue
            match = re.search(name_pattern, file_name)
            if match:
                if model_type != "vision":
                    original_name = match.group()
                    file_name_new = file_name.replace(original_name, model_name)
                else:
                    file_name_new = (
                        f"{match.group(1)}{model_name}_{match.group(5)}{match.group(6)}"
                    )
                data_dst = os.path.join(model_res_dir, file_name_new)
                shutil.copy(data_file, data_dst)
                print(f"已移动: {data_file} -> {data_dst}")

    except Exception as e:
        print(f"处理过程中出错: {str(e)}")
        return False

    return True


def _process_model_files_xh1(
    backend: str, quant_dir: str, result_dir: str, model_name: str, model_type: str
):
    try:
        # 1. 创建目标文件夹结构
        model_res_dir = os.path.join(result_dir, model_type)
        if model_type == "decode":
            model_res_dir = os.path.join(result_dir, "decoder")
        elif model_type == "vision":
            model_res_dir = os.path.join(result_dir, "visual")
        os.makedirs(model_res_dir, exist_ok=True)
        print(f"已创建目标文件夹结构: {model_res_dir}")

        if model_name in ["qwen3-vl", "qwen2.5-vl"] and model_type == "vision":
            model_type = "visual"

        # 2. 处理 hmquant_*_with_act.onnx
        onnx_files = find_file_recursive(
            quant_dir, f"hmquant_*{model_type}*with_act.onnx"
        )
        if len(onnx_files) == 0:
            return False
        # 取第一个匹配的文件
        onnx_src = onnx_files[0]
        onnx_dst = os.path.join(model_res_dir, f"hmquant_{model_name}_with_act.onnx")
        shutil.copy(onnx_src, onnx_dst)
        print(f"已重命名并移动: {onnx_src} -> {onnx_dst}")

        # 3. 处理golden数据
        golden = find_file_recursive(quant_dir, f"hmquant_*{model_type}*_input.npy")
        opt_golden = find_file_recursive(
            quant_dir, f"hmquant_*{model_type}*_output.npy"
        )
        golden += opt_golden

        name_pattern = r"(?<=hmquant_).*?_decode"
        if model_type == "prefill":
            name_pattern = r"(?<=hmquant_).*?_prefill"
        elif model_type in ["vision", "visual"]:
            name_pattern = r"(?<=hmquant_).*?_visual"

        if model_name in ["qwen3-vl", "qwen2.5-vl"]:
            if model_type == "decode":
                name_pattern = r"(?<=hmquant_).*?_decoder"
            elif model_type == "prefill":
                name_pattern = r"(?<=hmquant_).*?_Prefill"

        for data_file in golden:
            file_name = os.path.basename(data_file)
            if "image_embeds" in file_name:
                data_dst = os.path.join(model_res_dir, "image_embeds.npy")
                shutil.copy(data_file, data_dst)
                print(f"已移动: {data_file} -> {data_dst}")
                continue
            if model_type in ["vision", "visual"] and (
                "Prefill" in file_name
                or "decode" in file_name
                or "prefill" in file_name
            ):
                continue
            match = re.search(name_pattern, file_name)
            if match:
                original_name = match.group()
                file_name_new = file_name.replace(original_name, model_name)
                data_dst = os.path.join(model_res_dir, file_name_new)
                shutil.copy(data_file, data_dst)
                print(f"已移动: {data_file} -> {data_dst}")

    except Exception as e:
        print(f"处理过程中出错: {str(e)}")
        return False

    return True


def _process_model_files(
    backend: str, quant_dir: str, result_dir: str, model_name: str, model_type: str
):
    if backend == "xh1":
        return _process_model_files_xh1(
            backend, quant_dir, result_dir, model_name, model_type
        )
    elif backend == "xh2":
        return _process_model_files_xh2(
            backend, quant_dir, result_dir, model_name, model_type
        )


def _process_weight_file(quant_dir: str, result_dir: str):
    weight_src = f"{quant_dir}/weight.npy"
    if os.path.exists(weight_src):
        weight_dst = os.path.join(result_dir, "weight.npy")
        shutil.copy(weight_src, weight_dst)
        print(f"已移动: {weight_src} -> {weight_dst}")

    weight_src = f"{quant_dir}/decoder/weight.npy"
    if os.path.exists(weight_src):
        os.makedirs(f"{result_dir}/decoder", exist_ok=True)
        weight_dst = os.path.join(result_dir, "decoder", "weight.npy")
        shutil.copy(weight_src, weight_dst)
        print(f"已移动: {weight_src} -> {weight_dst}")

    weight_src = f"{quant_dir}/prefill/weight.npy"
    if os.path.exists(weight_src):
        os.makedirs(f"{result_dir}/prefill", exist_ok=True)
        weight_dst = os.path.join(result_dir, "prefill", "weight.npy")
        shutil.copy(weight_src, weight_dst)
        print(f"已移动: {weight_src} -> {weight_dst}")

    weight_src = f"{quant_dir}/visual/weight.npy"
    if os.path.exists(weight_src):
        os.makedirs(f"{result_dir}/visual", exist_ok=True)
        weight_dst = os.path.join(result_dir, "visual", "weight.npy")
        shutil.copy(weight_src, weight_dst)
        print(f"已移动: {weight_src} -> {weight_dst}")


if __name__ == "__main__":
    args = parse_args()

    model_name = args.model_name
    backend = args.backend
    quant_dir = args.quant_folder
    result_dir = args.result_folder
    zipped_name = args.zipped_name

    try:
        # 创建目标文件夹结构
        if os.path.exists(result_dir):
            shutil.rmtree(result_dir, ignore_errors=True)
        os.makedirs(result_dir, exist_ok=True)
        print(f"已创建目标文件夹结构: {result_dir}")

        if backend == "xh1":
            _process_weight_file(quant_dir, result_dir)

        # 处理token_embedding.pt
        for embedding_src in glob.glob(
            os.path.join(quant_dir, "**", "*embedding*.pt"), recursive=True
        ):
            file_name = os.path.basename(embedding_src)
            if "quant_embedding" not in file_name and "token_embedding" in file_name:
                file_name = file_name.replace("token_embedding", "quant_embedding")
            if "quant_embedding" not in file_name and "qembedding" in file_name:
                file_name = file_name.replace("qembedding", "quant_embedding")
            embedding_dst = os.path.join(result_dir, file_name)
            shutil.copy(embedding_src, embedding_dst)
            print(f"已移动: {embedding_src} -> {embedding_dst}")

        if not _process_model_files(
            backend, quant_dir, result_dir, model_name, "decode"
        ):
            print(f"错误: 处理decode文件夹失败")
            sys.exit(-1)
        if not _process_model_files(
            backend, quant_dir, result_dir, model_name, "prefill"
        ):
            print(f"错误: 处理prefill文件夹失败")
            sys.exit(-1)
        if (
            os.path.exists(f"{quant_dir}/vision")
            or os.path.exists(f"{quant_dir}/visual")
            or "-vl" in model_name
        ) and not _process_model_files(
            backend, quant_dir, result_dir, model_name, "vision"
        ):
            print(f"错误: 处理visual文件夹失败")
            sys.exit(-1)

        if os.path.exists(f"{quant_dir}/encoder") and not _process_model_files(
            backend, quant_dir, result_dir, model_name, "encoder"
        ):
            print(f"错误: 处理encoder文件夹失败")
            sys.exit(-1)
        print("文件处理完成！")

        # 如果指定了压缩文件名字，则压缩并上传到Jfrog
        if zipped_name:
            ret = os.system(f"cd {result_dir} && zip -r {zipped_name}.zip ./*")
            if ret != 0:
                print("压缩量化文件夹失败。")
                sys.exit(-1)

            model_folder = "deepseek" if "deepseek" in model_name else model_name
            jfrog_file_path = f"{QUANT_MODELS_URL}/{model_folder}/"
            ret = os.system(
                f"curl -u public:Password@123 -T {result_dir}/{zipped_name}.zip {jfrog_file_path}"
            )
            if ret != 0:
                print("上传量化文件压缩包失败。")
                sys.exit(-1)
            print(f"上传量化文件压缩包 {zipped_name}.zip 成功。")

    except Exception as e:
        print(f"处理过程中出错: {str(e)}")
        sys.exit(-1)
