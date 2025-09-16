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


def find_file_recursive(root_dir, pattern):
    """递归查找符合模式的文件"""
    matches = []
    for dirpath, _, filenames in os.walk(root_dir):
        for filename in filenames:
            if glob.fnmatch.fnmatch(filename, pattern):
                matches.append(os.path.join(dirpath, filename))
    return matches


def _process_model_files(
    quant_dir: str, result_dir: str, model_name: str, model_type: str
):
    try:
        # 1. 创建目标文件夹结构
        model_res_dir = os.path.join(result_dir, model_type)
        if model_type == "decode":
            model_res_dir = os.path.join(result_dir, "decoder")
        os.makedirs(model_res_dir, exist_ok=True)
        print(f"已创建目标文件夹结构: {model_res_dir}")

        # 2. 处理 hmquant_*_with_act.onnx
        onnx_files = find_file_recursive(
            quant_dir, f"hmquant_*_{model_type}_with_act.onnx"
        )
        if len(onnx_files) == 0:
            return False
        # 取第一个匹配的文件
        onnx_src = onnx_files[0]
        onnx_dst = os.path.join(model_res_dir, f"hmquant_{model_name}_with_act.onnx")
        shutil.copy(onnx_src, onnx_dst)
        print(f"已重命名并移动: {onnx_src} -> {onnx_dst}")

        # 3. 处理 *_external_data文件
        external_files = find_file_recursive(quant_dir, f"*_{model_type}_external_data")
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
        golden = find_file_recursive(quant_dir, f"hmquant_*_{model_type}_*_input.npy")
        opt_golden = find_file_recursive(
            quant_dir, f"hmquant_*_{model_type}_*_output.npy"
        )
        golden += opt_golden
        name_pattern = (
            r"(?<=hmquant_).*?(?=_decode)"
            if model_type == "decode"
            else r"(?<=hmquant_).*?(?=_prefill)"
        )
        for data_file in golden:
            file_name = os.path.basename(data_file)
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


if __name__ == "__main__":
    args = parse_args()

    model_name = args.model_name
    quant_dir = args.quant_folder
    result_dir = args.result_folder
    zipped_name = args.zipped_name

    try:
        # 创建目标文件夹结构
        if os.path.exists(result_dir):
            shutil.rmtree(result_dir, ignore_errors=True)
        os.makedirs(result_dir, exist_ok=True)
        print(f"已创建目标文件夹结构: {result_dir}")

        # 处理token_embedding.pt
        embedding_files = find_file_recursive(quant_dir, "token_embedding.pt")
        if embedding_files:
            # 使用找到的第一个文件
            embedding_src = embedding_files[0]
            embedding_dst = os.path.join(result_dir, "quant_embedding.pt")
            shutil.copy(embedding_src, embedding_dst)
            print(f"已移动: {embedding_src} -> {embedding_dst}")
        else:
            print(f"错误: 未找到embedding文件")
            sys.exit(-1)

        if not _process_model_files(quant_dir, result_dir, model_name, "decode"):
            print(f"错误: 处理decode文件夹失败")
            sys.exit(-1)
        if not _process_model_files(quant_dir, result_dir, model_name, "prefill"):
            print(f"错误: 处理prefill文件夹失败")
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
