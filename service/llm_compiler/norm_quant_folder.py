import argparse
import os
import shutil
import glob


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


def find_dir_recursive(root_dir, pattern):
    """递归查找符合模式的目录中的文件"""
    matches = []
    for dirpath, _, filenames in os.walk(root_dir):
        if glob.fnmatch.fnmatch(os.path.basename(dirpath), pattern):
            for filename in filenames:
                matches.append(os.path.join(dirpath, filename))
    return matches


if __name__ == "__main__":
    args = parse_args()

    model_name = args.model_name
    quant_dir = args.quant_folder
    result_dir = args.result_folder

    try:
        # 创建目标文件夹结构
        decoder_dir = os.path.join(result_dir, "decoder")
        if os.path.exists(decoder_dir):
            shutil.rmtree(result_dir, ignore_errors=True)
        os.makedirs(decoder_dir, exist_ok=True)
        prefill_dir = os.path.join(result_dir, "prefill")
        os.makedirs(prefill_dir, exist_ok=True)
        print(f"已创建目标文件夹结构: {result_dir} 和 {decoder_dir} 和 {prefill_dir}")

        # 1. 处理token_embedding.pt
        embedding_files = find_file_recursive(quant_dir, "token_embedding.pt")
        if embedding_files:
            # 使用找到的第一个文件
            embedding_src = embedding_files[0]
            embedding_dst = os.path.join(result_dir, "quant_embedding.pt")
            shutil.copy(embedding_src, embedding_dst)
            print(f"已移动: {embedding_src} -> {embedding_dst}")
        else:
            print(f"警告: 未找到文件 {embedding_src}")

        # 2. 处理hmquant_*_with_act.onnx
        decode_onnx_files = find_file_recursive(
            quant_dir, "hmquant_*_decode_with_act.onnx"
        )
        if decode_onnx_files:
            # 取第一个匹配的文件
            onnx_src = decode_onnx_files[0]
            onnx_dst = os.path.join(decoder_dir, f"hmquant_{model_name}_with_act.onnx")
            shutil.copy(onnx_src, onnx_dst)
            print(f"已重命名并移动: {onnx_src} -> {onnx_dst}")
        else:
            print(f"警告: 未找到匹配的onnx文件 hmquant_*_decode_with_act.onnx")

        # 3. 处理*_external_data文件
        data_files = find_file_recursive(quant_dir, "*_external_data")
        if data_files:
            for data_file in data_files:
                # 检查该文件是否与找到的onnx文件在同一目录
                onnx_dir = (
                    os.path.dirname(decode_onnx_files[0]) if decode_onnx_files else None
                )
                data_dir = os.path.dirname(data_file)

                # 如果找到了onnx文件，只移动同目录下的external_data文件
                if onnx_dir and data_dir == onnx_dir:
                    file_name = os.path.basename(data_file)
                    data_dst = os.path.join(decoder_dir, file_name)

                    shutil.copy(data_file, data_dst)
                    print(f"已移动: {data_file} -> {data_dst}")
        else:
            print(f"警告: 未找到匹配的external_data文件 *_external_data")

        # 4. 处理hmquant_*_with_act.onnx
        prefill_onnx_files = find_file_recursive(
            quant_dir, "hmquant_*_prefill_with_act.onnx"
        )
        if prefill_onnx_files:
            # 取第一个匹配的文件
            onnx_src = prefill_onnx_files[0]
            onnx_dst = os.path.join(prefill_dir, f"hmquant_{model_name}_with_act.onnx")
            shutil.copy(onnx_src, onnx_dst)
            print(f"已重命名并移动: {onnx_src} -> {onnx_dst}")
        else:
            print(f"警告: 未找到匹配的onnx文件 hmquant_*_prefill_with_act.onnx")

        # 5. 处理*_external_data文件
        if data_files:
            for data_file in data_files:
                # 检查该文件是否与找到的onnx文件在同一目录
                onnx_dir = (
                    os.path.dirname(prefill_onnx_files[0])
                    if prefill_onnx_files
                    else None
                )
                data_dir = os.path.dirname(data_file)

                # 如果找到了onnx文件，只移动同目录下的external_data文件
                if onnx_dir and data_dir == onnx_dir:
                    file_name = os.path.basename(data_file)
                    data_dst = os.path.join(prefill_dir, file_name)

                    shutil.copy(data_file, data_dst)
                    print(f"已移动: {data_file} -> {data_dst}")
        else:
            print(f"警告: 未找到匹配的external_data文件 *_external_data")

        print("文件处理完成！")

    except Exception as e:
        print(f"处理过程中出错: {str(e)}")
