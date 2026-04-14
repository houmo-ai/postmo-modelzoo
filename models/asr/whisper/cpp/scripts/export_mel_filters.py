#!/usr/bin/env python3
# Copyright (c) 2026 HOUMO AI
# SPDX-License-Identifier: Apache-2.0
#
# Export mel filter bank coefficients from HuggingFace transformers
# Output: mel_filters.h header file

"""
Generate mel filter bank header file for Whisper ASR preprocessing.

This script extracts the mel filter bank matrix from HuggingFace's
WhisperFeatureExtractor and exports it as a C header file.

Usage:
    python export_mel_filters.py [--model_path <path>] [--output <file>]

Default model: whisper-medium
Default output: mel_filters.h
"""

import argparse
import sys

try:
    from transformers import WhisperFeatureExtractor
    import numpy as np
except ImportError:
    print("Error: transformers and numpy are required")
    print("Install: pip install transformers numpy")
    sys.exit(1)


def export_mel_filters(model_path: str, output_path: str):
    """Export mel filter bank to C header file."""
    # Load feature extractor
    fe = WhisperFeatureExtractor.from_pretrained(model_path)
    mel_filters = fe.mel_filters.T  # Transpose to [80, 201] shape

    # Write header file
    with open(output_path, "w") as f:
        f.write("// Copyright (c) 2026 HOUMO AI\n")
        f.write("// SPDX-License-Identifier: Apache-2.0\n")
        f.write("//\n")
        f.write("// Mel filter bank coefficients for Whisper ASR\n")
        f.write("// Generated from HuggingFace transformers WhisperFeatureExtractor\n")
        f.write("//\n\n")
        f.write("#pragma once\n\n")
        f.write(f"static const int N_MELS = {mel_filters.shape[0]};\n")
        f.write(f"static const int N_FFT_BINS = {mel_filters.shape[1]};\n\n")
        f.write(f"static const float MEL_FILTERS[{mel_filters.shape[0]}][{mel_filters.shape[1]}] = {{\n")

        for i in range(mel_filters.shape[0]):
            f.write("    {")
            f.write(", ".join([f"{x:.18e}f" for x in mel_filters[i]]))
            f.write("},\n")

        f.write("};\n")

    print(f"Generated {output_path}")
    print(f"Shape: [{mel_filters.shape[0]} mel bins, {mel_filters.shape[1]} FFT bins]")
    print(f"Model: {model_path}")


def main():
    parser = argparse.ArgumentParser(description="Export mel filter bank for Whisper")
    parser.add_argument(
        "--model_path",
        default="openai/whisper-medium",
        help="Whisper model path (default: openai/whisper-medium)"
    )
    parser.add_argument(
        "--output",
        default="mel_filters.h",
        help="Output header file path (default: mel_filters.h)"
    )
    args = parser.parse_args()

    export_mel_filters(args.model_path, args.output)


if __name__ == "__main__":
    main()