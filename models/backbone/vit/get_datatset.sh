#!/bin/bash

DATASET_NAME=ILSVRC2012
DATA_DIR=/usr/local/src/data
DATASET_PATH=${DATA_DIR}/${DATASET_NAME}.tar.gz
mkdir -p ${DATA_DIR}
if ! [ -f "${DATASET_PATH}" ]; then
    echo "Downloading ILSVRC2012 dataset..."
    wget ftp://113.100.143.90:821//data/datasets/ILSVRC2012.tar.gz --ftp-user=ftp_guest --ftp-password=IfTy@2022 --directory-prefix=${DATA_DIR}
else
    echo "Dataset ${DATASET_NAME} already exists. Skip downloading."
fi
if ! [ -d "${DATA_DIR}/${DATASET_NAME}" ]; then
    echo "Extracting ILSVRC2012 dataset..."
    tar --skip-old-files -xvf ${DATA_DIR}/${DATASET_NAME}.tar.gz -C ${DATA_DIR}
else
    echo "Dataset ${DATASET_NAME} already exists. Skip extracting."
fi
ln -sf ${DATA_DIR}/${DATASET_NAME}
echo "Download ${DATASET_NAME} dataset from ftp successfully."