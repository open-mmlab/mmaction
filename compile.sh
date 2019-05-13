#!/bin/bash

PYTHON=${PYTHON:-"python"}

echo "Building package resample2d"
cd ./mmaction/ops/resample2d_package
if [ -d "build" ]; then
    rm -r build
fi

$PYTHON setup.py install --user

echo "Building package trajectory_conv..."
cd ../trajectory_conv_package
if [ -d "build" ]; then
    rm -r build
fi

$PYTHON setup.py install --user
