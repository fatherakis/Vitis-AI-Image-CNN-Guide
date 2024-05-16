#!/bin/bash

# Copyright © 2023 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

# Author: Daniele Bagni, Xilinx Inc
# date:  28 Apr. 2023


TARGET=$1

MODELFILE=$2

# check DPU prediction top1_accuracy
echo " "
echo " CIFAR10 ${MODELFILE} TOP5 ACCURACY"
echo " "
python3 ./code/src/check_runtime_top5_cifar10.py -n 400 -i ./rpt/predictions_cifar10_${MODELFILE}.log | tee ./rpt/results_predictions_${MODELFILE}.log


#echo " "
#echo " CIFAR10 ${MODELFILE} PERFORMANCE (fps)"
#echo " "
#./get_dpu_fps ./${MODELFILE}.xmodel  1 10000  | tee  ./rpt/log1.txt  # 1 thread
#./get_dpu_fps ./${MODELFILE}.xmodel  2 10000  | tee  ./rpt/log2.txt  # 2 threads
#./get_dpu_fps ./${MODELFILE}.xmodel  3 10000  | tee  ./rpt/log3.txt  # 3 threads
#cat ./rpt/log1.txt ./rpt/log2.txt ./rpt/log3.txt >  ./rpt/${MODELFILE}_CIFAR_results_fps.log
#rm -f ./rpt/log?.txt

echo " "
