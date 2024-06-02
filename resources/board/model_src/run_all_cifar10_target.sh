#!/bin/bash

# Copyright © 2023 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

# Author: Daniele Bagni, Xilinx Inc
# date:  28 Apr. 2023

MODELFILE=$2


#clean
clean_(){
echo " "
echo "clean"
echo " "
cd model_src
rm -rf test
rm -f *~
rm -f  run_cnn cnn* get_dpu_fps *.txt
rm -rf rpt
rm -f  *.txt
rm -f  *.log
mkdir -p rpt
cd ..
}

# compile CNN application
compile_(){
echo " "
echo "compile"
echo " "
cd model_src/code
echo "PWD1 = " $PWD
bash -x ./build_app.sh
mv code ../cnn_inf # change name of the application
bash -x ./build_get_dpu_fps.sh
mv code ../get_dpu_fps
cd ../..
echo "PWD2 = " $PWD
}

# build cifar10 test images
test_images_(){
echo " "
echo "build test images"
echo " "
cd model_src
bash ./build_cifar10_test.sh
cd ..
echo " "
echo "PWD3 = " $PWD
}

# now run the cifar10 classification with 4 CNNs using VART C++ APIs
run_cnn_(){
echo " "
echo " run CNN"
echo " "
cd model_src
./cnn_inf ./${MODELFILE}.xmodel ./test/ ./cifar10_labels.dat | tee ./rpt/predictions_${MODELFILE}.log
# check DPU prediction accuracy
bash -x ./cifar10_performance.sh ${MODELFILE}
echo "PWD4 = " $PWD
cd ..
}

#remove images
end_(){
echo " "
echo "end "
echo " "
cd model_src
rm -rf test
cd ../
echo "PWD5 = " $PWD
#tar -cvf target.tar ./target_*
}


main()
{
    clean_
    compile_
    test_images_
    run_cnn_
    end_
}




"$@"
