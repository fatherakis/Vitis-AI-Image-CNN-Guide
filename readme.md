# Vitis AI Image Classification Guide

This guide aims to help setting-up and running models on the Versal Platform VCK190.
It contains the basic steps in order to convert a model to its final version for Versal without pruning or any optional steps. This guide mainly focuses on PyTorch.

# Setup
Moving forward, this guide assumes your Versal board is fully setup and ready with Vitis-AI support as per the user guide provided.


# Table of Contents
1. [Preparation: Model & Host](#{prep})
2. [Vitis-AI 3.0](#vitis-ai)
3. [Versal](#versal)
4. [Credits](#credits)
## Preparation: Model & Host

First and foremost, about **60 - 100GB** are required for the docker container and the Vitis-AI repository.

### Model & Dataset


Any model can be applied if layers and operations are supported by Vitis-AI.\
https://docs.amd.com/r/3.0-English/ug1414-vitis-ai/Currently-Supported-Operators

For example:
* SiLU layers must be changed to LeakyReLU with 0.1 slope.
* Plus operations must be changed with FloatFunctional in case of Pytorch:
    >out += xx ---> out = self.ff.add(out, xx)

While datasets can be used with their respective loaders from PyTorch or Tensorflow or as .pkl files, they must be available as an Image Folder to run on versal without heavy code modifications.

For this example we use a ResNet20 model pre-trained on CIFAR-10 available at [[source]](https://github.com/chenyaofo/pytorch-cifar-models). The CIFAR-10 dataset is also extracted into an ImageFolder dataset type with each image in its respective class name folder.

*Also note that this formatting is proven useful when running inference on Versal since the correct label (ground truth) is available as each image's name.*

### Host
Assuming you have the model ready lets move on the preparation of the host.

1. Install [Docker](https://docs.docker.com/engine/install/): 
2. Verify its installation:
    * docker run hello-world <br />  
    * docker --version

3. Pull the [Vitis-AI container](https://xilinx.github.io/Vitis-AI/3.0/html/docs/install/install.html). Note: The VitisAI container is independent to the Vitis-AI 3.0 repo. 
    > docker pull xilinx/vitis-ai-\<framework>-\<arch>:latest <br />
    ``<framework>``: pytorch, tensorflow, tensorflow2\
    ``<arch>``: cpu, rocm

4. Now we are ready to pull the github repo:
    > git clone --branch 3.0 https://github.com/Xilinx/Vitis-AI.git

## Vitis-AI

After downloading the repository you can start Vitis-AI:
> ./docker_run.sh xilinx/vitis-ai-\<framework>-\<arch>:latest

where ``<framework>`` and ``<arch>`` the chosen installed container.

At this stage Vitis AI is running and you are ready to use its functions.

In this example we quantize and compile our pre-trained resnet20 model skipping any optional optimization steps. 

> [!IMPORTANT]
> This guide uses PyTorch. All codes and processes may differ from other frameworks.

* Activate the corresponding conda environment:
    > conda info --envs

    In case of Pytorch:
    > conda activate vitis-ai-pytorch

* You are now ready to run your python script based on the provided example.

    * Check /workspace/src/vai_q_pytorch/examples/ for effective examples.

    * It is also suggested to move any script and files in that folder for everything to run as smooth as possible.

Similarly to [<ins>resnet20_cifar_vai.py</ins>](resources/resnet20_cifar_vai.py) of this repo, in order to correctly quantize and compile a model you need 3 main things.

1. Your model loaded and modified with any conversions (if required).
2. Your dataset ready in order the calibration step of PTSQ quantization.
3. Any further modifications in normalization, image sizing etc. required by your model / dataset.

> [!Warning]
> The accuracy estimation of VitisAI does not reflect its actual performance whereas its resource estimations are spot on.

[<ins>resnet20_minimal_vai.py</ins>](resources/resnet20_minimal_vai.py) provides an example of the bare minimum requirements for the quantization and compilation to run smoothly.

### Running the script
Multiple argunents are available for each script most important of which is quantization mode: float, calib and test. We skip float mode in this guide as it simply runs inference on the model as every other tool.

- **calib** is a necessary step generating the VitisAI layer equivalent of the model along with its automatically generated bias correction. In this step we can use the same batch size as the floating point model however it should be noted that large values will require extreme resources exceeding 16GB RAM. If the program is killed during this step, simply reduce the batch size. In my tests it does not impact the performance of the model.

> [!TIP]
> If you intend to apply large models and batch sizes don't forget to change the processor and memory allocations from Docker.

> [!WARNING]
> If an error occurs stating any problems accessing the model or finding "ModelName.py" simply rerun the command and everything is fine.

- **test** is the final step where the model is prepared for the platform we want to apply it to. In this step, the batch size must be strictly 1. We also include arguments like --deploy in order to generate the .xmodel file which we transfer on Versal.

All in all, in order to convert our model we run the script twice with the following argumetns:

1. In order to quantize and calibrate our model:\
``python resnet20_cifar_vai.py --data_dir "dataset/CIFAR10" --model_dir "./" --model_name "resnet20_cifar" --batch_size 16 --target DPUCVDX8G_ISA3_C32B6 --quant_mode calib``

2. In order to deploy our quantized model in its .xmodel form:\
``python resnet20_cifar_vai.py --data_dir "dataset/CIFAR10" --model_dir "./" --model_name "resnet20_cifar" --batch_size 1 --target DPUCVDX8G_ISA3_C32B6 --quant_mode test --inspect --deploy``

> [!Note]
> Running the script as-is will result in failure. Simply extract the cifar dataset located in [cifar.zip](resources/cifar.zip) and update the --data_dir with your dataset location.

> [!CAUTION]
> **Each Versal Board has a specific DPU id which must be used as the target. DPUCVDX8G_ISA3_C32B6 is for VCK190**

### Finalizing Model

After running these commands, a "quantize_result" folder should be created containing the XIR .py version of the model and the exported .xmodel file along with some other ones for compatibility.

At this stage the .xmodel must be further processed by VitisAI Runtime in order to be utilized by the board.

In order to achieve that we must first run:
> /workspace/board_setup/vck190/host_cross_compiler_setup.sh

and\
``unset LD_LIBRARY_PATH``\
``source $install_path/environment-setup-cortexa72-cortexa53-xilinx-linux``

Which will resolve all the requirements and dependencies.

> [!Important]
> Do **<ins>NOT</ins>** run this before finishing with the model deployment as it will lead to unloading the PATH variable and requiring a docker restart.

> [!NOTE]
> When running the host_cross_compiler script, it will show the full path for the source command instead of "$install_path". Run that.

After the installation is done you can simply run:\
``vai_c_xir -x quantize_result/CifarResNet_int.xmodel -a /opt/vitis_ai/compiler/arch/DPUCVDX8G/VCK190/arch.json -o ./ -n resnet20CIFAR``

where -x takes the .xmodel generated from the script, -a is the corresponding architecture json for each board (**<ins>for VCK190  DO NOT CHANGE</ins>**), -o the output folder and -n the name of the finalized file.

The final file will be an isolated subgraph .xmodel  so make sure to give it a different name. If you want, you can see the differences between the previous and new .xmodel with [Netron](https://github.com/lutzroeder/netron)

## Versal

At this point the model is finalized and you can move on to the board.

In [Board_files](resources/board/) we can find all the required files for inference on our VCK190 Board for our ResNet20 on CIFAR10.\
The required files are broken down as followed:

1. Compiled model  (.xmodel)
2. Labels (for Image classification)  (cifar10_labels.dat)
3. Dataset in the form of images and folders (test.tar)
4. Bash scripts for output handling
5. Main source files in order to run the model. (C++ files)

The model is created from the previous process.

Labels can be easily changed with any text editor.

Do not add the whole dataset. Only the images you intend to test. In its current form the image name contains the correct label and its number as label_xxx.png.\
This can be changed to your preference but also update [check_runtime_top5_cifar10.py](resources/board/model_src/code/src/check_runtime_top5_cifar10.py) accordingly to have correct performance metrics and labels.

Bash Scripts:

* [run_src.sh](resources/board/run_src.sh) Simply runs the whole process

* [build_cifar10_test.sh](resources/board/model_src/build_cifar10_test.sh) unpacks the images and moves them in a single folder for easier access. You can change this accordingly to your labels and preferences.
* [cifar10_performance.sh](resources/board/model_src/cifar10_performance.sh) runs performance evaluations on the model for 1 to 6 DPU threads. There aren't much to change here except the number of threads and how the results are saved. By default it runs checks on the first 400 images. Change it according your needs.
* [run_all_cifar10_target.sh](resources/board/model_src/run_all_cifar10_target.sh) is an all in one script preparing the directories, compiling the binaries, running inference and then cleaning up. If you make any changes here make sure you change the corresponding files too.
* [Compilation arguments files](resources/board/model_src/code/) build_app.sh and build_get_dpu_fps.sh are the gcc compilation arguments for the corresponding sources files. It is not recommended to change these.
* [Inference handler](resources/board/model_src/code/src/check_runtime_top5_cifar10.py) is responsible for the label checking and handling of inference results along with accuracy calculations.


Main Files:

* [Common](resources/board/common/) These are included headers and dependencies, it is not recommended to change them.

* [Main](resources/board/model_src/code/src/main_int8.cc) is the main program which takes the model file and runs all actions. Here you should change the normalization arguments in lines 203 and 205 along with any other preprocessing steps you want to make. Also a change in line 256 may be required if the model is trained on [-1,1] range instead of [0,1].
* [DPU](resources/board/model_src/code/src/get_dpu_fps.cc) is a simple program for the DPU in order to use multiple threads. There shouldn't be any changes required here.


Having covered all files. You can easily run predictions on your model by connecting to the board and running:
``run_src.sh my_model``

## Credits

This work is based on the user guides of AMD and their examples on [GitHub](https://github.com/Xilinx/Vitis-AI.git) along with their suite of examples in [Vitis-AI-Tutorials](https://github.com/Xilinx/Vitis-AI-Tutorials/tree/3.0).

Also thanks to [@chenyaofo](https://github.com/chenyaofo/pytorch-cifar-models) for the pre-trained ResNet models.

The provided example contains heavily modified code and is not recommended to "copy-paste" without understanding its functionality.
