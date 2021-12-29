## Setup
'''bash
	python3 setup.py install 
'''
### Prerequisites

* Ubuntu\* 18.04
* Python\* 3.7
* TensorFlow\* 1.14

### Installation

1. Clone and checkout state of `tensorflow/models`: r1.13
    ```bash
    cd tfmodels
    git clone https://github.com/tensorflow/models/tree/r1.13.0
    ```

    If you have tensorflow/models previously, you can change path to models at file: utils/tfutils/helpers.py

    line 27: research_dir = path.realpath(path.join(path.dirname(__file__), '../../tfmodels/models/research/'))
    line 35: transformer_dir = path.realpath(path.join(path.dirname(__file__), '../../tfmodels/models/official/transformer'))
    

2. Install the modules: (Build function tfutils)

    ```bash
    cd lpr
    pip3 install -e .
    pip3 install -e ./utils
    ```

    In the case without GPU, use the `CPU_ONLY=true` environment variable:

    ```bash
    CPU_ONLY=true pip3 install -e .
    pip3 install -e ./utils
    ```
NOTE: * file 'spatial_transformer.py' is added.
## Train an LPRNet Model

### Prepare a Dataset

1. Create dataset
    - Prepare one line or rectangle vietnam number plate
    - Create file `dataset/annotation` with content:
        p3/23_80-CN-0_crop_0.jpg 30Y-1311
        p3/23_81-CN-0_crop_0.jpg 15A-471.65
        p3/23_82-CN-0_crop_0.jpg 80A-035.60
        p3/23_82-CN-1_crop_0.jpg 15A-471.65
        p3/23_83-CN-0_crop_0.jpg 80A-035.60
        p3/23_88-CN-0_crop_0.jpg 29A-338.06

2. Run the Python script from
    `make_train_val_split.py` to split the annotations into `train` and `val` by passing the path to `dataset/annotation`
    Use the command below:

    ```bash
    python3 make_train_val_split.py dataset/annotation
    ```

    The resulting structure of the folder:

    ```
    ./data/synthetic_license_plates/
    ├── make_train_val_split.py
    └── Synthetic_License_Plates/
        ├── annotation
        ├── crops/
        │   ├── 000000.png
        |   ...
        ├── LICENSE
        ├── README
        ├── train
        └── val
    ```
    ------------------------------
    Change your absolute path: train, val, test in ../vietnam_lpr/config.py

    line 27:    file_list_path = '/media/thanglmb/Bkav/AICAM/TrainModels/OCR4ANPR/LPRNet/tensorflow_toolkit/lpr/dataset/train'
    line 53:    file_list_path = '/media/thanglmb/Bkav/AICAM/TrainModels/OCR4ANPR/LPRNet/tensorflow_toolkit/lpr/dataset/val'
    line 65:    file_list_path = '/media/thanglmb/Bkav/AICAM/TrainModels/OCR4ANPR/LPRNet/tensorflow_toolkit/lpr/dataset/test_infer'

### Train and Evaluate

1. To start the training process, use the command below:

    ```bash
    export lprnet in your absolute path 
    $ export PYTHONPATH=/media/thanglmb/Bkav/AICAM/TrainModels/OCR4ANPR/lprnet
    $ python3 tools/train.py vietnam_lp/config.py
    ```

2. To start evaluation process, use the command below:

    ```bash
    python3 tools/eval.py chinese_lp/config.py
    ```

    > **NOTE** Before taking the step 4, make sure that the `eval.file_list_path` parameter in
    `lpr/vietnam_lp/config.py` points out to the file with
    annotations to test on. Take the step 4 in another terminal, so training and
    evaluation are performed simultaneously.

3. Training and evaluation artifacts are stored by default in `lpr/vietnam_lp/model`.
   To visualize training and evaluation, run the `tensorboard` with the command below:

    ```bash
    tensorboard --logdir=./model
    ```

### Export to Pb™

To run the model , freeze the TensorFlow graph using the Model Optimizer:

```Bash
python3 tools/export.py --output_dir model/export --checkpoint ./model/model.ckpt-250000.ckpt
```
###  Convert to TFlite

```
$ cp ./model/export/frozen_graph/graph.pb.frozen ./model/export/frozen_graph/graph.pb 
- To show network model:
$ netron ./model/export/frozen_graph/graph.pb
- Run convert
tflite_convert \
--graph_def_file=model/export/frozen_graph/graph.pb \
--output_file=model/lprnet.tflite \
--output_format=TFLITE \
--input_arrays=input \
--input_shapes=1,24,94,3 \
--inference_type=FLOAT \
--output_arrays="Squeeze" \
--allow_custom_ops
```
    *Note: Because TFlite does not support conversion CTC function , so we should convert from Squeeze then process CTC in inferencing model.


## Demo

### With a TFlite

```Bash
python3 tools/infer_tflite.py  --model <path_model> --image <path_image> 

python3 tools/infer_tflite.py  --model /media/thanglmb/Bkav/AICAM/TrainModels/OCR4ANPR/lprnet/model/lprnet_14.tflite \
--image /media/thanglmb/Bkav/AICAM/TrainModels/OCR4ANPR/lprnet/imgs_test/3.png
```

### With a Frozen Graph

```Bash
python3 tools/infer.py --model model/export/frozen_graph/graph.pb.frozen \
    --config vietnam_lp/config.py \
    --image <image_path>
```


### With the Latest Checkpoint

> **NOTE**: Input data for inference should be set via the `infer.file_list_path` parameter in
`vietnam_lp/config.py` and must look like a text file
with a list of paths to license plates images in the following format:

```
path_to_lp_image1
path_to_lp_image2
...
```

When the training is complete, the model from the checkpoint can be inferred on the
input data by running `vietnam_lp/infer.py`:

```Bash
python3 tools/infer_checkpoint.py vietnam_lp/config.py
```
