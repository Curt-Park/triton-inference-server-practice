https://github.com/triton-inference-server/server/blob/main/docs/quickstart.md

## 1. Install Triton with Docker image (For GPUs)
Install [Docker](https://docs.docker.com/engine/install/) and Check out [the triton image version](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/tritonserver/tags)

```bash
$ docker pull nvcr.io/nvidia/tritonserver:22.02-py3
```

## 2. Create a model repository

```bash
$ git clone git@github.com:triton-inference-server/server.git
$ cd server/docs/examples
$ ./fetch_models.sh
$ cd model_repository
$ tree model_repository
model_repository
├── densenet_onnx
│   ├── 1
│   │   └── model.onnx
│   ├── config.pbtxt
│   └── densenet_labels.txt
├── inception_graphdef
│   ├── 1
│   │   └── model.graphdef
│   ├── config.pbtxt
│   └── inception_labels.txt
├── simple
│   ├── 1
│   │   └── model.graphdef
│   └── config.pbtxt
├── simple_dyna_sequence
│   ├── 1
│   │   └── model.graphdef
│   └── config.pbtxt
├── simple_identity
│   ├── 1
│   │   └── model.savedmodel
│   │       └── saved_model.pb
│   └── config.pbtxt
├── simple_int8
│   ├── 1
│   │   └── model.graphdef
│   └── config.pbtxt
├── simple_sequence
│   ├── 1
│   │   └── model.graphdef
│   └── config.pbtxt
└── simple_string
    ├── 1
    │   └── model.graphdef
    └── config.pbtxt
```

## 3. Run Triton

```bash
$ docker run --gpus 1 --rm -p 9000:8000 -p 9001:8001 -p 9002:8002 -v $(pwd)/model_repository:/models nvcr.io/nvidia/tritonserver:22.02-py3 tritonserver --model-repository=/models

=============================
== Triton Inference Server ==
=============================

NVIDIA Release 22.02 (build 32400308)

Copyright (c) 2018-2021, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.

Various files include modifications (c) NVIDIA CORPORATION.  All rights reserved.

...

I0227 14:11:47.677986 1 server.cc:549]
+-------------+-------------------------------------------------------------------------+--------+
| Backend     | Path                                                                    | Config |
+-------------+-------------------------------------------------------------------------+--------+
| pytorch     | /opt/tritonserver/backends/pytorch/libtriton_pytorch.so                 | {}     |
| tensorflow  | /opt/tritonserver/backends/tensorflow1/libtriton_tensorflow1.so         | {}     |
| onnxruntime | /opt/tritonserver/backends/onnxruntime/libtriton_onnxruntime.so         | {}     |
| openvino    | /opt/tritonserver/backends/openvino_2021_2/libtriton_openvino_2021_2.so | {}     |
+-------------+-------------------------------------------------------------------------+--------+

I0227 14:11:47.678130 1 server.cc:592]
+----------------------+---------+--------+
| Model                | Version | Status |
+----------------------+---------+--------+
| densenet_onnx        | 1       | READY  |
| inception_graphdef   | 1       | READY  |
| simple               | 1       | READY  |
| simple_dyna_sequence | 1       | READY  |
| simple_identity      | 1       | READY  |
| simple_int8          | 1       | READY  |
| simple_sequence      | 1       | READY  |
| simple_string        | 1       | READY  |
+----------------------+---------+--------+
...
I0227 14:11:47.752336 1 metrics.cc:623] Collecting metrics for GPU 0: NVIDIA TITAN RTX
I0227 14:11:47.752785 1 tritonserver.cc:1932]
...
+----------------------------------+-------------------------------------------------------------+
| Option                           | Value                                                       |
+----------------------------------+-------------------------------------------------------------+
| server_id                        | triton                                                      |
| server_version                   | 2.19.0                                                      |
| server_extensions                | classification sequence model_repository ...tatistics trace |
| model_repository_path[0]         | /models                                                     |
| model_control_mode               | MODE_NONE                                                   |
| strict_model_config              | 1                                                           |
| rate_limit                       | OFF                                                         |
| pinned_memory_pool_byte_size     | 268435456                                                   |
| cuda_memory_pool_byte_size{0}    | 67108864                                                    |
| response_cache_byte_size         | 0                                                           |
| min_supported_compute_capability | 6.0                                                         |
| strict_readiness                 | 1                                                           |
| exit_timeout                     | 30                                                          |
+----------------------------------+-------------------------------------------------------------+
I0227 14:11:47.754516 1 grpc_server.cc:4375] Started GRPCInferenceService at 0.0.0.0:8001
I0227 14:11:47.754858 1 http_server.cc:3075] Started HTTPService at 0.0.0.0:8000
I0227 14:11:47.797229 1 http_server.cc:178] Started Metrics Service at 0.0.0.0:8002
```

## Verify Triton is running correctly
```bash
$ curl -v localhost:9000/v2/health/ready

*   Trying 127.0.0.1:9000...
*   * Connected to localhost (127.0.0.1) port 9000 (#0)
*   > GET /v2/health/ready HTTP/1.1
*   > Host: localhost:9000
*   > User-Agent: curl/7.78.0
*   > Accept: */*
*   >
*   * Mark bundle as not supporting multiuse
*   < HTTP/1.1 200 OK
*   < Content-Length: 0
*   < Content-Type: text/plain
*   <
*   * Connection #0 to host localhost left intact
```

## Getting the client exmaples
```bash
$ docker pull nvcr.io/nvidia/tritonserver:22.02-py3-sdk
$ docker run -it --rm --net=host nvcr.io/nvidia/tritonserver:22.02-py3-sdk

=================================
== Triton Inference Server SDK ==
=================================

NVIDIA Release 22.02 (build 32400314)
...

$ /workspace/install/bin/image_client -m densenet_onnx -c 3 -s INCEPTION /workspace/images/mug.jpg -u localhost:9000

Request 0, batch size 1
Image '/workspace/images/mug.jpg':
15.349564 (504) = COFFEE MUG
13.227462 (968) = CUP
10.424895 (505) = COFFEEPOT

$ install/bin/image_client --help
install/bin/image_client: unrecognized option '--help'
Usage: install/bin/image_client [options] <image filename / image folder>
    Note that image folder should only contain image files.
        -v
        -a
        --streaming
        -b <batch size>
        -c <topk>  # `-c 3` means asking for top 3 classification
        -s <NONE|INCEPTION|VGG>  # Preprocessing
        -p <proprocessed output filename>
        -m <model name>
        -x <model version>
        -u <URL for inference service>
        -i <Protocol used to communicate with inference service>
        -H <HTTP header>
```

## Add a TorchScript model for MNIST classification

#### Initialize python environment to train a MNIST CNN model
```bash
$ source init.sh
```

#### Train a CNN model
```bash
$ python train.py  # this saves the best model as "model.pt"
```

#### Create a directory for the model in `model_repository`, and move `model.pt` and `confing.pbtxt` with the following structure
```bash
# In model_repository
mnist_cnn
├── 1
│   └── model.pt  # the trained model
└── config.pbtxt
```
See [Model Repository](https://github.com/triton-inference-server/server/blob/main/docs/model_repository.md)
and [Model Configuration](https://github.com/triton-inference-server/server/blob/main/docs/model_configuration.md)
for more information about the file structure and `config.pbtxt`.

#### Run Triton again
```bash
$ docker run --gpus 1 --rm -p 9000:8000 -p 9001:8001 -p 9002:8002 -v $(pwd)/model_repository:/models nvcr.io/nvidia/tritonserver:22.02-py3 tritonserver --model-repository=/models

...
I0227 15:43:39.729784 1 server.cc:592]
+----------------------+---------+--------+
| Model                | Version | Status |
+----------------------+---------+--------+
| densenet_onnx        | 1       | READY  |
| inception_graphdef   | 1       | READY  |
| mnist_cnn            | 1       | READY  |
| simple               | 1       | READY  |
| simple_dyna_sequence | 1       | READY  |
| simple_identity      | 1       | READY  |
| simple_int8          | 1       | READY  |
| simple_sequence      | 1       | READY  |
| simple_string        | 1       | READY  |
+----------------------+---------+--------+
...
```

Check the model configuration.
```bash
$ curl localhost:9000/v2/models/mnist_cnn/config

{"name":"mnist_cnn","platform":"pytorch_libtorch","backend":"pytorch","version_policy":{"latest":{"num_versions":1}},"max_batch_size":100,"input":[{"name":"input__0","data_type":"TYPE_FP32","format":"FORMAT_NONE","dims":[28,28,1],"is_shape_tensor":false,"allow_ragged_batch":false,"optional":false}],"output":[{"name":"output__0","data_type":"TYPE_FP32","dims":[10],"label_filename":"","is_shape_tensor":false}],"batch_input":[],"batch_output":[],"optimization":{"priority":"PRIORITY_DEFAULT","input_pinned_memory":{"enable":true},"output_pinned_memory":{"enable":true},"gather_kernel_buffer_threshold":0,"eager_batching":false},"dynamic_batching":{"preferred_batch_size":[50],"max_queue_delay_microseconds":0,"preserve_ordering":false,"priority_levels":0,"default_priority_level":0,"priority_queue_policy":{}},"instance_group":[{"name":"mnist_cnn_0","kind":"KIND_GPU","count":1,"gpus":[0],"secondary_devices":[],"profile":[],"passive":false,"host_policy":""}],"default_model_filename":"model.pt","cc_model_filenames":{},"metric_tags":{},"parameters":{},"model_warmup":[]}
```

#### Install Triton Client

```bash
$ sudo apt update
$ sudo apt install libb64-dev
$ pip install "tritonclient[http]"
```

#### Run the model inference
Let's run inference on `mnist_sample.jpg`.

```bash
$ python infer.py

Shape: (1, 1, 28, 28) , Type: float32

{'model_name': 'mnist_cnn', 'model_version': '1', 'outputs': [{'name': 'output__0', 'datatype': 'FP32', 'shape': [1, 10], 'data': [-26.26696014404297, -22.597688674926758, -28.230932235717773, -6.869289875030518, -29.643720626831055, -0.0010399178136140108, -23.8032169342041, -23.27438735961914, -20.790000915527344, -16.399457931518555]}]}

Prediction Result: 5

{'model_stats': [{'name': 'mnist_cnn', 'version': '1', 'last_inference': 1645981740843, 'inference_count': 11, 'execution_count': 11, 'inference_stats': {'success': {'count': 11, 'ns': 1005967579}, 'fail': {'count': 0, 'ns': 0}, 'queue': {'count': 11, 'ns': 1994213}, 'compute_input': {'count': 11, 'ns': 122481903}, 'compute_infer': {'count': 11, 'ns': 880749163}, 'compute_output': {'count': 11, 'ns': 102340}, 'cache_hit': {'count': 0, 'ns': 0}}, 'batch_stats': [{'batch_size': 1,
'compute_input': {'count': 11, 'ns': 122481903}, 'compute_infer': {'count': 11, 'ns': 880749163}, 'compute_output': {'count': 11, 'ns': 102340}}]}]}
```

## References
- https://github.com/triton-inference-server/server/blob/main/docs/quickstart.md
- https://github.com/triton-inference-server/server/blob/main/docs/model_repository.md
- https://github.com/triton-inference-server/server/blob/main/docs/model_configuration.md
- https://github.com/triton-inference-server/backend/blob/main/README.md#backends
- https://github.com/triton-inference-server/server/blob/main/docs/model_configuration.md#auto-generated-model-configuration
- https://github.com/triton-inference-server/client/blob/main/src/python/examples/simple_http_infer_client.py
