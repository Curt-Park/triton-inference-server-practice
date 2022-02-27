"""Infer a handwritten digit image.

This script is a modified from:
    https://github.com/triton-inference-server/client/blob/main/src/python/examples/simple_http_infer_client.py

- Author: Jinwoo Park
- Email: www.jwpark.co.kr@gmail.com
"""

import argparse

import numpy as np
import tritonclient.http as httpclient
from PIL import Image

from dataloader import get_preprocessor

# args
parser = argparse.ArgumentParser()
parser.add_argument(
    "-v",
    "--verbose",
    action="store_true",
    required=False,
    default=False,
    help="Enable verbose output",
)
parser.add_argument(
    "-u",
    "--url",
    type=str,
    required=False,
    default="localhost:9000",
    help="Inference server URL. Default is localhost:9000.",
)
FLAGS = parser.parse_args()

# triton client
triton_client = httpclient.InferenceServerClient(url=FLAGS.url, verbose=FLAGS.verbose)

# read an image
preprocess = get_preprocessor()
image = Image.open("mnist_sample.jpg")
input0 = preprocess(image).numpy()
input0 = np.expand_dims(input0, axis=0)
print("Shape:", input0.shape, ", Type:", input0.dtype)
print()

# define inputs and outputs
inputs = []
outputs = []
inputs.append(httpclient.InferInput("input__0", [1, 1, 28, 28], "FP32"))
inputs[0].set_data_from_numpy(input0, binary_data=False)
outputs.append(httpclient.InferRequestedOutput("output__0", binary_data=False))

# infer
model_name = "mnist_cnn"
results = triton_client.infer(model_name, inputs, outputs=outputs)
res = results.get_response()
print(res)
print()
output0 = results.as_numpy("output__0")
prediction = output0.argmax()
print("Prediction Result:", prediction)
print()

# statistics
statistics = triton_client.get_inference_statistics(model_name=model_name)
print(statistics)
