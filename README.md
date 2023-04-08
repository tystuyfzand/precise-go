Precise-Lite for Go
===================

This is a Go port of the [MycroftAI Precise](https://github.com/MycroftAI/mycroft-precise) engine, using tflite and onnxruntime 
(based on the [OpenVoiceOS fork](https://github.com/OpenVoiceOS/precise-lite)).

I needed this for my Discord bot [Astra](https://astra.bot) - which I'm currently training a model for. Help us out [here](https://train.astra.bot/) and submit a few voice clips!


What is working?
----------------

Everything should work, though I'm not sure if my port is off in any way. I'm not an AI/Machine Learning expert, I just read the code!

Supported Backends
------------------

Currently, I've tested it with tflite and onnxruntime (cpu and gpu) - onnxruntime works and would be fine on CPU, but on
GPUs it is slower than CPU due to the simplicity of the model.

Example
-------

See `runner_test.go` - this contains an example usage of both tflite and onnxruntime - though onnxruntime is slower,
it was more of a test.

Docker
------

The Dockerfiles contain a pre-built Go image with all dependencies required to use this library. If you want to use it
yourself, you can use a FROM statement for `ghcr.io/tystuyfzand/precise-go` public image, use it to build, and then copy the tflite
runtime. These images are based off Ubuntu 22.04 (due to simplicity and unifying the GPU and non-GPU, but debian exists too)

The following tags are available, and all include Tensorflow 2.11.0 and ONNX 1.12.0

- latest, ubuntu-jammy (Ubuntu 22.04)
- debian (Debian 11)
- latest-gpu, ubuntu-jammy-gpu (Ubuntu 22.04, CUDA 11.8.0, CUDNN 8)

To use this in Docker, tensorflow doesn't exactly have a precompiled lite version that I've found. The easiest way is to
use the base docker image `ghcr.io/tystuyfzand/precise-go` (which contains the latest version of Go as of this)
and use it to build your application with a multi-stage build. Then, add the tensorflow lite libraries.

Example:

```dockerfile
FROM ghcr.io/tystuyfzand/precise-go:latest AS builder

ADD . /app
WORKDIR /app

RUN go build -o main

FROM ubuntu:jammy

COPY --from=builder /app/main /main
COPY --from=builder /usr/lib/libtensorflowlite.so /usr/lib/libtensorflowlite_c.so /usr/lib/

CMD ["/main"]
```

There is also support for onnxruntime (using `go build -tags onnx`) - but it's not necessary.

Note: GPU/CUDA support exists, but is actually SLOWER than CPU support due to how simple the model is. When using this, you're required to use single stage builds OR use the same base CUDA image.

Contributions, Bug Reports, etc.
--------------------------------

All of these are welcome! If I made an error in the logic or there's something wrong, please let me know!