ARG BASE_IMAGE=ubuntu:jammy
FROM $BASE_IMAGE AS builder

LABEL org.opencontainers.image.description="Base docker image with tensorflow lite, onnxruntime, and Go for precise-go"

ARG GO_VERSION=1.20.3

RUN apt-get update && \
    apt-get -y install git build-essential curl ca-certificates && \
    rm -rf /var/lib/apt/lists/*

# Even though we have a Golang image, we can just use our own.
# This lets us unify our GPU and Standard images.
RUN curl -L -o /usr/local/go${GO_VERSION}.tar.gz https://go.dev/dl/go${GO_VERSION}.linux-amd64.tar.gz && \
    tar -C /usr/local -xzf /usr/local/go${GO_VERSION}.tar.gz && \
    rm -f /usr/local/go${GO_VERSION}.tar.gz

ENV PATH=$PATH:/usr/local/go/bin

# Re-use our prebuilt tflite (for all builds)
COPY --from=ghcr.io/tystuyfzand/precise-go:tflite /usr/include/tf-include.tar.gz /usr/include
COPY --from=ghcr.io/tystuyfzand/precise-go:tflite /usr/lib/libtensorflowlite.so /usr/lib/libtensorflowlite_c.so /usr/lib/

# Extract the archive (done to ensure we have all that we need in one spot - bazel puts it into a "cache" folder that changes)
RUN tar -C /usr/include -xvf /usr/include/tf-include.tar.gz

ENV CGO_CFLAGS="-I/usr/src/tensorflow"
ENV CGO_LDFLAGS="-L/usr/src/tensorflow/bazel-bin/tensorflow/lite"

ARG ONNX_RUNTIME_VERSION=1.12.0

RUN FILE="onnxruntime-linux-x64-gpu-$ONNX_RUNTIME_VERSION.tgz" && \
    URL="https://github.com/microsoft/onnxruntime/releases/download/v$ONNX_RUNTIME_VERSION/onnxruntime-linux-x64-$ONNX_RUNTIME_VERSION.tgz" && \
    curl -L -o $FILE $URL && \
    tar xvf $FILE -C /usr/local/ --strip-components=1 && \
    rm -f $FILE

ENV LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/lib