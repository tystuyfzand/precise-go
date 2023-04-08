ARG BASE_IMAGE=ubuntu:22.04
FROM $BASE_IMAGE AS builder

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

ARG TENSORFLOW_VERSION=2.11.0

RUN git clone https://github.com/tensorflow/tensorflow.git /usr/src/tensorflow && \
    cd /usr/src/tensorflow && \
    git checkout v$TENSORFLOW_VERSION

RUN \
    # Install bazel (https://docs.bazel.build/versions/master/install-ubuntu.html) \
    mkdir -p /etc/apt/keyrings || true && \
    echo "deb [arch=amd64 signed-by=/etc/apt/keyrings/bazel.gpg] http://storage.googleapis.com/bazel-apt stable jdk1.8" | tee /etc/apt/sources.list.d/bazel.list && \
    curl -fsSL https://bazel.build/bazel-release.pub.gpg | tee | gpg --dearmor | tee /etc/apt/keyrings/bazel.gpg > /dev/null && \
    apt-get update && \
    apt-get -y install bazel-5.4.0 && \
    apt-get -y upgrade bazel-5.4.0 && \
    ln -s /usr/bin/bazel-5.4.0 /usr/bin/bazel && \
    # Unpack bazel for future use.
    bazel version

RUN apt-get install -y python3 python3-pip && \
    pip3 install numpy

RUN cd /usr/src/tensorflow && \
    ./configure && \
    bazel build --config opt --config monolithic //tensorflow/lite:libtensorflowlite.so && \
    bazel build --config opt --config monolithic //tensorflow/lite/c:libtensorflowlite_c.so

# These two files can be copied out into your resulting final image
RUN cp /usr/src/tensorflow/bazel-bin/tensorflow/lite/libtensorflowlite.so /usr/lib && \
    cp /usr/src/tensorflow/bazel-bin/tensorflow/lite/c/libtensorflowlite_c.so /usr/lib

ENV CGO_CFLAGS="-I/usr/src/tensorflow"
ENV CGO_LDFLAGS="-L/usr/src/tensorflow/bazel-bin/tensorflow/lite"

ARG ONNX_RUNTIME_VERSION=1.12.0

RUN FILE="onnxruntime-linux-x64-gpu-$ONNX_RUNTIME_VERSION.tgz" && \
    URL="https://github.com/microsoft/onnxruntime/releases/download/v$ONNX_RUNTIME_VERSION/onnxruntime-linux-x64-$ONNX_RUNTIME_VERSION.tgz" && \
    curl -L -o $FILE $URL && \
    tar xvf $FILE -C /usr/local/ --strip-components=1 && \
    rm -f $FILE

ENV LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/lib