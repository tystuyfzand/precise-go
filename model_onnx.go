//go:build onnx

package precise

import (
	"context"
	"errors"
	"github.com/ivansuteja96/go-onnxruntime"
	"gorgonia.org/tensor"
)

type DeviceType int

const (
	OnnxCPU      DeviceType = 1
	OnnxCUDA     DeviceType = 2
	OnnxTensorRT DeviceType = 3
)

func (t DeviceType) String() string {
	switch t {
	case OnnxCPU:
		return "CPU"
	case OnnxCUDA:
		return "CUDA"
	case OnnxTensorRT:
		return "TensorRT"
	}
	return ""
}

// NewONNXModel creates a new onnx model
func NewONNXModel(modelPath string, deviceType DeviceType) (Model, error) {
	ortEnvDet := onnxruntime.NewORTEnv(onnxruntime.ORT_LOGGING_LEVEL_ERROR, "development")
	ortDetSO := onnxruntime.NewORTSessionOptions()

	switch deviceType {
	case OnnxTensorRT:
		ortDetSO.AppendExecutionProviderTensorRT(onnxruntime.TensorRTOptions{
			DeviceID:               0,
			MaxWorkspaceSize:       1 * 1024 * 1024 * 1024,
			MaxPartitionIterations: 1000,
			MinSubgraphSize:        5,
		})
		fallthrough
	case OnnxCUDA:
		ortDetSO.AppendExecutionProviderCUDA(onnxruntime.CudaOptions{
			DeviceID:              0,
			GPUMemorylimit:        2 * 1024 * 1024 * 1024,
			DoCopyInDefaultStream: true,
		})
	}

	model, err := onnxruntime.NewORTSession(ortEnvDet, modelPath, ortDetSO)

	if err != nil {
		return nil, err
	}

	return &ONNXModel{
		model: model,
	}, nil
}

// ONNXModel represents a tensorflow lite model
type ONNXModel struct {
	model *onnxruntime.ORTSession
	ctx   context.Context
}

// Predict sends the input data into the input tensor, then invokes the model
func (m *ONNXModel) Predict(inputData tensor.Tensor) (float32, error) {
	if m.model == nil {
		return -1, ErrModelClosed
	}

	data := inputData.Data().([]float32)

	shape := inputData.Shape()

	res, err := m.model.Predict([]onnxruntime.TensorValue{
		{
			Value: data,
			// Shape is an interesting one
			// We're using 1 input tensor, with shape 29, 13
			Shape: []int64{1, int64(shape[0]), int64(shape[1])},
		},
	})

	if err != nil {
		return -1, err
	}

	outputVals := res[0].Value

	if v, ok := outputVals.([]float32); ok {
		return v[0], nil
	}

	return -1, errors.New("unexpected output value type")
}

// Close cleans up the model after use
func (m *ONNXModel) Close() error {
	m.model = nil
	return nil
}
