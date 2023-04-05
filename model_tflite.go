package precise

import (
	"errors"
	"github.com/mattn/go-tflite"
	"gorgonia.org/tensor"
	"sync"
)

// NewTFLiteModel creates a new tensorflow lite model
func NewTFLiteModel(modelPath string) (Model, error) {
	model := tflite.NewModelFromFile(modelPath)

	if model == nil {
		return nil, errors.New("cannot load model")
	}

	options := tflite.NewInterpreterOptions()

	interpreter := tflite.NewInterpreter(model, options)

	interpreter.AllocateTensors()

	return &TFLiteModel{
		model:       model,
		interpreter: interpreter,
		options:     options,
		lock:        new(sync.Mutex),
	}, nil
}

// TFLiteModel represents a tensorflow lite model
type TFLiteModel struct {
	model       *tflite.Model
	options     *tflite.InterpreterOptions
	interpreter *tflite.Interpreter
	lock        *sync.Mutex
}

// Predict sends the input data into the input tensor, then invokes the model
func (m *TFLiteModel) Predict(inputData tensor.Tensor) (float32, error) {
	if m.model == nil {
		return -1, ErrModelClosed
	}

	input := m.interpreter.GetInputTensor(0)

	copy(input.Float32s(), inputData.Data().([]float32))

	m.interpreter.Invoke()

	output := m.interpreter.GetOutputTensor(0)

	if output.Type() != tflite.Float32 {
		return -1, ErrUnexpectedType
	}

	return output.Float32s()[0], nil
}

// Close cleans up the model after use
func (m *TFLiteModel) Close() error {
	m.lock.Lock()
	defer m.lock.Unlock()

	if m.model == nil {
		return nil
	}

	m.model.Delete()
	m.options.Delete()
	m.interpreter.Delete()

	m.model = nil
	return nil
}
