package precise

import (
	"errors"
	"gorgonia.org/tensor"
)

var (
	ErrUnexpectedType = errors.New("unexpected tensor type")
)

type Model interface {
	Predict(inputData tensor.Tensor) (float32, error)
	Close() error
}
