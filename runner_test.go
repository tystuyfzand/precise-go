package precise

import (
	"fmt"
	"github.com/cryptix/wav"
	"os"
	"testing"
)

func TestNewRunner(t *testing.T) {
	testRunModel(t, "out.wav")
}

func testRunModel(t *testing.T, inputFile string) {
	model, err := NewONNXModel("astra.onnx", OnnxCUDA)

	if err != nil {
		t.Fatal("Unable to load model")
	}

	p := NewParams()

	l, err := NewListener(model, p)

	t.Log("Testing file", inputFile)

	f, err := os.Open(inputFile)

	if err != nil {
		t.Fatal(err)
	}

	stat, err := f.Stat()

	if err != nil {
		t.Fatal(err)
	}

	wr, err := wav.NewReader(f, stat.Size())

	if err != nil {
		t.Fatal(err)
	}

	wdr, err := wr.GetDumbReader()

	if err != nil {
		t.Fatal(err)
	}

	activated := false

	ch := make(chan struct{})

	t.Log("Setting up runner")

	var runner *Runner

	opts := []Option{
		WithActivationFunc(func() {
			activated = true
		}),
		WithExitFunc(func(err error) {
			close(ch)

			runner.Close()
		}),
		WithDetectorOpts(WithSensitivity(0.8)),
	}

	runner = NewRunner(l, -1, opts...)

	t.Log("Reading data")

	read, err := runner.ReadFrom(wdr)

	if err != nil {
		t.Fatal("Unable to read wav data", err)
	} else {
		t.Log("Successfully read", read, "bytes")
	}

	runner.Stop()

	<-ch

	if activated {
		t.Log("Sample activated")
	} else {
		t.Log("No activation found")
	}
}

var benchResult float32

func BenchmarkTFLiteRunner(b *testing.B) {
	model, err := NewTFLiteModel("astra.tflite")

	if err != nil {
		b.Fatal("Unable to load model")
	}

	p := NewParams()

	l, err := NewListener(model, p)

	defer l.Close()

	samples, err := loadSamples("out.wav")

	if err != nil {
		b.Fatal(err)
	}

	mfccs := l.updateVectors(samples)

	var val float32

	for i := 0; i < b.N; i++ {
		val, err = l.model.Predict(mfccs)

		if err != nil {
			b.Fatal(err)
		}
	}

	benchResult = val
}

func BenchmarkOnnxRunner(b *testing.B) {
	for t := OnnxCPU; t <= OnnxCUDA; t++ {
		b.Run(fmt.Sprintf("onnx_%s", t.String()), func(b *testing.B) {
			b.StopTimer()
			model, err := NewONNXModel("astra.onnx", t)

			if err != nil {
				b.Fatal("Unable to load model")
			}

			p := NewParams()

			l, err := NewListener(model, p)

			defer l.Close()

			samples, err := loadSamples("out.wav")

			if err != nil {
				b.Fatal(err)
			}

			mfccs := l.updateVectors(samples)

			var val float32

			b.StartTimer()

			for i := 0; i < b.N; i++ {
				val, err = l.model.Predict(mfccs)

				if err != nil {
					b.Fatal(err)
				}
			}

			benchResult = val
		})
	}
}

func loadSamples(inputFile string) ([]int16, error) {
	f, err := os.Open(inputFile)

	if err != nil {
		return nil, err
	}

	defer f.Close()

	stat, err := f.Stat()

	if err != nil {
		return nil, err
	}

	wr, err := wav.NewReader(f, stat.Size())

	if err != nil {
		return nil, err
	}

	samples := make([]int16, wr.GetSampleCount())

	for i := 0; i < len(samples); i++ {
		sample, err := wr.ReadSample()

		if err != nil {
			return nil, err
		}

		samples[i] = int16(sample)
	}

	return samples, nil
}
