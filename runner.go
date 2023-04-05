package precise

import (
	"io"
)

type ActivationFunc func()

type PredictionFunc func(prob float32)

type ExitFunc func(err error)

type Option func(*Runner)

// WithDetectorOpts sets detector options
func WithDetectorOpts(opts ...TriggerOption) Option {
	return func(r *Runner) {
		r.detectorOpts = opts
	}
}

// WithActivationFunc sets the func called when activated
func WithActivationFunc(f ActivationFunc) Option {
	return func(r *Runner) {
		r.OnActivation = f
	}
}

// WithPredictionFunc sets the func called after prediction
func WithPredictionFunc(f PredictionFunc) Option {
	return func(r *Runner) {
		r.OnPrediction = f
	}
}

// WithPredictionFunc sets the func called after prediction
func WithExitFunc(f ExitFunc) Option {
	return func(r *Runner) {
		r.OnExit = f
	}
}

// NewRunner creates a new network runner
func NewRunner(listener *Listener, chunkSize int, opts ...Option) *Runner {
	r := &Runner{
		listener:  listener,
		chunkSize: chunkSize,
		sampleCh:  make(chan []int16),
		closeCh:   make(chan bool),
	}

	for _, opt := range opts {
		opt(r)
	}

	r.detector = NewTriggerDetector(chunkSize, r.detectorOpts...)

	r.Start()

	return r
}

type Runner struct {
	listener     *Listener
	detector     *TriggerDetector
	detectorOpts []TriggerOption
	chunkSize    int
	running      bool
	sampleCh     chan []int16
	closeCh      chan bool

	OnPrediction PredictionFunc
	OnActivation ActivationFunc
	OnExit       ExitFunc
}

// Start will start the runner and the goroutine
func (r *Runner) Start() {
	if r.running {
		return
	}

	r.running = true

	go r.handlePredictions()
}

// Stop will stop the runner without closing it.
func (r *Runner) Stop() {
	if !r.running {
		return
	}

	r.running = false
}

// Close stops the neural network runner
func (r *Runner) Close() error {
	r.Stop()

	close(r.closeCh)

	if r.listener != nil {
		return r.listener.Close()
	}

	return nil
}

// Write allows a Runner to act as an io.Writer
func (r *Runner) Write(b []byte) (int, error) {
	samples := bytesToSamples(b)

	r.sampleCh <- samples

	return len(b) % 2, nil
}

// Queue passes in samples directly to the channel
func (r *Runner) Queue(samples []int16) {
	r.sampleCh <- samples
}

// ReadFrom allows the Runner to simply read from a reader
func (r *Runner) ReadFrom(reader io.Reader) (int64, error) {
	chunkSize := r.chunkSize

	if chunkSize == 0 || chunkSize == -1 {
		chunkSize = 2048
	}

	buf := make([]byte, chunkSize)

	var total int64

	for {
		read, err := reader.Read(buf)

		if err == io.EOF {
			break
		} else if err != nil {
			return -1, err
		}

		total += int64(read)

		r.sampleCh <- bytesToSamples(buf[:read])
	}

	return total, nil
}

// handlePredictions is a constantly running goroutine to read samples from our chan
func (r *Runner) handlePredictions() {
	var prob float32
	var err error

loop:
	for r.running {
		select {
		case samples, ok := <-r.sampleCh:
			if !ok {
				break loop
			}
			prob, err = r.listener.Update(samples)

			if err != nil {
				break
			}

			if r.OnPrediction != nil {
				r.OnPrediction(prob)
			}

			if r.detector.Update(prob) {
				r.OnActivation()
			}
		case <-r.closeCh:
			break loop
		}
	}

	r.running = false

	if r.OnExit != nil {
		r.OnExit(err)
	}
}

// bytesToSamples converts bytes to 16-bit samples
func bytesToSamples(b []byte) []int16 {
	readable := len(b) / 2

	samples := make([]int16, readable)

	for i := 0; i < readable; i++ {
		samples[i] = int16(b[i*2]) + int16(b[i*2+1])<<8
	}

	return samples
}
