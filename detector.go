package precise

import "math"

type TriggerOption func(*TriggerDetector)

// WithSensitivity sets the detector sensitivity
func WithSensitivity(sensitivity float32) TriggerOption {
	return func(t *TriggerDetector) {
		t.sensitivity = sensitivity
	}
}

// WithTriggerLevel sets the number of triggers required to return a
// valid trigger
func WithTriggerLevel(triggerLevel int) TriggerOption {
	return func(t *TriggerDetector) {
		t.triggerLevel = triggerLevel
	}
}

// NewTriggerDetector creates a new TriggerDetector
func NewTriggerDetector(chunkSize int, opts ...TriggerOption) *TriggerDetector {
	t := &TriggerDetector{
		chunkSize:    chunkSize,
		sensitivity:  0.5,
		triggerLevel: 3,
	}

	for _, opt := range opts {
		opt(t)
	}

	return t
}

type TriggerDetector struct {
	chunkSize    int
	sensitivity  float32
	triggerLevel int
	activation   int
}

// Update adds the current probability to the detection history
func (t *TriggerDetector) Update(prob float32) bool {
	chunkActivated := prob > 1.0-t.sensitivity

	if chunkActivated || t.activation < 0 {
		t.activation += 1

		hasActivated := t.activation > t.triggerLevel

		if hasActivated || chunkActivated && t.activation < 0 {
			t.activation = int(math.Floor(float64(-(8 * 2048) / t.chunkSize)))
		}

		if hasActivated {
			return true
		}
	} else if t.activation > 0 {
		t.activation -= 1
	}

	return false
}
