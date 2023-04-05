package precise

import "math"

// Params is the Precise model parameter list.
// This can be loaded from JSON, however for tflite models these are hardcoded.
// Defaults are set for these elsewhere via NewParams, not using the tags,
// but they are there for reference
type Params struct {
	WindowT         float32 `json:"window_t" default:"0.1"`
	HopT            float32 `json:"hop_t" default:"0.05"`
	BufferT         float32 `json:"buffer_t" default:"1.5"`
	SampleRate      int     `json:"sample_rate" default:"16000"`
	SampleDepth     int     `json:"sample_depth" default:"2"`
	NMFCC           int     `json:"n_mfcc" default:"13"`
	NFilt           int     `json:"n_filt" default:"20"`
	NFft            int     `json:"n_fft" default:"512"`
	UseDelta        bool    `json:"use_delta" default:"false"`
	ThresholdConfig MuStd   `json:"threshold_config" default:"[[6,4]]"`
	ThresholdCenter float32 `json:"threshold_center" default:"0.2"`
}

func NewParams() Params {
	return Params{
		WindowT:         0.1,
		HopT:            0.05,
		BufferT:         1.5,
		SampleRate:      16000,
		SampleDepth:     2,
		NMFCC:           13,
		NFilt:           20,
		NFft:            512,
		ThresholdConfig: MuStd{{6, 4}},
		ThresholdCenter: 0.2,
	}
}

func (p Params) BufferSamples() int {
	samples := int(float32(p.SampleRate)*p.BufferT + 0.5)
	return p.HopSamples() * int(math.Floor(float64(samples/p.HopSamples())))
}

func (p Params) NFeatures() int {
	return 1 + int(math.Floor(float64(p.BufferSamples()-p.WindowSamples())/float64(p.HopSamples())))
}

func (p Params) WindowSamples() int {
	return int(float32(p.SampleRate)*p.WindowT + 0.5)
}

func (p Params) HopSamples() int {
	return int(float32(p.SampleRate)*p.HopT + 0.5)
}
