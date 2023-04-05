package precise

import (
	"github.com/yut-kt/gomfcc"
	"gorgonia.org/tensor"
)

var (
	int16Divider = 1.0 / 32768.0
)

// int16ToFloatSlice Converts an int16 slice to float64 audio
func int16ToFloatSlice(input []int16) []float64 {
	output := make([]float64, len(input))

	for i, s := range input {
		output[i] = float64(s) * int16Divider
	}

	return output
}

func mfccSpec(audio []int16, params Params) *tensor.Dense {
	mfcc := gomfcc.NewGoMFCC(int16ToFloatSlice(audio), params.SampleRate, gomfcc.LowFrequency(0))

	// window_t and hop_t are in fractions of a second, while we need milliseconds
	features := mfcc.GetFeatureByMS(params.NMFCC, params.NFilt, float64(params.WindowT)*1000, float64(params.HopT)*1000)

	backing := make([]float32, len(features)*len(features[0]))

	// Create a backing slice of float32s for the Tensor
	for i, featureList := range features {
		for index := 0; index < len(featureList); index++ {
			backing[i*len(featureList)+index] = float32(featureList[index])
		}
	}

	return tensor.New(tensor.Of(tensor.Float32), tensor.WithShape(len(features), len(features[0])), tensor.WithBacking(backing))
}
