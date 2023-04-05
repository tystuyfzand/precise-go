package precise

import "testing"

func TestNewThresholdDecoder(t *testing.T) {
	td := NewThresholdDecoder(MuStd{{6, 2}}, DefaultThreshold)

	t.Log("Decoded value:", td.Decode(0.9585))
}
