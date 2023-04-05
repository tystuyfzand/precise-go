package precise

import (
	"errors"
	"gorgonia.org/tensor"
)

var (
	ErrModelClosed = errors.New("model closed")
)

func NewListener(model Model, p Params) (*Listener, error) {
	l := &Listener{
		params:      p,
		model:       model,
		windowAudio: make([]int16, 0),
		mfccs:       tensor.New(tensor.Of(tensor.Float32), tensor.WithShape(p.NFeatures(), p.NMFCC)),
	}

	config := DefaultThreshold
	config.Center = p.ThresholdCenter
	l.decoder = NewThresholdDecoder(p.ThresholdConfig, config)

	return l, nil
}

type Listener struct {
	params      Params
	model       Model
	windowAudio []int16
	mfccs       *tensor.Dense
	decoder     *ThresholdDecoder
}

func (p *Listener) updateVectors(audio []int16) tensor.Tensor {
	p.windowAudio = append(p.windowAudio, audio...)

	if len(p.windowAudio) >= p.params.WindowSamples() {
		var newFeatures tensor.Tensor
		newFeatures = mfccSpec(p.windowAudio, p.params)

		p.windowAudio = p.windowAudio[newFeatures.Shape()[0]*p.params.HopSamples():]

		if newFeatures.Shape()[0] > p.mfccs.Shape()[0] {
			// TODO: Slice only necessary features, this is done in Rust as:
			// nrows = len, dim = mfccs.Dim()?
			// new_features = new_features.slice(s![new_features.nrows() - self.mfccs.dim().0..,..]).to_owned();
			newFeatures = tensorMust(newFeatures.Slice(tensor.S(newFeatures.Shape()[0]-p.mfccs.Shape()[0], newFeatures.Shape()[0])))
		}

		// TODO: Concatenate mfccs
		// Rust example:
		// Axis = dimension? (0, top level)
		// nrows = len?
		// self.mfccs = concatenate![Axis(0), self.mfccs.slice(s![new_features.nrows()..,..]).to_owned(), new_features];
		// self.mfccs = np.concatenate((self.mfccs[len(new_features):], new_features))

		if newFeatures.Shape()[0] == p.mfccs.Shape()[0] {
			p.mfccs = newFeatures.(*tensor.Dense)
		} else {
			slicedMfccs := tensorMust(p.mfccs.Slice(tensor.S(newFeatures.Shape()[0], p.mfccs.Shape()[0])))

			p.mfccs = tensorMust(tensor.Concat(0, slicedMfccs, newFeatures.(*tensor.Dense)))
		}
	}

	return p.mfccs
}

func (p *Listener) Update(audio []int16) (float32, error) {
	if p.model == nil {
		return -1, ErrModelClosed
	}

	mfccs := p.updateVectors(audio)

	rawOutput, err := p.model.Predict(mfccs)

	if err != nil {
		return -1, err
	}

	return p.decoder.Decode(rawOutput), nil
}

func (p *Listener) Close() error {
	err := p.model.Close()

	if err != nil {
		return err
	}

	p.model = nil

	return nil
}
