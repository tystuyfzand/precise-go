package precise

import (
	"gorgonia.org/tensor"
	"math"
	"sort"
)

// MuStd represents threshold configs with easy to use functions.
type MuStd [][]float32

// Min calculates the minimum with minZ, then sorts and returns the lowest value
func (m MuStd) Min(minZ float32) float32 {
	vals := m.Calculate(minZ)

	sort.Slice(vals, func(i, j int) bool {
		return vals[i] < vals[j]
	})

	return vals[0]
}

// Max calculates the maximum with maxZ, then sorts and returns the highest value
func (m MuStd) Max(maxZ float32) float32 {
	vals := m.Calculate(maxZ)

	sort.Slice(vals, func(i, j int) bool {
		return vals[i] > vals[j]
	})

	return vals[0]
}

// Calculate applies the value to the configuration
func (m MuStd) Calculate(value float32) []float32 {
	out := make([]float32, len(m))

	for i := 0; i < len(m); i++ {
		out[i] = m[i][0] + value*m[i][1]
	}

	return out
}

type ThresholdOptions struct {
	MinZ       float32
	MaxZ       float32
	Center     float32
	Resolution int
}

// DefaultThreshold is the default set of options for the decoder.
var DefaultThreshold = ThresholdOptions{
	MinZ:       -4.0,
	MaxZ:       4.0,
	Center:     0.5,
	Resolution: 200,
}

// NewThresholdDecoder creates a new ThresholdDecoder with calculated distributions.
func NewThresholdDecoder(muStds MuStd, options ThresholdOptions) *ThresholdDecoder {
	min := int(muStds.Min(options.MinZ))
	max := int(muStds.Max(options.MaxZ))

	td := &ThresholdDecoder{
		options:  options,
		MinOut:   min,
		MaxOut:   max,
		OutRange: max - min,
	}

	var cumulative float32

	td.cd = tensorMust(td.calcPd(muStds).Apply(func(v float32) float32 {
		cumulative += v
		return cumulative
	}))

	return td
}

// ThresholdDecoder decodes raw neural network output
type ThresholdDecoder struct {
	options    ThresholdOptions
	minZ, maxZ float32
	MinOut     int
	MaxOut     int
	OutRange   int
	cd         *tensor.Dense
}

// Decode takes raw neural network output and normalizes it using
// probability distribution
func (t *ThresholdDecoder) Decode(rawOutput float32) float32 {
	if rawOutput == 1 || rawOutput == 0 {
		return rawOutput
	}

	var cp float32

	if t.OutRange == 0 {
		if rawOutput > float32(t.MinOut) {
			cp = 1
		} else {
			cp = 0
		}
	} else {
		ratio := (asigmoid(rawOutput) - float32(t.MinOut)) / float32(t.OutRange)
		ratio = float32(math.Min(math.Max(float64(ratio), 0.0), 1.0))

		cp = t.cd.GetF32(int(ratio*float32(t.cd.Size()-1) + 0.5))
	}

	if cp < t.options.Center {
		return 0.5 * cp / t.options.Center
	}

	return 0.5 + 0.5*(cp-t.options.Center)/(1.0-t.options.Center)
}

// calcPd Fills a list with probability distributions
func (t *ThresholdDecoder) calcPd(muStds MuStd) *tensor.Dense {
	points := LinSpace(float32(t.MinOut), float32(t.MaxOut), t.options.Resolution*t.OutRange)

	pointsSize := points.Shape()[0]

	// Why this you ask? Well, it's hard to create a tensor from multiple tensors.
	items := make([]float32, len(muStds)*pointsSize)

	for i, muStd := range muStds {
		d := pdf(points, muStd[0], muStd[1])

		// Copy into a pre-allocated slice
		copy(items[i*pointsSize:], d.Data().([]float32))
	}

	data := tensor.New(tensor.Of(tensor.Float32), tensor.WithShape(len(muStds), pointsSize), tensor.WithBacking(items))

	// Dimensions should be x, len(points)

	summed := tensorMust(data.Sum(0))

	// After sum, we should have a single len(points) tensor

	return tensorMust(summed.DivScalar(float32(t.options.Resolution*len(muStds)), true))
}

// asigmoid Inverse sigmoid (logit) for scalars
func asigmoid(x float32) float32 {
	return float32(-math.Log(float64(1.0/x - 1.0)))
}

// pdf is a Probability density function (normal distribution)
func pdf(x *tensor.Dense, mu, std float32) *tensor.Dense {
	if std == 0 {
		return tensorMust(x.MulScalar(0.0, true))
	}

	// return (1.0 / (std * sqrt(2 * pi))) * np.exp(-(x - mu) ** 2 / (2 * std ** 2))

	// (1.0 / (std * sqrt(2 * pi)))
	a1 := 1.0 / (std * float32(math.Sqrt(2.0*math.Pi)))

	//let b1 =  -(x - mu).mapv(|v|v.powf(2.0));
	muSub := tensorMust(x.SubScalar(mu, true))
	muSub = tensorMust(muSub.PowScalar(float32(2.0), true))

	b1 := tensorMust(muSub.Apply(func(v float32) float32 {
		return -v
	}))

	//let b2 =  b1 / (2.0 * std.powf(2.0));

	b2 := tensorMust(b1.DivScalar(float32(2.0*math.Pow(float64(std), 2.0)), true))
	//a1 * (b2.mapv(|v|v.exp()))

	// a1 * np.exp(...)
	return tensorMust(tensor.Mul(a1, tensorMust(tensor.Exp(b2))))
}

// tensorMust is a simple utility that takes the output of a tensor operation and
// requires err to be nil
// This bypasses some unnecessary error checking on tensor operations
func tensorMust(t tensor.Tensor, err error) *tensor.Dense {
	if err != nil {
		panic(err)
	}

	return t.(*tensor.Dense)
}

// LinSpace returns evenly spaced numbers over a specified closed interval.
func LinSpace(start, stop float32, num int) (res *tensor.Dense) {
	if num <= 0 {
		return tensor.New()
	}
	if num == 1 {
		return tensor.New(tensor.WithBacking([]float32{start}))
	}

	step := (stop - start) / float32(num-1)
	resFloats := make([]float32, num)
	resFloats[0] = start
	for i := 1; i < num; i++ {
		resFloats[i] = start + float32(i)*step
	}
	resFloats[num-1] = stop

	return tensor.New(tensor.WithShape(len(resFloats)), tensor.WithBacking(resFloats))
}
