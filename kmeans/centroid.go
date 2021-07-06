/*
This file contains an implementation for 'centroids' in a kmeans context.
The centroid impl here follows the common.Centroid interface.
*/
package kmeans

import (
	"github.com/crunchypi/net-means/common"
	"github.com/crunchypi/net-means/mathutils"
)

// These are simply abbreviations.
type (
	vecContainer     = common.VecContainer
	payloadContainer = common.PayloadContainer
	payloadReceiver  = common.PayloadReceiver
)

// Iface hint.
var _ common.Centroid = new(Centroid)

// Named parameter funcs.
type vecGenerator = func() ([]float64, bool)
type knnSearchFunc = func(targetVec []float64, vecs vecGenerator, k int) []int

// Centroid T in kmeans context. Implements common.Centroid interface.
type Centroid struct {
	vec           []float64
	DataPoints    []payloadContainer
	knnSearchFunc knnSearchFunc
	kfnSearchFunc knnSearchFunc
}

type NewCentroidArgs struct {
	InitVec       []float64
	InitCap       int
	KNNSearchFunc knnSearchFunc
	KFNSearchFunc knnSearchFunc
}

// NewCentroidFromVec creates a new centroid with the specified vector.
func NewCentroid(args NewCentroidArgs) (*Centroid, bool) {
	if args.KNNSearchFunc == nil || args.KFNSearchFunc == nil {
		return nil, false
	}

	c := Centroid{
		vec:           make([]float64, len(args.InitVec)),
		DataPoints:    make([]payloadContainer, 0, args.InitCap),
		knnSearchFunc: args.KNNSearchFunc,
		kfnSearchFunc: args.KFNSearchFunc,
	}
	for i, v := range args.InitVec {
		c.vec[i] = v
	}
	return &c, true
}

// Vec returns the vector of a centroid.
func (c *Centroid) Vec() []float64 { return c.vec }

// AddPayload adds a payload the relevant centroid. Returns false if the vector
// contained in p is of different length that the vector of the centroid.
func (c *Centroid) AddPayload(p payloadContainer) bool {
	if len(p.Vec()) != len(c.vec) || p.Expired() {
		return false
	}
	c.DataPoints = append(c.DataPoints, p)
	return true
}

// rmPayload removes an internal payload at an index, this is done unsafely
// (without bounds checking) on purpose. Note, it is a very simple thing
// but was put here for code clarity where this method is called.
func (c *Centroid) rmPayload(index int) {
	// _Should_ be re-sliced with O(1) going by Go documentation/code.
	c.DataPoints = append(c.DataPoints[:index], c.DataPoints[index+1:]...)
}

// payloadVecGenerator creates a generator which iterates through all internal
// payloads/data points and returns their vec. Auto-expires expired payloads.
func (c *Centroid) payloadVecGenerator() func() ([]float64, bool) {
	i := 0
	return func() ([]float64, bool) {
		// Check bounds and skip expired datapoints.
		for i < len(c.DataPoints) && c.DataPoints[i].Expired() {
			c.rmPayload(i)
		}
		if i >= len(c.DataPoints) {
			return nil, false
		}
		i++
		return c.DataPoints[i-1].Vec(), true
	}
}

// DrainUnordered drains n internal payloads/datapoints in a manner that
// has no particular significance, specifically by how they are stored
// internally -- in no order.
func (c *Centroid) DrainUnordered(n int) []payloadContainer {
	res := make([]payloadContainer, 0, n)
	for len(c.DataPoints) != 0 && len(res) < n {
		if !c.DataPoints[0].Expired() {
			res = append(res, c.DataPoints[0])
		}
		c.rmPayload(0)
	}
	return res
}

// DrainOrdered drains n internal payloads/datapoints that are furthest
// away from the internal vector of a centroid. Furthest away can mean
// different things, depending on the 'KFNSearchFunc' field used in the
// 'NewCentroidArgs' struct when creating a new Centroid with 'NewCentroid'.
// If that field is net-means/searchutils.KNFCos, then drain away data that
// has lowest cosine similarity.
func (c *Centroid) DrainOrdered(n int) []payloadContainer {
	res := make([]payloadContainer, 0, n)
	// Furthest neigh.
	indexes := c.kfnSearchFunc(c.vec, c.payloadVecGenerator(), n)
	for _, i := range indexes {
		res = append(res, c.DataPoints[i])
	}
	// Second loop for draining, as the vals in 'indexes' might not be ordered.
	for _, i := range indexes {
		c.rmPayload(i)
	}
	return res
}

// Expire looks through internal payload/datapoints and removes the ones
// that have expired. This needs a follow-up with Centroid.MemTrim() to
// completely free up the space and reduce the internal cap.
func (c *Centroid) Expire() {
	i := 0
	for i < len(c.DataPoints) {
		if c.DataPoints[i].Expired() {
			c.rmPayload(i)
			continue
		}
		i++
	}
}

func (c *Centroid) LenDP() int { return len(c.DataPoints) }

// MemTrim creates a new internal payload/datapoint slice where capacity
// equals len. Note, it's a costly operation.
func (c *Centroid) MemTrim() {
	// @ Currently inefficient since memory is essentially doubled
	// @ while doing this procedure.
	dp := make([]payloadContainer, 0, len(c.DataPoints))
	for i := 0; i < len(c.DataPoints); i++ {
		if !c.DataPoints[i].Expired() {
			dp = append(dp, c.DataPoints[i])
		}
	}
	c.DataPoints = dp
}

// MoveVector moves the internal centroid vector to be the mean of all
// contained payload/datapoints.
func (c *Centroid) MoveVector() bool {
	vec, ok := mathutils.VecMean(c.payloadVecGenerator())
	if ok {
		c.vec = vec
	}
	return ok
}

// DistributePayload removes n internal payloads/datapoints using the
// DrainOrdered method (see documentation for it), which is costly but default
// as this method would auto-distribute the worst internal data. The removed
// data is then added to receivers, which can fail if the receivers can't
// accept the data. In that case, the data goes back into self.
func (c *Centroid) DistributePayload(n int, receivers []payloadReceiver) {
	if receivers == nil || len(receivers) == 0 {
		return
	}
	// Need to have a slice here (i.e can't draw datapoints directly from
	// c.DataPoints) because this instance (c) can be one of the distributers.
	data := c.DrainOrdered(n)
	i := 0
	generator := func() ([]float64, bool) {
		if i >= len(receivers) {
			return nil, false
		}
		i++
		return receivers[i-1].Vec(), true
	}

	for j := 0; j < len(data); j++ {
		i = 0 // Reset generator.
		indexes := c.knnSearchFunc(data[j].Vec(), generator, 1)
		// Put back into self if (1) search failed or (2) adder failed to add.
		if len(indexes) == 0 || !receivers[indexes[0]].AddPayload(data[j]) {
			c.AddPayload(data[j])
		}
	}
}

// KNNLookup uses the supplied 'vec' to lookup 'n' best-fit payloads/datapoints
// and returns them; 'drain'=true will remove them from self as well. Best fit
// will depend on the 'KFNSearchFunc' field used in the 'NewCentroidArgs' struct
// when crating a new Centroid with 'NewCentroid'. If that field is for instance
// net-means/searchutils.KNNCos, then best fit equals best cosine similarity.
func (c *Centroid) KNNLookup(vec []float64, k int, drain bool) []payloadContainer {
	res := make([]payloadContainer, 0, k)

	indexes := c.knnSearchFunc(vec, c.payloadVecGenerator(), k)
	for _, i := range indexes {
		res = append(res, c.DataPoints[i])
	}

	if drain {
		for _, i := range indexes {
			c.rmPayload(i)
		}
	}
	return res
}
