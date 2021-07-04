/*
This file contains a few tools for measuring similarity/distance
between vectors, such as Euclidean distance and cosine similarity.
*/
package mathutils

import (
	"errors"
	"math"
)

// EuclideanDistance finds the Euclidean distance between
// two vectors. Returns an err if the vectors are of diff
// lengths, or if one of the vecs is nil.
func EuclideanDistance(v1, v2 []float64) (float64, error) {
	if v1 == nil || v2 == nil {
		return .0, errors.New("nil vec")
	}
	if len(v1) != len(v2) {
		s := "distance measurement attempt failed: "
		s += "vectors are of different lengths"
		return 0, errors.New(s)
	}
	var r float64
	for i := 0; i < len(v1); i++ {
		r += math.Sqrt((v1[i] - v2[i]) * (v1[i] - v2[i]))
	}
	return r, nil
}

func norm(vec []float64) float64 {
	var x float64
	for i := 0; i < len(vec); i++ {
		x += vec[i] * vec[i]
	}
	return math.Sqrt(x)
}

// CosineSimilarity finds the cosine similarity between
// two vectors. Returns an err if the vectors are of diff
// lengths, or if one of the vecs is nil.
func CosineSimilarity(v1, v2 []float64) (float64, error) {
	if v1 == nil || v2 == nil {
		return .0, errors.New("nil vec")
	}
	if len(v1) != len(v2) {
		s := "similarity measurement attempt failed: "
		s += "vectors are of different lengths"
		return 0, errors.New(s)
	}
	norm1, norm2 := norm(v1), norm(v2)
	if norm1 == 0 && norm2 == 0 {
		return 0, nil
	}
	var dot float64
	for i := 0; i < len(v1); i++ {
		dot += v1[i] * v2[i]
	}
	return dot / norm1 / norm2, nil
}
