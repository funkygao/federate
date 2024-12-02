package similarity

import (
	"testing"

	"github.com/mfonda/simhash"
)

func TestSimhash(t *testing.T) {
	var docs = [][]byte{
		[]byte("this is a test phrase"),
		[]byte("this is a test phrass"),
		[]byte("foo bar"),
	}

	hashes := make([]uint64, len(docs))
	featureSets := make([]*simhash.WordFeatureSet, len(docs))
	for i, d := range docs {
		featureSets[i] = simhash.NewWordFeatureSet(d)
		hashes[i] = simhash.Simhash(featureSets[i])
		t.Logf("Simhash of doc[%d] `%21s`: %x\n", i, d, hashes[i])
	}

	for i, fs := range featureSets {
		features := fs.GetFeatures()
		vector := simhash.Vectorize(features)
		fingerprint := simhash.Fingerprint(vector)
		t.Logf("doc[%d] %d features:%+v, vectorized:#%d %+v, fingerprint:%+v", i, len(features), features, len(vector), vector, fingerprint)
	}

	t.Logf("Comparison of `%s` and `%s`: %d\n", docs[0], docs[1], simhash.Compare(hashes[0], hashes[1]))
	t.Logf("Comparison of `%s` and `%s`: %d\n", docs[0], docs[2], simhash.Compare(hashes[0], hashes[2]))
}
