//go:build !amd64

package similarity

import "math/bits"

func hammingDistance(hash1, hash2 uint64) int {
	return bits.OnesCount64(hash1 ^ hash2)
}
