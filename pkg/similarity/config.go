package similarity

const (
	// 特征向量的位数，通常使用64位
	SimHashBits = 64

	// 将 SimHash 的 64 位特征向量分割成多个小段：band
	// 每个文档被放入 4 个 buckets，牺牲空间，提升召回率
	NumBands = 4

	// 每个 band 的位数：16
	BandBits = SimHashBits / NumBands

	simhashCacheKey = "simhash"
)
