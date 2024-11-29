package main

// LengthPredictor predicts decode lengths for requests
type LengthPredictor struct {
	BatchSize int
}

func (p *LengthPredictor) PredictLength(requests []Request) map[string]LengthRange {
	predictions := make(map[string]LengthRange)
	for i := 0; i < len(requests); i += p.BatchSize {
		end := i + p.BatchSize
		if end > len(requests) {
			end = len(requests)
		}
		batch := requests[i:end]
		batchPredictions := p.processBatch(batch)
		for id, prediction := range batchPredictions {
			predictions[id] = prediction
		}
	}
	return predictions
}

func (p *LengthPredictor) processBatch(batch []Request) map[string]LengthRange {
	predictions := make(map[string]LengthRange)
	for _, req := range batch {
		predictions[req.ID] = LengthRange{
			Min: req.PromptTokens,
			Max: req.PromptTokens * 2, // Simple placeholder prediction
		}
	}
	return predictions
}
