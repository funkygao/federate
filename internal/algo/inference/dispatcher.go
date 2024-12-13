package main

import (
	"log"
	"math/rand"
	"sort"
)

// Dispatcher selects a decode instance and forwards KV cache
type Dispatcher struct {
	DecodeInstances []*DecodeInstance
	ClusterMonitor  *ClusterMonitor
}

func NewDispatcher(clusterMonitor *ClusterMonitor) *Dispatcher {
	return &Dispatcher{
		ClusterMonitor: clusterMonitor,
	}
}

func (d *Dispatcher) DispatchRequest(result PrefillResult) {
	log.Printf("Dispatcher starting to dispatch request %s", result.RequestID)

	d.DecodeInstances = d.ClusterMonitor.GetDecodeInstances()

	// Step 1: Categorize decode instances
	alphaSet, betaSet := d.categorizeInstances(result.PredictedLength)
	log.Printf("Categorized instances: %d in alpha set, %d in beta set", len(alphaSet), len(betaSet))

	// Step 2: Use power-of-two algorithm to choose two instances from alpha set
	if len(alphaSet) < 2 {
		log.Printf("Not enough instances in alpha set for request %s, aborting dispatch", result.RequestID)
		return
	}

	candidates := d.powerOfTwoSelection(alphaSet)
	log.Printf("Selected %d candidates for power-of-two selection", len(candidates))

	// Step 3: Choose the instance with the least interference
	selectedInstance := d.selectLeastInterference(candidates)
	log.Printf("Selected decode instance %s for request %s", selectedInstance.ID, result.RequestID)

	// Transfer KV Cache to the selected decode instance
	d.transferKVCache(selectedInstance, result)
	log.Printf("Completed KV cache transfer for request %s to decode instance %s", result.RequestID, selectedInstance.ID)

}

func (d *Dispatcher) categorizeInstances(predictedLength LengthRange) ([]*DecodeInstance, []*DecodeInstance) {
	var alphaSet, betaSet []*DecodeInstance
	for _, instance := range d.DecodeInstances {
		if instance.AvailableResources >= predictedLength.Max {
			alphaSet = append(alphaSet, instance)
		} else {
			betaSet = append(betaSet, instance)
		}
	}
	return alphaSet, betaSet
}

func (d *Dispatcher) powerOfTwoSelection(instances []*DecodeInstance) []*DecodeInstance {
	idx1 := rand.Intn(len(instances))
	idx2 := rand.Intn(len(instances) - 1)
	if idx2 >= idx1 {
		idx2++
	}
	return []*DecodeInstance{instances[idx1], instances[idx2]}
}

func (d *Dispatcher) selectLeastInterference(candidates []*DecodeInstance) *DecodeInstance {
	sort.Slice(candidates, func(i, j int) bool {
		ratio1 := float64(candidates[i].HeavyDecodeCount) / float64(candidates[i].LightDecodeCount+1)
		ratio2 := float64(candidates[j].HeavyDecodeCount) / float64(candidates[j].LightDecodeCount+1)
		return ratio1 < ratio2
	})
	return candidates[0]
}

func (d *Dispatcher) transferKVCache(instance *DecodeInstance, result PrefillResult) {
	chunkSize := 1 << 20
	for i := 0; i < len(result.KVCache); i += chunkSize {
		end := i + chunkSize
		if end > len(result.KVCache) {
			end = len(result.KVCache)
		}
		chunk := result.KVCache[i:end]
		d.sendChunkRDMA(instance, chunk)
	}

	// Update instance load information
	d.updateInstanceLoad(instance, result.PredictedLength)
}

func (d *Dispatcher) sendChunkRDMA(instance *DecodeInstance, chunk []byte) {
	log.Printf("RDMA send chunk to %+v", instance)
}

func (d *Dispatcher) updateInstanceLoad(instance *DecodeInstance, predictedLength LengthRange) {
	// Update instance load information
	instance.AvailableResources -= predictedLength.Max
	if predictedLength.Max > 100 { // Example threshold, adjust as needed
		instance.HeavyDecodeCount++
	} else {
		instance.LightDecodeCount++
	}

	// Notify cluster monitor of the update
	d.ClusterMonitor.UpdateInstanceLoad(instance)
}
