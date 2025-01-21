package main

import (
	"fmt"
	"log"
	"math"
)

// Order represents an order with its location and ID
type Order struct {
	ID   int
	X, Y float64
}

// Courier represents a courier with its location and ID
type Courier struct {
	ID   int
	X, Y float64
}

// BipartiteGraph represents a weighted bipartite graph for the KM algorithm
type BipartiteGraph struct {
	Orders   []Order
	Couriers []Courier

	// Weight is a 2D slice storing the weight (negative distance) between each order and courier
	Weight [][]float64

	// Lx stores the labels for the left side (orders) of the bipartite graph
	Lx []float64

	// Ly stores the labels for the right side (couriers) of the bipartite graph
	Ly []float64

	// MatchX stores the matching information for the left side (orders)
	// MatchX[i] = j means order i is matched to courier j
	MatchX []int

	// MatchY stores the matching information for the right side (couriers)
	// MatchY[j] = i means courier j is matched to order i
	MatchY []int

	// Slack is used in the KM algorithm to store the smallest slack for each courier
	Slack []float64
}

// NewBipartiteGraph creates a new BipartiteGraph instance
func NewBipartiteGraph(orders []Order, couriers []Courier) *BipartiteGraph {
	n, m := len(orders), len(couriers)
	bg := &BipartiteGraph{
		Orders:   orders,
		Couriers: couriers,
		Weight:   make([][]float64, n),
		Lx:       make([]float64, n),
		Ly:       make([]float64, m),
		MatchX:   make([]int, n),
		MatchY:   make([]int, m),
		Slack:    make([]float64, m),
	}
	for i := range bg.Weight {
		bg.Weight[i] = make([]float64, m)
	}
	for i := range bg.MatchX {
		bg.MatchX[i] = -1
	}
	for i := range bg.MatchY {
		bg.MatchY[i] = -1
	}
	return bg
}

// calculateWeight computes the weight (negative distance) between an order and a courier
func (bg *BipartiteGraph) calculateWeight(i, j int) float64 {
	dx := bg.Orders[i].X - bg.Couriers[j].X
	dy := bg.Orders[i].Y - bg.Couriers[j].Y
	return -math.Sqrt(dx*dx + dy*dy) // Negative because KM finds maximum weight matching
}

// InitializeWeights fills the weight matrix
func (bg *BipartiteGraph) InitializeWeights() {
	for i := range bg.Orders {
		for j := range bg.Couriers {
			bg.Weight[i][j] = bg.calculateWeight(i, j)
		}
	}
}

// KMAlgorithm implements the Kuhn-Munkres algorithm for maximum weight matching
// in the bipartite graph. It finds the optimal assignment of orders to couriers
// that maximizes the total weight (minimizes the total distance in this case).
//
// The algorithm works as follows:
//  1. Initialize labels for orders (Lx) and couriers (Ly)
//  2. For each unmatched order:
//     a. Try to find an augmenting path using DFS
//     b. If no augmenting path is found, update labels and try again
//  3. Repeat step 2 until all orders are matched or no improvement is possible
//
// Returns the total weight of the matching (negative total distance)
func (bg *BipartiteGraph) KMAlgorithm() float64 {
	n, m := len(bg.Orders), len(bg.Couriers)

	log.Println("Starting KM Algorithm")
	log.Printf("Number of orders: %d, Number of couriers: %d", n, m)

	// Initialize labels
	for i := 0; i < n; i++ {
		bg.Lx[i] = -math.Inf(1)
		for j := 0; j < m; j++ {
			if bg.Weight[i][j] > bg.Lx[i] {
				bg.Lx[i] = bg.Weight[i][j]
			}
		}
	}
	log.Printf("Labels initialized: %+v, weights: %+v", bg.Lx, bg.Weight)

	for i := 0; i < n; i++ {
		log.Printf("Processing order %d", bg.Orders[i].ID)

		for j := range bg.Slack {
			bg.Slack[j] = math.Inf(1)
		}

		iterCount := 0
		for {
			iterCount++
			log.Printf("Iteration %d for order %d", iterCount, bg.Orders[i].ID)

			visx := make([]bool, n)
			visy := make([]bool, m)
			if bg.dfs(i, visx, visy) {
				log.Printf("Augmenting path found for order %d", bg.Orders[i].ID)
				break
			}

			delta := math.Inf(1)
			for j := 0; j < m; j++ {
				if !visy[j] && bg.Slack[j] < delta {
					delta = bg.Slack[j]
				}
			}
			if delta == math.Inf(1) {
				log.Printf("No improvement possible for order %d", bg.Orders[i].ID)
				break
			}

			log.Printf("Updating labels with delta: %f", delta)
			for k := 0; k < n; k++ {
				if visx[k] {
					bg.Lx[k] -= delta
				}
			}
			for k := 0; k < m; k++ {
				if visy[k] {
					bg.Ly[k] += delta
				} else {
					bg.Slack[k] -= delta
				}
			}
		}
	}

	totalWeight := 0.0
	for i := 0; i < n; i++ {
		if bg.MatchX[i] != -1 {
			totalWeight += bg.Weight[i][bg.MatchX[i]]
		}
	}

	log.Printf("KM Algorithm completed. Total weight: %f", totalWeight)
	return totalWeight
}

// dfs performs depth-first search to find an augmenting path
func (bg *BipartiteGraph) dfs(u int, visx, visy []bool) bool {
	visx[u] = true
	log.Printf("DFS visiting order %d", bg.Orders[u].ID)

	for v := 0; v < len(bg.Couriers); v++ {
		if visy[v] {
			continue
		}

		t := bg.Lx[u] + bg.Ly[v] - bg.Weight[u][v]
		if math.Abs(t) < 1e-6 {
			log.Printf("Considering courier %d for order %d", bg.Couriers[v].ID, bg.Orders[u].ID)

			visy[v] = true
			if bg.MatchY[v] == -1 || bg.dfs(bg.MatchY[v], visx, visy) {
				bg.MatchX[u] = v
				bg.MatchY[v] = u
				log.Printf("Matched order %d with courier %d", bg.Orders[u].ID, bg.Couriers[v].ID)
				return true
			}
		} else if t < bg.Slack[v] {
			bg.Slack[v] = t
			log.Printf("Updated slack for courier %d: %f", bg.Couriers[v].ID, t)
		}
	}

	log.Printf("No match found for order %d in this DFS", bg.Orders[u].ID)
	return false
}

// PrintMatching outputs the final matching result
func (bg *BipartiteGraph) PrintMatching() {
	for i, v := range bg.MatchX {
		if v != -1 {
			fmt.Printf("订单 %d 分配给配送员 %d, 距离: %.2f\n",
				bg.Orders[i].ID, bg.Couriers[v].ID, -bg.Weight[i][v])
		}
	}
}

func main() {
	orders := []Order{
		{ID: 1, X: 1, Y: 2},
		{ID: 2, X: 3, Y: 4},
		{ID: 3, X: 2, Y: 6},
	}
	couriers := []Courier{
		{ID: 1, X: 0, Y: 0},
		{ID: 2, X: 2, Y: 3},
		{ID: 3, X: 8, Y: 4},
	}

	log.SetFlags(0)

	bg := NewBipartiteGraph(orders, couriers)
	bg.InitializeWeights()

	totalWeight := bg.KMAlgorithm()

	fmt.Printf("最大权匹配总权重: %.2f\n", -totalWeight)
	bg.PrintMatching()
}
