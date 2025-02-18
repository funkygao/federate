package main

import (
	"fmt"
	"math"
	"sort"
	"time"
)

type Task struct {
	ID                 int
	Origin             Point
	Destination        Point
	TimeWindow         TimeWindow
	SpecialRequirement bool    // 特殊需求标志（如大件物品）
	Priority           int     // 1-5，5最高
	Size               float64 // 物品大小
}

type TimeWindow struct {
	Start time.Time
	End   time.Time
}

// 配送员
type Courier struct {
	ID            int
	Position      Point
	Capacity      float64 // 最大容量
	CurrentLoad   float64 // 当前负载
	AvailableTime time.Time
	EndTime       time.Time // 工作结束时间
	Skills        []string
	AssignedTasks []int // 已分配的任务ID
}

func (c *Courier) CanCompleteTaskInTime(task Task, timeToComplete time.Duration) bool {
	estimatedArrival := c.AvailableTime.Add(timeToComplete)
	return estimatedArrival.Before(task.TimeWindow.End)
}

func (c *Courier) HasCapacityFor(task Task) bool {
	return c.CurrentLoad+task.Size <= c.Capacity
}

func (c *Courier) IsAvailable(currentTime time.Time) bool {
	return currentTime.Before(c.EndTime)
}

func (c *Courier) HasSkill(skill string) bool {
	for _, s := range c.Skills {
		if s == skill {
			return true
		}
	}
	return false
}

func (c *Courier) InGeoFence(geoFence GeoFence) bool {
	return c.Position.Distance(geoFence.Center) <= geoFence.Radius
}

// 定义点结构体
type Point struct {
	X, Y float64
}

// 欧几里得距离
func (p1 *Point) Distance(p2 Point) float64 {
	return math.Sqrt(math.Pow(p1.X-p2.X, 2) + math.Pow(p1.Y-p2.Y, 2))
}

// 地理围栏
type GeoFence struct {
	Center Point
	Radius float64 // 围栏半径
}

// 定义实时路况数据
type TrafficData struct {
	RoadCondition map[string]float64 // 路况信息，键为道路标识，值为拥堵程度（0-1）
}

func (t *TrafficData) CourierFactor(courierID int) float64 {
	// 假设每个配送员都有一个唯一的道路标识
	roadID := fmt.Sprintf("road_%d", courierID)
	trafficFactor := t.RoadCondition[roadID]
	if trafficFactor == 0 {
		trafficFactor = 1.0 // 默认无拥堵
	}
	return trafficFactor
}

func calculateScore(courier Courier, task Task, dist float64, currentTime time.Time) float64 {
	timeScore := math.Max(0, float64(task.TimeWindow.End.Sub(currentTime))/float64(time.Hour)) * 10
	priorityScore := float64(task.Priority) * 20
	distanceScore := (100 - dist) * 2 // 假设最大距离为100
	return timeScore + priorityScore + distanceScore
}

// OD任务分配算法：Origin-Destination
func assignTasks(tasks []Task, couriers []Courier, geoFence GeoFence, trafficData TrafficData, currentTime time.Time) map[int][]int {
	assignments := make(map[int][]int)

	sort.Slice(tasks, func(i, j int) bool {
		return tasks[i].Priority > tasks[j].Priority
	})

	for _, task := range tasks {
		bestScore := math.Inf(-1)
		assignedCourierID := -1

		fmt.Printf("\nAttempting to assign Task %d\n", task.ID)

		for i, courier := range couriers {
			fmt.Printf("  Checking Courier %d\n", courier.ID)

			if !courier.InGeoFence(geoFence) {
				fmt.Printf("    Courier %d is not in geo fence\n", courier.ID)
				continue
			}

			if !courier.IsAvailable(currentTime) {
				fmt.Printf("    Courier %d is not available\n", courier.ID)
				continue
			}

			dist := task.Origin.Distance(courier.Position)
			dist *= trafficData.CourierFactor(courier.ID)

			timeToComplete := time.Duration(dist / 10 * float64(time.Hour))
			fmt.Printf("    Distance: %.2f, Time to complete: %v\n", dist, timeToComplete)

			if !courier.CanCompleteTaskInTime(task, timeToComplete) {
				fmt.Printf("    Courier %d cannot complete task in time\n", courier.ID)
				continue
			}

			if !courier.HasCapacityFor(task) {
				fmt.Printf("    Courier %d does not have capacity for task\n", courier.ID)
				continue
			}

			if task.SpecialRequirement && !courier.HasSkill("heavy_items") {
				fmt.Printf("    Courier %d does not have required skill\n", courier.ID)
				continue
			}

			score := calculateScore(courier, task, dist, currentTime)
			fmt.Printf("    Score: %.2f\n", score)

			if score > bestScore {
				bestScore = score
				assignedCourierID = i
			}
		}

		if assignedCourierID != -1 {
			courier := &couriers[assignedCourierID]
			courier.AssignedTasks = append(courier.AssignedTasks, task.ID)
			courier.CurrentLoad += task.Size
			timeToComplete := time.Duration(task.Origin.Distance(courier.Position) / 10 * float64(time.Hour))
			courier.AvailableTime = courier.AvailableTime.Add(timeToComplete)
			courier.Position = task.Destination

			if assignments[courier.ID] == nil {
				assignments[courier.ID] = []int{}
			}
			assignments[courier.ID] = append(assignments[courier.ID], task.ID)

			fmt.Printf("  Task %d assigned to Courier %d\n", task.ID, courier.ID)
		} else {
			fmt.Printf("  No suitable courier found for Task %d\n", task.ID)
		}
	}

	return assignments
}

func printCourierStatus(couriers []Courier) {
	for _, c := range couriers {
		fmt.Printf("Courier %d - Position: (%.2f, %.2f), CurrentLoad: %.2f, AvailableTime: %v\n",
			c.ID, c.Position.X, c.Position.Y, c.CurrentLoad, c.AvailableTime)
	}
}

func printTaskDetails(tasks []Task) {
	for _, t := range tasks {
		fmt.Printf("Task %d - Origin: (%.2f, %.2f), Destination: (%.2f, %.2f), TimeWindow: %v to %v, Priority: %d, Size: %.2f\n",
			t.ID, t.Origin.X, t.Origin.Y, t.Destination.X, t.Destination.Y, t.TimeWindow.Start, t.TimeWindow.End, t.Priority, t.Size)
	}
}

func main() {
	geoFence := GeoFence{
		Center: Point{X: 5, Y: 5},
		Radius: 15,
	}

	now := time.Now()
	tasks := []Task{
		{ID: 1, Origin: Point{X: 0, Y: 0}, Destination: Point{X: 10, Y: 10}, TimeWindow: TimeWindow{Start: now, End: now.Add(4 * time.Hour)}, SpecialRequirement: false, Priority: 3, Size: 1},
		{ID: 2, Origin: Point{X: 5, Y: 5}, Destination: Point{X: 15, Y: 15}, TimeWindow: TimeWindow{Start: now, End: now.Add(3 * time.Hour)}, SpecialRequirement: true, Priority: 5, Size: 2},
		{ID: 3, Origin: Point{X: 2, Y: 2}, Destination: Point{X: 8, Y: 8}, TimeWindow: TimeWindow{Start: now, End: now.Add(5 * time.Hour)}, SpecialRequirement: false, Priority: 1, Size: 0.5},
	}

	couriers := []Courier{
		{ID: 1, Position: Point{X: 1, Y: 1}, Capacity: 5, CurrentLoad: 0, AvailableTime: now, EndTime: now.Add(8 * time.Hour), Skills: []string{"heavy_items"}},
		{ID: 2, Position: Point{X: 6, Y: 6}, Capacity: 3, CurrentLoad: 0, AvailableTime: now, EndTime: now.Add(8 * time.Hour), Skills: []string{}},
		{ID: 3, Position: Point{X: 3, Y: 3}, Capacity: 4, CurrentLoad: 0, AvailableTime: now, EndTime: now.Add(8 * time.Hour), Skills: []string{"heavy_items"}},
	}

	trafficData := TrafficData{
		RoadCondition: map[string]float64{
			"road_1": 1.2, // 道路1有轻微拥堵
			"road_2": 1.0, // 道路2无拥堵
			"road_3": 1.5, // 道路3有严重拥堵
		},
	}

	fmt.Println("Initial Courier Status:")
	printCourierStatus(couriers)

	fmt.Println("\nTask Details:")
	printTaskDetails(tasks)

	// 分配任务
	assignments := assignTasks(tasks, couriers, geoFence, trafficData, now)

	// 打印分配结果
	fmt.Println("\nTask Assignments:")
	for courierID, taskIDs := range assignments {
		fmt.Printf("Courier %d assigned to Tasks: %v\n", courierID, taskIDs)
	}

	fmt.Println("\nFinal Courier Status:")
	printCourierStatus(couriers)
}
