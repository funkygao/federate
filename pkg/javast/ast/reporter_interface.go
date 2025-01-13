package ast

func (i *Info) showInterfacesReport() {
	interfaceCounts := make(map[string]int)
	for _, interfaces := range i.Interfaces {
		for _, iface := range interfaces {
			interfaceCounts[iface]++
		}
	}

	i.showNameCountSection("Implemented Interfaces", []string{"Interface"}, topN(mapToNameCount(interfaceCounts), TopK))
}
