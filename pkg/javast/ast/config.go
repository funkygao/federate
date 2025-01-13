package ast

import "federate/pkg/primitive"

var ignoredInterfaces = primitive.NewStringSet()

func init() {
	ignoredInterfaces.Add("Serializable")
}

func ignoreInteface(iface string) bool {
	return ignoredInterfaces.Contains(iface)
}
