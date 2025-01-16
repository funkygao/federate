package api

// Info 是整个API信息的顶层结构
type Info map[string]InterfaceInfo

type InterfaceInfo struct {
	ExtendedInterfaces []string     `json:"extendedInterfaces"`
	Methods            []MethodInfo `json:"methods"`
}

// MethodInfo 表示一个方法的信息
type MethodInfo struct {
	Name       string          `json:"name"`
	Parameters []ParameterInfo `json:"parameters"`
	ReturnType string          `json:"returnType"`
}

// ParameterInfo 表示一个参数的信息
type ParameterInfo struct {
	Name string `json:"name"`
	Type string `json:"type"`
}
