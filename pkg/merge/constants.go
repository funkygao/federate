package merge

const (
	federatedDubboConsumerXmlFn = "federated-dubbo-consumer.xml"
	federatedJSFConsumerXmlFn   = "federated-jsf-consumer.xml"

	metaInf = "META-INF/"

	beanIdPathSeparator      = "." // 加载所有bean到内存时，嵌套关系
	beanIdReconcileSeparator = "-" // `#` 不允许，see https://www.w3.org/TR/xmlschema-2/#ID
)

const (
	RpcJsf   = "JSF"
	RpcDubbo = "Dubbo"
)
