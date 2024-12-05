package java

import (
	"regexp"
)

var (
	resourceExtensions = map[string]struct{}{
		".json":       {}, // private ConfigMaps
		".xml":        {}, // beans, mybatis-config
		".properties": {}, // i18N message bundles, properties
		".yml":        {}, // spring boot
		".html":       {}, // email templates
	}

	// 使用不区分大小写的正则表达式来匹配 Spring 配置文件
	yamlFilePattern = regexp.MustCompile(`(?i)^application(-[a-zA-Z0-9\-]+)?\.(yml)$`)
)
