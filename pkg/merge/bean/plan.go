package bean

import (
	"fmt"
	"log"
	"sort"
	"strings"

	"federate/pkg/federated"
	"federate/pkg/manifest"
	"federate/pkg/tablerender"
)

type ReconcilePlan struct {
	beanIdModificationFiles map[string]map[string]string // key 是文件路径，value 是 bean id 的修改映射
	beanIdConflicts         []BeanIdConflictDetail       // 冲突详情列表

	redundantClasses map[string]map[string]struct{} // key: classFullName/xmlFilePath

	reconcile *manifest.ReconcileSpec
}

// BeanIdConflictDetail 结构体用于存储冲突的详细信息
type BeanIdConflictDetail struct {
	Index     int    // 冲突索引
	FilePath  string // 文件路径
	OldBeanId string // 旧的 bean id
	NewBeanId string // 新的 bean id
}

func newReconcilePlan(m *manifest.Manifest) ReconcilePlan {
	return ReconcilePlan{
		beanIdModificationFiles: make(map[string]map[string]string),
		redundantClasses:        make(map[string]map[string]struct{}),
		beanIdConflicts:         []BeanIdConflictDetail{},
		reconcile:               &m.Main.Reconcile,
	}
}

func (p *ReconcilePlan) registerBeanIdConflict(info BeanIdInfo, beanId, newBeanId string) {
	p.beanIdConflicts = append(p.beanIdConflicts, BeanIdConflictDetail{
		Index:     len(p.beanIdConflicts) + 1,
		FilePath:  info.TargetFilePath,
		OldBeanId: beanId,
		NewBeanId: newBeanId,
	})
	if _, exists := p.beanIdModificationFiles[info.TargetFilePath]; !exists {
		p.beanIdModificationFiles[info.TargetFilePath] = make(map[string]string)
	}
	p.beanIdModificationFiles[info.TargetFilePath][beanId] = newBeanId
}

func (p *ReconcilePlan) HasConflict() bool {
	return len(p.beanIdConflicts) > 0
}

func (p *ReconcilePlan) ConflictCount() int {
	return len(p.beanIdConflicts)
}

func (p *ReconcilePlan) IsRedundantClass(classFullName, filePath string) bool {
	_, present := p.redundantClasses[classFullName][filePath]
	return present
}

func (p *ReconcilePlan) registerRedundantClass(classFullName, filePath string) {
	if _, exists := p.redundantClasses[classFullName]; !exists {
		p.redundantClasses[classFullName] = make(map[string]struct{})
	}
	p.redundantClasses[classFullName][filePath] = struct{}{}
}

func (p *ReconcilePlan) ShowConflictReport() {
	header := []string{"Generated Resource File", "Old Bean ID", "New Bean ID"}
	var data [][]string
	files := make(map[string]struct{})
	for _, detail := range p.beanIdConflicts {
		data = append(data, []string{
			federated.ResourceBaseName(detail.FilePath),
			detail.OldBeanId,
			detail.NewBeanId,
		})
		files[detail.FilePath] = struct{}{}
	}

	log.Printf("XML Bean id conflicts: %d, files: %d. Reconcile plan:", len(p.beanIdConflicts), len(files))
	tablerender.DisplayTable(header, data, true, 0)
}

// 在内存中构建解决冲突计划
func (b *XmlBeanManager) makeReconcilePlan() {
	for beanId, infos := range b.beanIdMap {
		if len(infos) <= 1 { // no conflicts
			continue
		}

		// 对 infos 按 ComponentName 和 ParentPath 进行排序，以便更改 bean id
		sort.Slice(infos, func(i, j int) bool {
			if infos[i].ComponentName != infos[j].ComponentName {
				return infos[i].ComponentName < infos[j].ComponentName
			}
			return strings.Join(infos[i].ParentPath, "/") < strings.Join(infos[j].ParentPath, "/")
		})

		componentBeanIdCount := make(map[string]int)
		for i, info := range infos {
			if b.plan.reconcile.ExcludeBeanClass(info.ClassFullName) {
				b.plan.registerRedundantClass(info.ClassFullName, info.TargetFilePath)
				continue
			}

			if i == 0 {
				continue // 保留第一个 bean id 不变
			}

			if b.plan.reconcile.SingletonBeanClass(info.ClassFullName) {
				b.plan.registerRedundantClass(info.ClassFullName, info.TargetFilePath)
				continue
			}

			newBeanId := b.decideNewBeanId(info, componentBeanIdCount, beanId)
			b.plan.registerBeanIdConflict(info, beanId, newBeanId)
		}
	}
}

func (b *XmlBeanManager) newBeanId(beanId string, seq int) string {
	return fmt.Sprintf("%s%s%d", beanId, beanIdReconcileSeparator, seq)
}

func (b *XmlBeanManager) decideNewBeanId(info BeanIdInfo, componentBeanIdCount map[string]int, beanId string) string {
	for {
		componentBeanIdCount[info.ComponentName]++
		newBeanId := b.newBeanId(beanId, componentBeanIdCount[info.ComponentName])

		// 检查新的 Bean ID 是否与现有的冲突
		if !b.isBeanIdConflict(newBeanId, info) {
			return newBeanId
		}
	}
}

func (b *XmlBeanManager) isBeanIdConflict(newBeanId string, info BeanIdInfo) bool {
	// 遍历所有现有的 Bean ID
	for _, infos := range b.beanIdMap {
		for _, existingInfo := range infos {
			// 跳过与当前 Bean 完全相同的条目
			// 这种情况下我们不考虑它是冲突的，因为它就是我们正在处理的 Bean
			if existingInfo.ComponentName == info.ComponentName &&
				existingInfo.TargetFilePath == info.TargetFilePath &&
				strings.Join(existingInfo.ParentPath, "/") == strings.Join(info.ParentPath, "/") {
				continue
			}

			// 检查是否存在冲突
			// 冲突的定义：相同的目标文件路径、相同的父路径，但 Bean ID 相同
			// 这表示在同一个 XML 文件的同一位置有两个相同的 Bean ID
			if existingInfo.TargetFilePath == info.TargetFilePath &&
				strings.Join(existingInfo.ParentPath, "/") == strings.Join(info.ParentPath, "/") &&
				newBeanId == existingInfo.BeanId {
				return true // 发现冲突
			}
		}
	}
	return false // 没有发现冲突
}
