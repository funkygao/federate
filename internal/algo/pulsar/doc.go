// 一个topic在时间序列上被分为多个Ledger，使用LedgerId标识，在一个物理集群中，LedgerId不会重复，采用全局分配模式
// 一个partition在同一时刻只会有一个Ledger在写入
//
// journal 里不同 ledger 的都混在一起，一个 entry log 文件只包含属于同一个 ledger 的 entries
package main
