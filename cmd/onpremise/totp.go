package onpremise

import (
	"crypto/hmac"
	"crypto/sha1"
	"encoding/binary"
	"fmt"
	"time"
)

// 生成基于时间的一次性密码
func generateTOTP() {
	if !isCurrentUserRoot() {
		fmt.Println("只有 root 有权限执行")
		return
	}

	// 将当前时间转换为给定时间步长的整数倍
	currentTime := time.Now().Unix() / int64(totpTtlSec)

	// 将时间值转换为字节序列
	timeBytes := make([]byte, 8)
	binary.BigEndian.PutUint64(timeBytes, uint64(currentTime))

	// 使用HMAC-SHA1算法和密钥（secret）来生成HMAC值
	mac := hmac.New(sha1.New, []byte(totpSecret))
	mac.Write(timeBytes)
	hmacValue := mac.Sum(nil)

	// 使用动态截取（Dynamic Truncation）技术从HMAC值中提取一个4字节的动态二进制码
	offset := hmacValue[len(hmacValue)-1] & 0x0f
	code := (int(hmacValue[offset])&0x7f)<<24 |
		(int(hmacValue[offset+1])&0xff)<<16 |
		(int(hmacValue[offset+2])&0xff)<<8 |
		(int(hmacValue[offset+3]) & 0xff)

	// 生成6位数字密码
	otp := code % 1000000

	// 将6位数字转换为字符串
	passcode := fmt.Sprintf("%06d", otp)
	fmt.Println(passcode)
	fmt.Printf("%d 秒内有效\n", totpTtlSec)
	fmt.Println("")
	fmt.Println("您可以执行以下命令来关闭所有节点。执行命令后，所有节点将在1分钟内自动关闭。")
	fmt.Printf("curl http://admin.wms.local/shutdown?s=%s\n", passcode)
}
