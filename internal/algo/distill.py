import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# chatgpt:      teacher_model
# stolen_model: student_model
chatgpt = AutoModelForSequenceClassification.from_pretrained('chatgpt-4o')
stolen_model = AutoModelForSequenceClassification.from_pretrained('借鸡下蛋模型')

class DistillationDataset(Dataset):
    def __init__(self, tokenizer, max_length, num_samples):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.num_samples = num_samples
        self.data = self.generate_data()

    def generate_data(self):
        # 软标签是指老师模型对输入样本的输出概率分布，而硬标签通常是指概率分布中概率最高的那个类别的标签
        # 这里使用老师模型生成数据和软标签
        data = []
        for _ in range(self.num_samples):
            # 生成一个句子或问题
            sentence = "生成的句子或问题"
            inputs = self.tokenizer(sentence, return_tensors='pt', max_length=self.max_length, truncation=True)
            with torch.no_grad():
                outputs = chatgpt(**inputs)
                soft_labels = torch.softmax(outputs.logits, dim=1)
            data.append((sentence, soft_labels))
        return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sentence, soft_labels = self.data[idx]
        inputs = self.tokenizer(sentence, return_tensors='pt', max_length=self.max_length, truncation=True)
        return inputs, soft_labels

# 初始化数据集和数据加载器
tokenizer = AutoTokenizer.from_pretrained('somewhere')
distillation_dataset = DistillationDataset(tokenizer, max_length=128, num_samples=10000)
data_loader = DataLoader(distillation_dataset, batch_size=32, shuffle=True)

# 定义损失函数和优化器
criterion = nn.KLDivLoss(reduction='batchmean')
optimizer = torch.optim.Adam(stolen_model.parameters(), lr=1e-5)

# 训练学生模型
stolen_model.train()
for epoch in range(5):  # 训练数据集全部扫5遍
    for inputs, soft_labels in data_loader:
        optimizer.zero_grad()
        outputs = stolen_model(**inputs)
        log_probs = torch.log_softmax(outputs.logits, dim=1)
        loss = criterion(log_probs, soft_labels)
        loss.backward()
        optimizer.step()
        print(f"Epoch {epoch+1}, Loss: {loss.item()}")

# 保存学生模型
stolen_model.save_pretrained('stolen.pth')
