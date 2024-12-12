import torch
import torch.nn as nn
from transformer import Transformer
from encoder import Encoder, EncoderBlock
from decoder import Decoder, DecoderBlock
from embeddings import InputEmbeddings, PositionalEncoding
from attention import MultiHeadAttentionBlock
from layers import FeedForwardBlock
from projection import ProjectionLayer

# 模型参数
d_model = 512
num_heads = 8
num_layers = 6
d_ff = 2048
src_vocab_size = 5000  # 简化的词汇表大小
tgt_vocab_size = 5000
dropout = 0.1
max_seq_len = 100


# 创建模型组件
def create_model():
    src_embed = InputEmbeddings(d_model, src_vocab_size)
    tgt_embed = InputEmbeddings(d_model, tgt_vocab_size)
    src_pos = PositionalEncoding(d_model, max_seq_len, dropout)
    tgt_pos = PositionalEncoding(d_model, max_seq_len, dropout)

    encoder_blocks = nn.ModuleList([
        EncoderBlock(
            MultiHeadAttentionBlock(d_model, num_heads, dropout),
            FeedForwardBlock(d_model, d_ff, dropout),
            dropout
        ) for _ in range(num_layers)
    ])

    decoder_blocks = nn.ModuleList([
        DecoderBlock(
            MultiHeadAttentionBlock(d_model, num_heads, dropout),
            MultiHeadAttentionBlock(d_model, num_heads, dropout),
            FeedForwardBlock(d_model, d_ff, dropout),
            dropout
        ) for _ in range(num_layers)
    ])

    encoder = Encoder(encoder_blocks)
    decoder = Decoder(decoder_blocks)
    projection_layer = ProjectionLayer(d_model, tgt_vocab_size)

    model = Transformer(encoder, decoder, src_embed, tgt_embed, src_pos, tgt_pos, projection_layer)
    return model


# 简化的词汇表和tokenization
src_vocab = {f"en_word_{i}": i for i in range(src_vocab_size)}
tgt_vocab = {f"fr_word_{i}": i for i in range(tgt_vocab_size)}
inv_src_vocab = {v: k for k, v in src_vocab.items()}
inv_tgt_vocab = {v: k for k, v in tgt_vocab.items()}


def tokenize(text, vocab):
    return [vocab.get(word, vocab['en_word_0']) for word in text.split()]


def detokenize(token_ids, inv_vocab):
    return ' '.join([inv_vocab.get(id, 'en_word_0') for id in token_ids])


# 翻译函数
def translate(model, src_sentence):
    model.eval()
    src_tokens = tokenize(src_sentence, src_vocab)
    src_tensor = torch.LongTensor(src_tokens).unsqueeze(0)
    src_mask = torch.ones(1, 1, len(src_tokens))

    with torch.no_grad():
        encoder_output = model.encode(src_tensor, src_mask)

        # 贪婪解码
        tgt_tensor = torch.LongTensor([[tgt_vocab['fr_word_0']]])  # 开始符号
        for _ in range(max_seq_len):
            tgt_mask = torch.triu(torch.ones(1, tgt_tensor.size(1), tgt_tensor.size(1)), diagonal=1).bool()
            out = model.decode(encoder_output, src_mask, tgt_tensor, tgt_mask)
            prob = model.project(out[:, -1])
            _, next_word = torch.max(prob, dim=1)
            tgt_tensor = torch.cat([tgt_tensor, next_word.unsqueeze(0)], dim=1)

            if next_word.item() == tgt_vocab['fr_word_1']:  # 假设 'fr_word_1' 是结束符号
                break

    return detokenize(tgt_tensor.squeeze().tolist(), inv_tgt_vocab)


# 主程序
def main():
    model = create_model()
    print("Transformer model created.")

    # 在实际应用中，你需要用预训练的权重加载模型
    # model.load_state_dict(torch.load('path_to_pretrained_weights.pth'))

    while True:
        src_sentence = input("Enter an English sentence (or 'q' to quit): ")
        if src_sentence.lower() == 'q':
            break
        translation = translate(model, src_sentence)
        print(f"Translation: {translation}")


if __name__ == "__main__":
    main()
