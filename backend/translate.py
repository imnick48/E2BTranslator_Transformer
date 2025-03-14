import torch
from transfomermod import Transformer
from tokenizer import Tokenizer
import re
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MAX_SEQ_LENGTH = 50
BATCH_SIZE = 32
D_MODEL = 512
NUM_HEADS = 8
NUM_LAYERS = 6
NUM_EPOCHS = 25
LEARNING_RATE = 0.0001
def clean_text(text):
    text = re.sub(r'[^\w\s\u0980-\u09FF]', '', str(text))
    text = text.lower().strip()
    return text
def pad_sequences(sequences, maxlen, padding_value=0):
    padded_sequences = []
    for sequence in sequences:
        if len(sequence) > maxlen:
            padded_sequences.append(sequence[:maxlen])
        else:
            padded_sequences.append(sequence + [padding_value] * (maxlen - len(sequence)))
    return torch.LongTensor(padded_sequences)
def load_pretrained_model(model_path):
    checkpoint = torch.load(model_path, map_location=torch.device('cpu'), weights_only=False)
    en_tokenizer = checkpoint['en_tokenizer']
    bn_tokenizer = checkpoint['bn_tokenizer']

    model = Transformer(
        d_model=D_MODEL,
        num_heads=NUM_HEADS,
        encoder_vocab_size=en_tokenizer.size(),
        decoder_vocab_size=bn_tokenizer.size(),
        num_layers=NUM_LAYERS,
        pad_idx=0
    ).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    return model, en_tokenizer, bn_tokenizer


def translate_text(model, text, en_tokenizer, bn_tokenizer, max_length=50):
    model.eval()
    cleaned_text = clean_text(text)
    input_seq = en_tokenizer.texts_to_sequences([cleaned_text])
    padded_input = pad_sequences(input_seq, max_length)
    src = padded_input.to(device)
    tgt = torch.LongTensor([[bn_tokenizer.word_index["<sos>"]]]).to(device)

    with torch.no_grad():
        for i in range(max_length-1):
            output = model(src, tgt)
            next_token = output[:, -1].argmax(-1).unsqueeze(1)
            tgt = torch.cat([tgt, next_token], dim=1)
            if next_token.item() == bn_tokenizer.word_index["<eos>"]:
                break
    output_seq = tgt.squeeze().cpu().numpy().tolist()
    translated_text = bn_tokenizer.sequences_to_texts([output_seq])[0]
    return translated_text
model, en_tokenizer, bn_tokenizer = load_pretrained_model('translation_transformer.pth')

def translate(text):
    return translate_text(model, text, en_tokenizer, bn_tokenizer)
