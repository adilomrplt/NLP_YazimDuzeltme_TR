from flask import Flask, request, render_template, jsonify
import torch
import torch.nn as nn
import pandas as pd

# ==== MODEL SINIFI ====
class Seq2SeqLSTM(nn.Module):
    def __init__(self, vocab_size, emb_size=256, hidden_size=512, num_layers=3, dropout=0.3):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_size, padding_idx=0)
        self.encoder = nn.LSTM(emb_size, hidden_size, num_layers, dropout=dropout, batch_first=True)
        self.decoder = nn.LSTM(emb_size, hidden_size, num_layers, dropout=dropout, batch_first=True)
        self.fc = nn.Linear(hidden_size, vocab_size)
    def forward(self, src, trg, teacher_forcing_ratio=0.5):
        device = src.device
        embedded_src = self.embedding(src)
        _, (hidden, cell) = self.encoder(embedded_src)
        batch_size, trg_len = trg.size()
        outputs = torch.zeros(batch_size, trg_len, self.fc.out_features).to(device)
        input_token = trg[:, 0]
        for t in range(1, trg_len):
            embedded = self.embedding(input_token).unsqueeze(1)
            output, (hidden, cell) = self.decoder(embedded, (hidden, cell))
            pred = self.fc(output.squeeze(1))
            outputs[:, t] = pred
            teacher_force = torch.rand(1).item() < teacher_forcing_ratio
            top1 = pred.argmax(1)
            input_token = trg[:, t] if teacher_force else top1
        return outputs

# ==== MAPPINGLER VE MODEL YÜKLEME ====
df = pd.read_csv('dataset.csv', sep=';')
UZUNLUK_SINIRI = 30
if 'yanlis_kelime' in df.columns and 'dogru_kelime' in df.columns:
    pass
else:
    df.columns = ['yanlis_kelime', 'dogru_kelime']

df = df[(df['yanlis_kelime'].str.len() <= UZUNLUK_SINIRI) & (df['dogru_kelime'].str.len() <= UZUNLUK_SINIRI)]
df = df[df['yanlis_kelime'] != df['dogru_kelime']]
df = df.drop_duplicates(subset=['yanlis_kelime', 'dogru_kelime']).reset_index(drop=True)
karakterler = set(''.join(df['yanlis_kelime']) + ''.join(df['dogru_kelime']))
karakterler = sorted(list(karakterler))
karakterler = ['<pad>', '<s>', '<e>'] + karakterler
char2idx = {ch: i for i, ch in enumerate(karakterler)}
idx2char = {i: ch for ch, i in char2idx.items()}
VOCAB_SIZE = len(char2idx)
MAX_KELIME = max(df['yanlis_kelime'].str.len().max(), df['dogru_kelime'].str.len().max())

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Seq2SeqLSTM(VOCAB_SIZE, 256, 512, 3, 0.3).to(device)
model.load_state_dict(torch.load('./model/full_lstm_seq2seq_epoch5.pt', map_location=device))
model.eval()

def predict_word(model, yanlis_kelime, char2idx, idx2char, device, max_len=MAX_KELIME):
    seq = [char2idx['<s>']] + [char2idx.get(ch, char2idx['<pad>']) for ch in yanlis_kelime] + [char2idx['<e>']]
    seq = seq[:max_len+2]
    if len(seq) < max_len+2:
        seq += [char2idx['<pad>']] * (max_len + 2 - len(seq))
    inp = torch.tensor([seq], dtype=torch.long, device=device)
    with torch.no_grad():
        embedded = model.embedding(inp)
        _, (hidden, cell) = model.encoder(embedded)
        input_token = torch.tensor([char2idx['<s>']], device=device)
        decoded = []
        for _ in range(max_len+2):
            emb = model.embedding(input_token).unsqueeze(1)
            output, (hidden, cell) = model.decoder(emb, (hidden, cell))
            pred = model.fc(output.squeeze(1))
            next_token = pred.argmax(1)
            if idx2char[next_token.item()] == '<e>':
                break
            decoded.append(idx2char[next_token.item()])
            input_token = next_token
    return ''.join(decoded)

def cumle_duzelt(metin):
    kelimeler = metin.strip().split()
    yanlislar = []
    dogrular = []
    duzeltilmis_cumle = []
    for kelime in kelimeler:
        duzelt = predict_word(model, kelime, char2idx, idx2char, device)
        dogrular.append(duzelt)
        if duzelt != kelime:
            yanlislar.append(kelime)
        duzeltilmis_cumle.append(duzelt)
    return yanlislar, dogrular, ' '.join(duzeltilmis_cumle)

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    mtn = ""
    h_metin = []
    d_metin = []
    if request.method == 'POST':
        mtn = request.form.get('metin', '')
        if mtn:
            yanlislar, dogrular, duzeltilmis_cumle = cumle_duzelt(mtn)
            h_metin = yanlislar
            d_metin = [duzeltilmis_cumle]
    return render_template('index.html', mtn=mtn, h_metin=h_metin, d_metin=d_metin)

@app.route('/duzelt', methods=['POST'])
def duzelt():
    data = request.json
    kelime = data.get('kelime', '')
    duzeltme = ""
    if kelime.strip():
        duzeltme = predict_word(model, kelime, char2idx, idx2char, device)
    return jsonify({'tahmin': duzeltme})

if __name__ == '__main__':
    app.run(debug=True)
