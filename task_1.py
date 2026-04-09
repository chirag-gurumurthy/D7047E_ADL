import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report
from nltk.corpus import stopwords
from nltk import word_tokenize


# ── Preprocessing ────────────────────────────────────────────────────────────

def preprocess_pandas(data, columns):
    df_ = pd.DataFrame(columns=columns)
    data['Sentence'] = data['Sentence'].str.lower()
    data['Sentence'] = data['Sentence'].replace('[a-zA-Z0-9-_.]+@[a-zA-Z0-9-_.]+', '', regex=True)
    data['Sentence'] = data['Sentence'].replace(r'((25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)(\.|$)){4}', '', regex=True)
    data['Sentence'] = data['Sentence'].str.replace(r'[^\w\s]', '', regex=True)
    data['Sentence'] = data['Sentence'].replace(r'\d', '', regex=True)
    for index, row in data.iterrows():
        word_tokens = word_tokenize(row['Sentence'])
        filtered_sent = [w for w in word_tokens if w not in stopwords.words('english')]
        df_.loc[len(df_)] = {
            "index": row['index'],
            "Class": row['Class'],
            "Sentence": " ".join(filtered_sent)
        }
    return df_


# ── Model ─────────────────────────────────────────────────────────────────────

class MLPClassifier(nn.Module):
    """Simple feed-forward neural network for binary text classification."""

    def __init__(self, input_dim: int, hidden_dims: list[int], num_classes: int = 2, dropout: float = 0.3):
        super().__init__()
        layers = []
        prev = input_dim
        for h in hidden_dims:
            layers += [
                nn.Linear(prev, h),
                nn.ReLU(),
                nn.Dropout(dropout),
            ]
            prev = h
        layers.append(nn.Linear(prev, num_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ── Training & Evaluation ─────────────────────────────────────────────────────

def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss, correct = 0.0, 0
    for X, y in loader:
        X, y = X.to(device), y.to(device)
        optimizer.zero_grad()
        logits = model(X)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(y)
        correct += (logits.argmax(1) == y).sum().item()
    n = len(loader.dataset)
    return total_loss / n, correct / n


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, correct = 0.0, 0
    all_preds, all_labels = [], []
    for X, y in loader:
        X, y = X.to(device), y.to(device)
        logits = model(X)
        total_loss += criterion(logits, y).item() * len(y)
        preds = logits.argmax(1)
        correct += (preds == y).sum().item()
        all_preds.extend(preds.cpu().tolist())
        all_labels.extend(y.cpu().tolist())
    n = len(loader.dataset)
    return total_loss / n, correct / n, all_preds, all_labels


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Hyperparameters
    HIDDEN_DIMS = [256, 128]
    DROPOUT     = 0.3
    BATCH_SIZE  = 64
    EPOCHS      = 20
    LR          = 1e-3
    MAX_FEATURES = 50_000

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load & preprocess
    data = pd.read_csv("amazon_cells_labelled.txt", delimiter='\t', header=None)
    data.columns = ['Sentence', 'Class']
    data['index'] = data.index
    columns = ['index', 'Class', 'Sentence']
    data = preprocess_pandas(data, columns)

    # Split
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        data['Sentence'].values.astype('U'),
        data['Class'].values.astype('int32'),
        test_size=0.10,
        random_state=0,
        shuffle=True,
    )

    # TF-IDF vectorisation
    vectorizer = TfidfVectorizer(
        analyzer='word', ngram_range=(1, 2),
        max_features=MAX_FEATURES, max_df=0.5,
        use_idf=True, norm='l2',
    )
    X_train = torch.from_numpy(np.array(vectorizer.fit_transform(train_texts).todense())).float()
    X_val   = torch.from_numpy(np.array(vectorizer.transform(val_texts).todense())).float()
    y_train = torch.from_numpy(train_labels).long()
    y_val   = torch.from_numpy(val_labels).long()

    input_dim = X_train.shape[1]

    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(TensorDataset(X_val,   y_val),   batch_size=BATCH_SIZE)

    # Model, loss, optimiser
    model     = MLPClassifier(input_dim, HIDDEN_DIMS, dropout=DROPOUT).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)

    print(model)
    print(f"\nInput dim: {input_dim} | Train: {len(train_texts)} | Val: {len(val_texts)}\n")

    # Training loop
    for epoch in range(1, EPOCHS + 1):
        tr_loss, tr_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        vl_loss, vl_acc, _, _ = evaluate(model, val_loader, criterion, device)
        print(f"Epoch {epoch:02d}/{EPOCHS}  "
              f"train_loss={tr_loss:.4f}  train_acc={tr_acc:.4f}  "
              f"val_loss={vl_loss:.4f}  val_acc={vl_acc:.4f}")

    # Final report
    _, _, preds, labels = evaluate(model, val_loader, criterion, device)
    print("\nClassification Report (Validation):")
    print(classification_report(labels, preds, target_names=["Negative", "Positive"]))
