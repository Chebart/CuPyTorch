from ..blocks import Embedding, TransformerEncoderLayer, Linear, Dropout, LayerNorm, GELU
from .abstract_model import AbstractModel
from ..data import Tensor

class BertEmbeddings:
    def __init__(
        self, 
        vocab_size: int,
        max_len: int, 
        d_model: int, 
        dropout: float = 0.1
    ):
        self.token_emb = Embedding(vocab_size, d_model)
        self.pos_emb = Embedding(max_len, d_model)
        self.seg_emb = Embedding(2, d_model)

        self.ln = LayerNorm(d_model)
        self.dropout = Dropout(dropout)

    def forward(self, input_ids, token_type_ids):
        _, T = input_ids.shape
        pos_ids = Tensor.arange(T, dtype = input_ids.dtype, device = input_ids.device)[None, :]

        x = (
            self.token_emb(input_ids)
            + self.pos_emb(pos_ids)
            + self.seg_emb(token_type_ids)
        )

        x = self.ln.forward(x)
        return self.dropout.forward(x)

    def backward(self, dLdy):
        dLdy = self.dropout.backward(dLdy)
        dLdy = self.ln.backward(dLdy)

        self.token_emb.backward(dLdy)
        self.pos_emb.backward(dLdy.sum(axis=0, keepdims=True))
        self.seg_emb.backward(dLdy)
        
class MLMHead:
    def __init__(
        self, 
        d_model: int, 
        vocab_size: int
    ):
        self.fc1 = Linear(d_model, d_model)
        self.gelu = GELU()
        self.ln = LayerNorm(d_model)
        self.fc2 = Linear(d_model, vocab_size)

    def forward(self, x):
        return self.fc2(self.ln(self.gelu(self.fc1(x))))

    def backward(self, dLdy):
        dLdy = self.fc2.backward(dLdy)
        dLdy = self.ln.backward(dLdy)
        dLdy = self.gelu.backward(dLdy)
        return self.fc1.backward(dLdy)

class BertClassifier:
    def __init__(self, d_model):
        self.fc = Linear(d_model, 1)

    def forward(self, hidden_states):
        self.seq_len = hidden_states.shape[1]
        cls = hidden_states[:, 0]
        return self.fc(cls)

    def backward(self, dLdy):
        dcls = self.fc.backward(dLdy)
        B, D = dcls.shape
        dx = Tensor.zeros((B, self.seq_len, D), dtype=dcls.dtype, device=dcls.device)
        dx[:, 0] = dcls
        return dx

class BERT(AbstractModel):
    def __init__(
        self,
        vocab_size: int, 
        d_model: int, 
        num_heads: int, 
        num_layers: int, 
        d_ff: int, 
        use_rope: bool = False,
        max_seq_length: int = 512, 
        dropout: float = 0.1
    ):
        self.task = None

        self.layers = [
            BertEmbeddings(vocab_size, max_seq_length, d_model, dropout),
        ]
        self.layers += [
            TransformerEncoderLayer(d_model, num_heads, d_ff, use_rope, max_seq_length, dropout) 
            for _ in range(num_layers)
        ]
        self.mlm_head = MLMHead(d_model, vocab_size)
        self.classifier = BertClassifier(d_model)
        super().__init__()

    def set_task(self, task_name):
        self.task = task_name

    def generate_mask(self, input_ids):
        return (input_ids == 0)[:, None, None, :]

    def forward(self, input_ids, token_type_ids):
        assert self.task is not None, "before call forward() set task for BERT"
        
        mask = self.generate_mask(input_ids)
        for _, layer in enumerate(self.layers):
            if isinstance(layer, BertEmbeddings):
                x = layer(input_ids, token_type_ids)
            else:
                x = layer(x, mask) 

        if self.task == "mlm":
            x = self.mlm_head(x)
        elif self.task == "cls":
            x = self.classifier(x)

        return x

    def backward(self, dLdy):
        assert self.task is not None, "before call backward() set task for BERT"

        if self.task == "mlm":
            dLdy = self.mlm_head.backward(dLdy)
        elif self.task == "cls":
            dLdy = self.classifier.backward(dLdy)

        for _, layer in enumerate(reversed(self.layers)):
            dLdy = layer.backward(dLdy)    
