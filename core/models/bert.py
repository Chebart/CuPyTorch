from ..blocks import Embedding, TransformerEncoderLayer, Linear, Dropout, LayerNorm, GELU, Softmax
from .abstract_model import AbstractModel
from ..data import Tensor

class BertEmbeddings(AbstractModel):
    def __init__(
        self, 
        vocab_size: int,
        max_len: int, 
        d_model: int, 
        dropout: float = 0.1
    ):
        self.token_emb = Embedding(vocab_size, d_model)
        self.pos_emb = Embedding(max_len, d_model)

        self.ln = LayerNorm(d_model)
        self.dropout = Dropout(dropout)

    def forward(self, input_ids):
        _, T = input_ids.shape
        pos_ids = Tensor.arange(T, dtype = input_ids.dtype, device = input_ids.device)[None, :]

        x = (
            self.token_emb(input_ids)
            + self.pos_emb(pos_ids)
        )

        x = self.ln.forward(x)
        return self.dropout.forward(x)

    def backward(self, dLdy):
        dLdy = self.dropout.backward(dLdy)
        dLdy = self.ln.backward(dLdy)
        self.token_emb.backward(dLdy)
        self.pos_emb.backward(dLdy.sum(axis=0, keepdims=True))

    def parameters(self):
        return (
            self.token_emb.parameters() +
            self.pos_emb.parameters() +
            self.ln.parameters()
        )

    def to_device(self, device: str):
        self.token_emb.to_device(device)
        self.pos_emb.to_device(device)
        self.ln.to_device(device)
        self.dropout.to_device(device)
        return self

    def train(self):
        self.dropout.train()

    def eval(self):
        self.dropout.eval()

class MLMHead(AbstractModel):
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

    def parameters(self):
        return (
            self.fc1.parameters() +
            self.ln.parameters() +
            self.fc2.parameters()
        )

    def to_device(self, device: str):
        self.fc1.to_device(device)
        self.gelu.to_device(device)
        self.ln.to_device(device)
        self.fc2.to_device(device)
        return self

class BertClassifier(AbstractModel):
    def __init__(self, d_model):
        self.fc = Linear(d_model, 2)

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

    def parameters(self):
        return self.fc.parameters()

    def to_device(self, device: str):
        self.fc.to_device(device)
        return self

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
        self.softmax = Softmax()
        super().__init__()

    def set_task(self, task_name):
        self.task = task_name

    def generate_mask(self, input_ids):
        return (input_ids == 0)[:, None, None, :]

    def forward(self, input_ids):
        assert self.task is not None, "before call forward() set task for BERT"
        
        mask = self.generate_mask(input_ids)
        for _, layer in enumerate(self.layers):
            if isinstance(layer, BertEmbeddings):
                x = layer(input_ids)
            else:
                x = layer(x, mask) 

        if self.task == "mlm":
            x = self.mlm_head(x)
        elif self.task == "classification":
            x = self.classifier(x)

        return self.softmax(x)

    def backward(self, dLdy):
        assert self.task is not None, "before call backward() set task for BERT"

        dLdy = self.softmax.backward(dLdy)
        
        if self.task == "mlm":
            dLdy = self.mlm_head.backward(dLdy)
        elif self.task == "classification":
            # for classification, do not train the full model.
            dLdy = self.classifier.backward(dLdy)
            return

        for _, layer in enumerate(reversed(self.layers)):
            dLdy = layer.backward(dLdy)    

    def parameters(self):
        params = self.mlm_head.parameters() + self.classifier.parameters()
        for _, layer in enumerate(self.layers):
            params += layer.parameters()

        return params

    def to_device(self, device: str):
        self.classifier = self.classifier.to_device(device)
        self.mlm_head = self.mlm_head.to_device(device)
        for layer in self.layers:
            layer.to_device(device)

        return self
    
    # ----------------------
    # this methods works only for layers that use dropout
    # ----------------------
    def train(self):
        for _, layer in enumerate(self.layers):
            layer.train()

    def eval(self):
        for _, layer in enumerate(self.layers):
            layer.eval()