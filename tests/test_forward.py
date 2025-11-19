import sys, pathlib
ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import get_config
from src.model import get_model, get_tokenizer, encode_text

def test_forward_pass():
    config = get_config()
    tokenizer = get_tokenizer(config)
    model = get_model(config)
    sentence = "Aspirin caused a mild headache but no nausea."
    enc = encode_text(sentence, tokenizer, config.max_seq_len)
    outputs = model(**enc)
    assert outputs.last_hidden_state.shape[0] == 1
