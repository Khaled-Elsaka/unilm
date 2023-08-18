model_dir = '/path/to/pretrained/model/directory'
model = "english-arabic-model.pt"
spm = '/path/to/sentencepiece/model'

# from fairseq.models.transformer import TransformerModel
import torch
from deltalm.models.deltalm import DeltaLMModel

# Create a new DeltaLM model instance
model = DeltaLMModel.from_pretrained(
    model_dir,
    checkpoint_file=model,
    bpe='sentencepiece',
    sentencepiece_model=spm,
    task = "translation"
)
# Set the model to evaluation mode
model.eval()

# Translate a sentence from German to English
src_sentence = 'Hello, world!'
tgt_sentence = model.translate(src_sentence, beam=5)
print(tgt_sentence)
