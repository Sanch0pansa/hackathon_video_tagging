import torch
from transformers import BertModel, BertTokenizer

# Model BERT text to embeddings


class FeatureExtractor:
    def __init__(self, model_name='bert-base-uncased'):
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.model = BertModel.from_pretrained(
            model_name, output_hidden_states=True)

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self.model.to(device)

    def extract_features(self, text: str) -> torch.Tensor:
        # Tokenize input text
        inputs = self.tokenizer(text, return_tensors="pt",
                                padding=True, truncation=True, max_length=512)

        # Get embeddings
        with torch.no_grad():
            outputs = self.model(**inputs)

        # Extract embeddings from the layer
        embeddings = outputs.hidden_states[-2]

        # Average embeddings of all tokens
        sentence_embedding = torch.mean(embeddings, dim=1)

        return sentence_embedding

    def cosine_similarity(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.cosine_similarity(a, b, dim=1)
