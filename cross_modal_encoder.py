from transformers import CLIPModel, CLIPProcessor


class CrossModalEncoder:
    def __init__(self, model_name="openai/clip-vit-base-patch32"):
        """
        Initializes the CrossModalEncoder with the specified CLIP model.
        Args:
        model_name: The name of the pretrained CLIP model to use.
        """
        self.model = CLIPModel.from_pretrained(model_name)
        self.processor = CLIPProcessor.from_pretrained(model_name)
        self.output_feature_length = self.model.config.projection_dim

    def encode_text(self, text):
        """
        Encodes the given text into a feature vector.
        Args:
        text: A string or list of strings to encode.
        Returns:
        A normalized tensor representing the text features.
        """
        inputs = self.processor(text=[text] if isinstance(text, str) else text, return_tensors="pt", truncation=True)

        text_features = self.model.get_text_features(**inputs)
        return text_features / text_features.norm(dim=-1, keepdim=True)

    def encode_image(self, image):
        """
        Encodes the given image into a feature vector.
        Args:
        image: A PIL Image or a list of PIL Images to encode.
        Returns:
        A normalized tensor representing the image features.
        """
        inputs = self.processor(images=image, return_tensors="pt")
        image_features = self.model.get_image_features(**inputs)
        return image_features / image_features.norm(dim=-1, keepdim=True)


encoder = CrossModalEncoder()
