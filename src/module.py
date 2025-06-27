import dspy
from .signature import ProductResolutionSignature

class ProductResolutionModule(dspy.Module):
    def __init__(self):
        super().__init__()
        self.predictor = dspy.ChainOfThought(ProductResolutionSignature)

    def forward(self, product1, product2):
        result = self.predictor(product1=product1, product2=product2)
        return result 