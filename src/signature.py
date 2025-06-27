import dspy

class ProductResolutionSignature(dspy.Signature):
    """
    Determines if two product records refer to the same item.
    """
    
    product1 = dspy.InputField(desc="First product record with details.")
    product2 = dspy.InputField(desc="Second product record with details.")
    
    label = dspy.OutputField(desc="Output ONLY 'Yes' or 'No' as the first line. Do NOT include explanations or extra text in the label line.") 