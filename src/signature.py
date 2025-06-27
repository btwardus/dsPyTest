import dspy

class ProductResolutionSignature(dspy.Signature):
    """
    Determines if two product records refer to the same item.
    """
    
    product1 = dspy.InputField(desc="First product record with details.")
    product2 = dspy.InputField(desc="Second product record with details.")
    
    # Add an explanation field for ChainOfThought
    explanation = dspy.OutputField(desc="A short, step-by-step explanation of the reasoning process.")
    confidence = dspy.OutputField(desc="A number 0-100 indicating confidence the two products are the same.")
    label = dspy.OutputField(desc="Output ONLY 'Yes' or 'No' as the first line. Do NOT include explanations or extra text in the label line.") 