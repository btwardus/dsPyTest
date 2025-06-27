import dspy

class ProductResolutionSignature(dspy.Signature):
    """
    Determines if two product records refer to the same item, providing an explanation for the decision.
    """
    
    product1 = dspy.InputField(desc="First product record with details.")
    product2 = dspy.InputField(desc="Second product record with details.")
    
    explanation = dspy.OutputField(desc="A detailed explanation of why the two products are the same or not the same.")
    label = dspy.OutputField(desc="Output ONLY 'Yes' or 'No' as the first line. Do NOT include explanations or extra text in the label line.") 