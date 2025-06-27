import os
import openai
from dotenv import load_dotenv

def test_openai_chat_completion():
    """
    Tests making a paid chat completion request to the OpenAI API.
    """
    print("--- Testing OpenAI API Chat Completion ---")
    
    try:
        # Load environment variables from .env file
        load_dotenv()
        
        # Get the API key
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key or api_key == "your_openai_api_key_here":
            print("ERROR: OPENAI_API_KEY not found or not set in .env file.")
            return

        # Initialize the OpenAI client
        client = openai.OpenAI(api_key=api_key)
        
        # Make a simple chat completion call
        model_to_test = "gpt-4.1" # The model you are trying to use in main.py
        print(f"Making a test call to the chat completion endpoint with model: {model_to_test}...")
        
        response = client.chat.completions.create(
            model=model_to_test,
            messages=[
                {"role": "user", "content": "Hello, world!"}
            ]
        )
        
        # Check if we received a response
        if hasattr(response, 'choices') and response.choices:
            print("\nSUCCESS: Chat completion call was successful.")
            print(f"Response: {response.choices[0].message.content}")
        else:
            print("\nWARNING: API call successful, but no response was generated.")

    except openai.AuthenticationError:
        print("\nERROR: Authentication failed. Your API key is incorrect.")
        
    except openai.RateLimitError as e:
        print("\nERROR: Rate limit or quota exceeded. This is the likely cause of the issue.")
        print("Please check your plan and billing details on the OpenAI platform.")
        print(f"Error details: {getattr(e, 'body', str(e))}")

    except openai.NotFoundError as e:
        print(f"\nERROR: Model '{model_to_test}' not found.")
        print("This could mean you do not have access to this model.")
        print(f"Error details: {getattr(e, 'body', str(e))}")

    except openai.APIConnectionError:
        print("\nERROR: Failed to connect to the OpenAI API. Check your network.")

    except Exception as e:
        print(f"\nAn unexpected error occurred: {e}")
        
    finally:
        print("\n--- Test Complete ---")


if __name__ == "__main__":
    test_openai_chat_completion() 