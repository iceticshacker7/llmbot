import openai
import pandas as pd

# Function to read the CSV file
def read_csv(file_path):
    return pd.read_csv(file_path)

def train_model_on_chunks(data_chunks, api_key):
    openai.api_key = api_key
    for chunk in data_chunks:
        # Convert DataFrame to list of dictionaries
        messages = [{"role": "user", "content": row["Question"]} for _, row in chunk.iterrows()]
        # Train the model on each chunk
        openai.ChatCompletion.create(
            model="text-davinci-003",
            messages=messages,
            max_tokens=150,
            temperature=0.7,
            stop=["\n"]
        )
# Function to interactively chat with the trained model
def chat_with_model(api_key):
    openai.api_key = api_key
    while True:
        user_input = input("You: ")
        if user_input.lower() == 'exit':
            break
        response = openai.Completion.create(
            model="text-davinci-003",
            prompt=user_input + "\n",
            max_tokens=50,
            temperature=0.7,
            stop=["\n"]
        )
        print("Bot:", response.choices[0].text.strip())

# Main function
def main(csv_file_path, api_key):
    # Read the CSV file
    data = read_csv(csv_file_path)

    # Convert data to chunks for training
    data_chunks = [data[i:i+5] for i in range(0, len(data), 5)]

    # Train the model on chunks
    train_model_on_chunks(data_chunks, api_key)

    # Interactively chat with the trained model
    chat_with_model(api_key)

if __name__ == "__main__":
    # Provide the path to your CSV file
    csv_file_path = "C:/Users/hp/Documents/Airline_review.csv"

    # Provide your OpenAI API key
    api_key = "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"

    main(csv_file_path, api_key)
