import os
import glob
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
from potassium import App, Route, JSONResponse

# Function to extract text from a PDF file
def extract_text_from_pdf(file):
    # TODO: Implement PDF text extraction
    pass


# Function to load data from a directory of PDF files
def load_data(data_dir: str) -> torch.Tensor:
    # Find all PDF files in the specified directory
    pdf_files = glob.glob(os.path.join(data_dir, "*.pdf"))

    # Extract text from each PDF file and concatenate into a single string
    documents = []
    for pdf_file in pdf_files:
        with open(pdf_file, "rb") as f:
            pdf_text = extract_text_from_pdf(f)
            documents.append(pdf_text)

    # Tokenize the concatenated text using the GPT-Neo tokenizer
    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neo-2.7B")
    inputs = tokenizer('\n\n'.join(documents), return_tensors='pt', padding=True, truncation=True)

    # Return the input IDs as a PyTorch tensor
    return inputs["input_ids"]


# Function to fine-tune the GPT-Neo model on a dataset of PDFs
def fine_tune_model(train_dataset: torch.Tensor) -> None:
    # Load the GPT-Neo model
    model = AutoModelForCausalLM.from_pretrained("EleutherAI/gpt-neo-2.7B")

    # Define the training arguments
    training_args = TrainingArguments(
        output_dir='./results',                      # Directory to save the fine-tuned model
        num_train_epochs=5,                          # Number of training epochs
        per_device_train_batch_size=8,               # Batch size per GPU during training
        per_device_eval_batch_size=8,                # Batch size per GPU during evaluation
        logging_steps=5000,                          # Log every n steps
        save_steps=10000,                            # Save checkpoint every n steps
        evaluation_strategy="epoch",                 # Evaluate every epoch
    )

    # Define the trainer object with the model, training arguments, and train dataset
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
    )

    # Train the model
    trainer.train()

    # Save the fine-tuned model
    model.save_pretrained("./fine_tuned_model")


# Define the Flask app using the Potassium library
app = App()

# Define a route for the fine-tune endpoint
@app.route("/fine_tune", methods=["POST"])
def fine_tune(request):
    # Get the data directory from the JSON payload
    data_dir = request.json["data_dir"]

    # Load the data from the specified directory
    train_dataset = load_data(data_dir)

    # Fine-tune the model on the data
    fine_tune_model(train_dataset)

    # Return a JSON response indicating that fine-tuning is complete
    return JSONResponse({"message": "Fine-tuning completed."})


# Start the Flask app
if __name__ == "__main__":
    app.run()
