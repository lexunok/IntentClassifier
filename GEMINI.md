# Project Overview

This project is a command-line application for intent classification built with C# and .NET 8. It uses the `TorchSharp` library for deep learning and `Tokenizers.DotNet` for text tokenization. The core of the project is a simple neural network model that can be trained to classify text into different intents.

The project is structured into several key components:
- **`Program.cs`**: The main entry point that handles command-line arguments for training the model (`train`) and running inference (`infer`).
- **`SimpleClassifier.cs`**: Defines the neural network architecture, which consists of an embedding layer followed by two fully-connected layers.
- **`Trainer.cs`**: Manages the model training process, including data loading, splitting, training, validation, and saving the best model.
- **`Predictor.cs`**: Loads a pre-trained model to perform inference on new text inputs.
- **`DataLoader.cs`**: Provides utilities for loading data from CSV files, splitting it into training and validation sets, and creating batches.
- **`TokenizerWrapper.cs`**: A wrapper for the `Tokenizers.DotNet` library to handle text encoding.

# Building and Running

## Prerequisites
- .NET 8 SDK
- A `tokenizer.json` file is required for the tokenizer to work.
- A `data/train.csv` file is required for training. If not present, a tiny synthetic sample will be created.

## Building the project
To build the project, run the following command:
```bash
dotnet build
```

## Running the application
The application has two main commands: `train` and `infer`.

### Training the model
To train the model, run the following command:
```bash
dotnet run -- train
```
This will:
1. Load the training data from `data/train.csv`.
2. Split the data into training and validation sets.
3. Train the `SimpleClassifier` model.
4. Save the best model to the `checkpoints` directory.

### Running inference
To run inference on a piece of text, use the `infer` command:
```bash
dotnet run -- infer "Your text here"
```
This will load the trained model from the `checkpoints` directory and output the predicted intent for the given text.

# Development Conventions

- The project follows standard C# coding conventions.
- The model architecture is defined in `SimpleClassifier.cs` and is a simple feed-forward network.
- The training process includes validation and early stopping to prevent overfitting.
- The project uses `System.Text.Json` for handling JSON configuration.
