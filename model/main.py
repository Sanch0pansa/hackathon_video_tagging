import click
import os
import torch
from extractors.Extractor import Extractor
from classificators.mlp_classifier.Classificator import MultiTaskClassifier
from classificators.mlp_classifier.DataModule import VideoDataset
import pandas as pd
from tqdm import tqdm


def load_model(checkpoint_path, input_dim=1536, num_classes=None):
    # If the number of classes is not provided, load it from the checkpoint
    if num_classes is None:
        checkpoint = torch.load(checkpoint_path, weights_only=True)
        num_classes = checkpoint['hyper_parameters']['num_classes']
    
    # Load the model with the appropriate number of classes
    model = MultiTaskClassifier(input_dim=input_dim, num_classes=num_classes)
    checkpoint = torch.load(checkpoint_path, weights_only=True)
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()  # Set the model to evaluation mode
    return model

def get_num_classes(categories_file):
    # Read the categories file and get the number of unique categories
    categories_df = pd.read_csv(categories_file)
    categories_df['full_category'] = categories_df.apply(
        lambda row: ': '.join(filter(lambda x: str(x) != 'nan', 
                                     [row['Уровень 1 (iab)'], row['Уровень 2 (iab)'], row['Уровень 3 (iab)']])), 
        axis=1
    )
    
    return len(categories_df['full_category'].unique())

def predict_tags(model, tensor_path, categories_file, threshold=0.5):
    # Load the tensor for the video
    tensor = torch.load(tensor_path, weights_only=True)
    tensor = tensor.view(tensor.shape[1])  # Ensure the tensor has the correct shape (1D)
    
    # Run the tensor through the model to get predictions
    with torch.no_grad():
        preds = model(tensor.unsqueeze(0))  # Add batch dimension

    # Convert probabilities to binary labels (tags)
    predicted_labels = (preds > threshold).squeeze().cpu().numpy()
    

    # Load the list of categories
    categories_df = pd.read_csv(categories_file)
    categories_df['full_category'] = categories_df.apply(
        lambda row: ': '.join(filter(lambda x: str(x) != 'nan', 
                                     [row['Уровень 1 (iab)'], row['Уровень 2 (iab)'], row['Уровень 3 (iab)']])), 
        axis=1
    )
    category_names = categories_df['full_category'].tolist()

    # Convert binary labels to textual categories
    predicted_tags = [category_names[i] for i, val in enumerate(predicted_labels) if val == 1]

    return predicted_tags


def extract(video_path, title, description, output_tensor_path):
    # Getting the output directories
    video_directory = os.path.dirname(video_path)
    output_directory = os.path.dirname(output_tensor_path)

    # Defining extractor
    print("Defining extractor...")
    extractor = Extractor(
        video_directory,
        output_directory,
        use_video_embeddings=True,
        use_text_embeddings=True,
    )

    # Extracting features
    data = extractor(os.path.basename(video_path).split(".")[0], title, description, save=False, show_progress=True)

    # Saving features
    torch.save(data, output_tensor_path)


def extract_all(video_directory, data_table_path, output_directory):
    # Reading dataset
    dataset = pd.read_csv(data_table_path)
    
    # Defining extractor
    extractor = Extractor(
        video_directory,
        output_directory,
        use_video_embeddings=True,
        use_text_embeddings=True,
    )

    # Iterating over dataset and extracting features
    for _, row in tqdm(dataset.iterrows(), desc="Extracting features from dataset", total=len(dataset)):
        video_id = row["video_id"]
        title = row["title"]
        description = row["description"]

        if os.path.exists(os.path.join(video_directory, video_id + ".mp4")):
            extractor(video_id, title, description, save=True, add_embeddings_dimension=False)


def inference(features_path, save_to_file):
    checkpoint_path = MODEL_PATH  # Path to model checkpoint
    categories_file = TAGS_TABLE_PATH  # Path to CSV with categories
    
    # Get the number of classes from the categories file
    num_classes = get_num_classes(categories_file)
    
    # Load the trained model
    model = load_model(checkpoint_path, input_dim=MODEL_INPUT_SIZE, num_classes=num_classes)
    
    # Run inference and get textual tags
    predicted_tags = predict_tags(model, features_path, categories_file)
    
    # Output the result
    print(f"Predicted tags: {predicted_tags}")

    # Save the result
    if save_to_file:
        with open(save_to_file, 'w') as f:
            for tag in predicted_tags:
                f.write(tag + '\n')

def inference_all(features_dir, save_to_directory):
    checkpoint_path = MODEL_PATH  # Path to model checkpoint
    categories_file = TAGS_TABLE_PATH  # Path to CSV with categories

    # Get the number of classes from the categories file
    num_classes = get_num_classes(categories_file)
    
    # Load the trained model
    model = load_model(checkpoint_path, input_dim=MODEL_INPUT_SIZE, num_classes=num_classes)

    # Iterate through the feature files in the directory and run inference
    for filename in tqdm(os.listdir(features_dir)):
        if filename.endswith(".pt"):
            predicted_tags = predict_tags(model, os.path.join(features_dir, filename), categories_file)

            # Save the result
            if save_to_directory:
                with open(os.path.join(save_to_directory, os.path.basename(filename).split(".")[0] + ".txt"), 'w') as f:
                    for tag in predicted_tags:
                        f.write(tag + '\n')


# Constants
DEFAULT_OUTPUT_TENSOR_PATH = "output_tensor.pt"  # Default path for output tensor file
DEFAULT_TAGS_OUTPUT_PATH = "tags_output.txt"     # Default path for inference tags output
MODEL_PATH = "./checkpoints/final_model.ckpt"    # Path to the trained model
MODEL_INPUT_SIZE = 1536                          # Input size of the model
TAGS_TABLE_PATH = "./config/tags.csv"            # Path to the tags table

# Command 1: Extract features from a single video
@click.command(name="extract-features")
@click.argument('video_path', type=click.Path(exists=True, dir_okay=False))
@click.argument('title', type=str)
@click.argument('description', type=str)
@click.option('--output-tensor-path', '-o', type=click.Path(), default=DEFAULT_OUTPUT_TENSOR_PATH, 
              help="Path to save the extracted tensor. Default is output_tensor.pt")
def extract_features(video_path, title, description, output_tensor_path):
    """
    Extract features from a video file.
    Arguments:
        video_path: Path to the video file.
        title: Title of the video.
        description: Description of the video.
    Options:
        output_tensor_path: Optional path to save the extracted tensor.
    """
    extract(video_path, title, description, output_tensor_path)


# Command 2: Run inference using a feature file
@click.command(name="run-inference")
@click.argument('features_path', type=click.Path(exists=True, dir_okay=False))
@click.option('--save-to-file', '-s', type=click.Path(), default=None, 
              help="Optional path to save tags. If not provided, tags will be printed to stdout.")
def run_inference(features_path, save_to_file):
    """
    Run inference using the extracted features from a file.
    Arguments:
        features_path: Path to the file with extracted features.
    Options:
        save-to-file: Optional path to save the tags. If not provided, tags will be printed to stdout.
    """
    inference(features_path, save_to_file)


# Command 3: Full inference (combine feature extraction and inference)
@click.command(name="full-inference")
@click.argument('video_path', type=click.Path(exists=True, dir_okay=False))
@click.argument('title', type=str)
@click.argument('description', type=str)
@click.option('--output-tensor-path', '-o', type=click.Path(), default=DEFAULT_OUTPUT_TENSOR_PATH, 
              help="Path to save the extracted tensor. Default is output_tensor.pt")
@click.option('--save-to-file', '-s', type=click.Path(), default=None, 
              help="Optional path to save tags. If not provided, tags will be printed to stdout.")
def full_inference(video_path, title, description, output_tensor_path, save_to_file):
    """
    Perform both feature extraction and inference in one step.
    Arguments:
        video_path: Path to the video file.
        title: Title of the video.
        description: Description of the video.
    Options:
        output-tensor-path: Path to save the extracted tensor.
        save-to-file: Optional path to save the tags.
    """
    extract(video_path, title, description, output_tensor_path)
    inference(output_tensor_path, save_to_file)


# Command 4: Extract features from a directory of video files
@click.command(name="extract-features-dir")
@click.argument('directory', type=click.Path(exists=True, file_okay=False))
@click.argument('data_table_path', type=click.Path(exists=True, dir_okay=False))
@click.option('--output-dir', '-o', type=click.Path(), default=None,
              help="Directory to save the output tensors. If not provided, saves to the current directory.")
def extract_features_dir(directory, data_table_path, output_dir, ):
    """
    Extract features from all video files in a directory.
    Arguments:
        directory: Path to the directory with video files.
    Options:
        output-dir: Directory to save the output tensors. Default is the current directory.
    """
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    extract_all(directory, data_table_path, output_dir)
    pass


# Command 5: Run inference on a directory of feature files
@click.command(name="run-inference-dir")
@click.argument('features_dir', type=click.Path(exists=True, file_okay=False))
@click.option('--save-dir', '-s', type=click.Path(), default=None, 
              help="Optional directory to save the output tags. If not provided, prints to stdout.")
def run_inference_dir(features_dir, save_dir):
    """
    Run inference on all feature files in a directory.
    Arguments:
        features_dir: Path to the directory containing feature files.
    Options:
        save-dir: Directory to save the output tags. If not provided, tags will be printed to stdout.
    """
    if save_dir and not os.path.exists(save_dir):
        os.makedirs(save_dir)
    inference_all(features_dir, save_dir)
    pass


# CLI entry point
@click.group()
def cli():
    """
    Command-line interface for video feature extraction and inference.
    """
    pass


# Add all commands to the CLI group
cli.add_command(extract_features)
cli.add_command(run_inference)
cli.add_command(full_inference)
cli.add_command(extract_features_dir)
cli.add_command(run_inference_dir)

if __name__ == "__main__":
    cli()
