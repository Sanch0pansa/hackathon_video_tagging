import click
import os
import torch
from extractors.Extractor import Extractor


# Constants
DEFAULT_OUTPUT_TENSOR_PATH = "output_tensor.pt"  # Default path for output tensor file
DEFAULT_TAGS_OUTPUT_PATH = "tags_output.txt"     # Default path for inference tags output

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
    print(features_path, save_to_file)
    pass


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
    print(video_path, title, description, output_tensor_path, save_to_file)
    pass


# Command 4: Extract features from a directory of video files
@click.command(name="extract-features-dir")
@click.argument('directory', type=click.Path(exists=True, file_okay=False))
@click.option('--output-dir', '-o', type=click.Path(), default=None,
              help="Directory to save the output tensors. If not provided, saves to the current directory.")
def extract_features_dir(directory, output_dir):
    """
    Extract features from all video files in a directory.
    Arguments:
        directory: Path to the directory with video files.
    Options:
        output-dir: Directory to save the output tensors. Default is the current directory.
    """
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    print(directory, output_dir)
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
    print(features_dir, save_dir)
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
