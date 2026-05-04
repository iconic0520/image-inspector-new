import h5py
import os

def fix_model_layer_names(model_path, output_path):
    """
    Renames layers in a Keras H5 model file to remove '/' characters, 
    which causes errors in newer Keras/TensorFlow versions.
    """
    if not os.path.exists(model_path):
        print(f"Error: Model not found at {model_path}")
        return

    # Copy the file to the output path first
    import shutil
    shutil.copy(model_path, output_path)

    with h5py.File(output_path, 'a') as f:
        if 'model_config' in f.attrs:
            config = f.attrs['model_config']
            # Convert to string if it's bytes
            if isinstance(config, bytes):
                config = config.decode('utf-8')
            
            # Replace '/' with '_' in the configuration string
            new_config = config.replace('conv1/conv', 'conv1_conv')
            new_config = new_config.replace('conv1/bn', 'conv1_bn')
            new_config = new_config.replace('conv1/relu', 'conv1_relu')
            # Add more replacements if needed, or use a regex for all /
            import re
            new_config = re.sub(r'"name":\s*"([^"]+)/([^"]+)"', r'"name": "\1_\2"', new_config)
            
            f.attrs['model_config'] = new_config.encode('utf-8')
            print("Successfully updated model_config in H5 file.")

        # Also need to update the group names in the H5 file itself if they contain slashes
        # However, usually it's the 'model_config' attribute that Keras 3.0+ checks first.
        # If it still fails, we might need to recreate the model architecture and load weights.

if __name__ == "__main__":
    fix_model_layer_names('ELA_Training/model_ela.h5', 'ELA_Training/model_ela_fixed.h5')
