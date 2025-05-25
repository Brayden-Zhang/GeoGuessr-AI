import torch
import tensorflow as tf
import tensorflowjs as tfjs
from vit_model import create_vit_model
import os

def convert_pytorch_to_tfjs():
    # Load the PyTorch model
    checkpoint = torch.load('model/checkpoints/best_vit_model.pth', map_location='cpu')
    model = create_vit_model(num_classes=checkpoint['model_state_dict']['classifier.weight'].shape[0])
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # Create a dummy input
    dummy_input = torch.randn(1, 3, 224, 224)

    # Export the model to ONNX
    torch.onnx.export(model, dummy_input, 'model/temp_model.onnx',
                     input_names=['input'],
                     output_names=['output'],
                     dynamic_axes={'input': {0: 'batch_size'},
                                 'output': {0: 'batch_size'}})

    # Convert ONNX to TensorFlow
    import onnx
    from onnx_tf.backend import prepare

    onnx_model = onnx.load('model/temp_model.onnx')
    tf_rep = prepare(onnx_model)
    tf_rep.export_graph('model/tf_model')

    # Convert TensorFlow model to TensorFlow.js format
    tfjs.converters.convert_tf_saved_model(
        'model/tf_model',
        'extension/model'
    )

    # Clean up temporary files
    os.remove('model/temp_model.onnx')
    import shutil
    shutil.rmtree('model/tf_model')

if __name__ == '__main__':
    convert_pytorch_to_tfjs() 