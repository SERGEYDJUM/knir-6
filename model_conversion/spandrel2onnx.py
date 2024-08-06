import argparse
import torch
from spandrel import ImageModelDescriptor, ModelLoader


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="Spandrel2ONNX",
        description="Converts any Spandrel-supported PyTorch model to ONNX graph",
    )

    parser.add_argument("input", type=str, help="Path to a Spandrel PyTorch model")
    parser.add_argument("size", type=int, help="Target resolution of input images")
    parser.add_argument(
        "--opset", type=int, default=20, help="Target ONNX opset (default is 20)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="",
        help="Path to the final ONNX file (default is {input}.onnx)",
    )

    args = parser.parse_args()

    if args.output == "":
        args.output = args.input + ".onnx"

    return args


if __name__ == "__main__":
    args = get_args()

    model = ModelLoader().load_from_file(args.input)
    assert isinstance(model, ImageModelDescriptor)
    model.eval()

    print(f"[info] Input model architecture: {model.architecture.name}")

    input_tensor = torch.rand(1, 3, args.size, args.size)

    print(f"[info] Target input tensor size: {list(input_tensor.size())}")

    print(f"[info] Exporting model...", flush=True)

    with torch.no_grad():
        torch.onnx.export(
            model.model,
            input_tensor,
            args.output,
            opset_version=args.opset,
            input_names=["input"],
            output_names=["output"],
        )

    print(f"[info] ONNX model written to `{args.output}`")
