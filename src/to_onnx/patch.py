from datetime import datetime, timezone
from typing import Optional

import onnx

import config as cfg
from src.to_onnx.version import encode_semver


def patch_onnx_model(onnx_model: onnx.ModelProto):
    """
    **This will patch the model in-place without making a copy!**

    Patch the passed ONNX model with:

    * `raw_input` input that accepts the raw 3 channel RGB image
    * `decoded_output` output that is the decoded output of the model
    """

    # region Add raw image input tensor
    input_tensor = onnx_model.graph.input[0]

    raw_input_tensor = onnx.helper.make_tensor_value_info(
        "raw_image",
        onnx.TensorProto.UINT8,
        [cfg.MODEL.image_height, cfg.MODEL.image_width, 3],
    )

    # image = np.transpose(image, (2, 0, 1))
    raw_input_transpose_node = onnx.helper.make_node(
        "Transpose",
        inputs=[raw_input_tensor.name],
        outputs=["raw_input_transpose"],
        perm=[2, 0, 1],
    )

    # image = image[None, ...]
    unsqueeze_tensor = onnx.helper.make_node(
        "Constant",
        inputs=[],
        outputs=["axes"],
        value=onnx.helper.make_tensor(
            "axes", onnx.TensorProto.INT64, dims=[1], vals=[0]
        ),
    )
    raw_input_unsqueeze_node = onnx.helper.make_node(
        "Unsqueeze",
        inputs=[raw_input_transpose_node.output[0], unsqueeze_tensor.output[0]],
        outputs=["raw_input_unsqueeze"],
        # axes=[0],
    )

    # image = image.astype(np.float32)
    raw_input_cast_node = onnx.helper.make_node(
        "Cast",
        inputs=[raw_input_unsqueeze_node.output[0]],
        outputs=[input_tensor.name],
        to=onnx.TensorProto.FLOAT,
    )

    onnx_model.graph.node.extend(
        [
            raw_input_transpose_node,
            unsqueeze_tensor,
            raw_input_unsqueeze_node,
            raw_input_cast_node,
        ]
    )
    onnx_model.graph.input.pop()
    onnx_model.graph.input.extend([raw_input_tensor])
    # endregion

    # region Add decoded output tensor
    output_tensor = onnx_model.graph.output[0]

    # torch_outputs.permute(1, 0, 2)
    permute_node = onnx.helper.make_node(
        "Transpose",
        inputs=[output_tensor.name],
        outputs=["decode_permute_output"],
        perm=[1, 0, 2],
    )

    # torch.softmax(torch_outputs, 2)
    softmax_node = onnx.helper.make_node(
        "Softmax",
        inputs=[permute_node.output[0]],
        outputs=["decode_softmax_output"],
        axis=2,  # Apply softmax along axis 2
    )

    # torch.argmax(torch_outputs, 2)
    argmax_node = onnx.helper.make_node(
        "ArgMax",
        inputs=[softmax_node.output[0]],
        outputs=["decoded_output"],
        axis=2,  # Compute argmax along axis 2
        keepdims=0,  # Remove the axis
    )

    onnx_model.graph.node.extend([permute_node, softmax_node, argmax_node])

    # Update the model's output tensor
    onnx_model.graph.output.extend(
        [
            onnx.helper.make_tensor_value_info(
                argmax_node.output[0],
                onnx.TensorProto.INT64,
                [output_tensor.type.tensor_type.shape.dim[1].dim_value, 1],
            )
        ]
    )
    # endregion


def write_onnx_custom_metadata(
    onnx_model: onnx.ModelProto,
    version: tuple[int, int, int] = (1, 0, 0),
    graph_name: str = "crnn_patched",
    built_timestamp: Optional[int] = None,
):
    onnx_model.graph.name = graph_name
    onnx_model.model_version = encode_semver(*version)

    if built_timestamp is not None:
        built_timstamp_metadata = onnx_model.metadata_props.add()
        built_timstamp_metadata.key = "built_timestamp"
        built_timstamp_metadata.value = str(built_timestamp)

    patched_timstamp_metadata = onnx_model.metadata_props.add()
    patched_timstamp_metadata.key = "patched_timestamp"
    patched_timstamp_metadata.value = str(int(datetime.now(timezone.utc).timestamp()))

    image_width_metadata = onnx_model.metadata_props.add()
    image_width_metadata.key = "image_width"
    image_width_metadata.value = str(cfg.MODEL.image_width)

    image_height_metadata = onnx_model.metadata_props.add()
    image_height_metadata.key = "image_height"
    image_height_metadata.value = str(cfg.MODEL.image_height)

    blank_token_metadata = onnx_model.metadata_props.add()
    blank_token_metadata.key = "blank_token"
    blank_token_metadata.value = cfg.MODEL.blank_token

    pad_token_metadata = onnx_model.metadata_props.add()
    pad_token_metadata.key = "pad_token"
    pad_token_metadata.value = cfg.MODEL.pad_token
