from tensorflow.keras.backend import set_learning_phase
from tensorflow.keras.models import load_model
import tensorflow as tf
from tensorflow.python.framework import graph_io
import platform
import subprocess
import typer


def freeze_graph(
    graph,
    session,
    output,
    save_pb_dir=".",
    save_pb_name="frozen_model.pb",
    save_pb_as_text=False,
):
    with graph.as_default():
        graphdef_inf = tf.graph_util.remove_training_nodes(graph.as_graph_def())
        graphdef_frozen = tf.graph_util.convert_variables_to_constants(
            session, graphdef_inf, output
        )
        graph_io.write_graph(
            graphdef_frozen, save_pb_dir, save_pb_name, as_text=save_pb_as_text
        )
        return graphdef_frozen


def to_pb(model_path: str):
    # Convert .h5 to .pb
    set_learning_phase(0)
    model = load_model(model_path)
    session = tf.compat.v1.keras.backend.get_session()

    INPUT_NODE = [t.op.name for t in model.inputs]
    OUTPUT_NODE = [t.op.name for t in model.outputs]
    print(INPUT_NODE, OUTPUT_NODE)

    frozen_graph = freeze_graph(
        session.graph,
        session,
        [out.op.name for out in model.outputs],
        save_pb_name=model_path.replace("h5", "pb"),
    )


def open_vino(model_path: str, mo_tf_path: str):
    to_pb(model_path)

    pb_file = model_path.replace("h5", "pb")
    output_dir = model_path.replace("h5", "")
    img_height = 224
    input_shape = [1, img_height, img_height, 3]
    input_shape_str = str(input_shape).replace(" ", "")

    subprocess.Popen(
        f"python {mo_tf_path} --input_model {pb_file} --output_dir {output_dir} --input_shape {input_shape_str} --data_type FP32",
        shell=True,
    )


def main(model_path: str, mo_tf_path: str):
    open_vino(model_path, mo_tf_path)


if __name__ == "__main__":
    typer.run(main)
