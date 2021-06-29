from stress_model import CNN
import typer


def main(input_path: str, output_path: str):
    model = CNN(input_path)
    model.to_blob(output_path)


if __name__ == "__main__":
    typer.run(main)
