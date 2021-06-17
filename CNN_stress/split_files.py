from typing import Iterable, Union
from pathlib import Path
import os
import pandas as pd
import typer
from shutil import copyfile, rmtree
from itertools import combinations


def split_data_best_combination(
    prop_df: pd.DataFrame, split_rate_range: Iterable[float] = (0.7, 0.8)
) -> tuple[Union[dict, float]]:
    videos_set = set(list(prop_df.index))

    total_images = prop_df["total"].sum()

    best_split = None
    best_split_diff = 1
    for comb in combinations(videos_set, len(videos_set) - 2):
        comb = set(comb)

        train = comb
        val = videos_set - comb

        stress_prob_train = (
            prop_df.loc[train, "stressed"].sum() / prop_df.loc[train, "total"].sum()
        )
        stress_prob_val = (
            prop_df.loc[val, "stressed"].sum() / prop_df.loc[val, "total"].sum()
        )

        comb_prop_diff = abs(stress_prob_train - stress_prob_val)

        split_rate = prop_df.loc[train, "total"].sum() / total_images

        if (
            comb_prop_diff < best_split_diff
            and split_rate_range[0] <= split_rate <= split_rate_range[1]
        ):
            best_split = {"train": train, "validation": val, "split_rate": split_rate}
            best_split_diff = comb_prop_diff

    return best_split, best_split_diff


def split_files_by_video(
    folder_path: Union[
        Path, str
    ] = "/home/users/ucadatalab_group/javierj/SHARED/baby-stress"
):
    non_stressed = pd.DataFrame(
        {"file_name": os.listdir(f"{folder_path}/non-stressed")}
    )
    non_stressed["stress"] = "non-stressed"

    stressed = pd.DataFrame({"file_name": os.listdir(f"{folder_path}/stressed")})
    stressed["stress"] = "stressed"

    images_df = pd.concat([non_stressed, stressed])

    images_df["video"] = images_df["file_name"].map(lambda x: x[0:2])

    videos_folder_path = f"{folder_path}/videos_images"
    try:
        # Create target Directory
        os.mkdir(videos_folder_path)
    except FileExistsError:
        print("Directory ", videos_folder_path, " already exists")

    proportions_df = pd.DataFrame()

    for video_name, group in images_df.groupby("video"):
        proportions_df = proportions_df.append(
            {
                "video": video_name,
                "stressed": (group["stress"] == "stressed").sum(),
                "total": len(group),
                "prop": (group["stress"] == "stressed").mean(),
            },
            ignore_index=True,
        )

    proportions_df = proportions_df.set_index("video")
    print(proportions_df)

    split_sets, stress_prop_difference = split_data_best_combination(proportions_df)
    print(f"\nTrain videos: {split_sets['train']}")
    print(f"Validation videos: {split_sets['validation']}")
    print(f"Split rate: {split_sets['split_rate']}")
    print(f"Stress proportion difference: {stress_prop_difference}")

    # Create training data folders
    training_data_path = f"{folder_path}/training_data"
    if os.path.exists(training_data_path):
        rmtree(f"{training_data_path}")

    os.mkdir(training_data_path)
    os.mkdir(f"{training_data_path}/train")
    os.mkdir(f"{training_data_path}/train/stressed")
    os.mkdir(f"{training_data_path}/train/non-stressed")
    os.mkdir(f"{training_data_path}/validation")
    os.mkdir(f"{training_data_path}/validation/stressed")
    os.mkdir(f"{training_data_path}/validation/non-stressed")

    for video_name, group in images_df.groupby("video"):
        if video_name in split_sets["train"]:
            group.apply(
                lambda img_row: copyfile(
                    f"{folder_path}/{img_row['stress']}/{img_row['file_name']}",
                    f"{training_data_path}/train/{img_row['stress']}/{img_row['file_name']}",
                ),
                axis=1,
            )
        else:
            group.apply(
                lambda img_row: copyfile(
                    f"{folder_path}/{img_row['stress']}/{img_row['file_name']}",
                    f"{training_data_path}/validation/{img_row['stress']}/{img_row['file_name']}",
                ),
                axis=1,
            )


def main(folder_path: str = "/home/users/ucadatalab_group/javierj/SHARED/baby-stress"):
    split_files_by_video(folder_path)


if __name__ == "__main__":
    typer.run(main)
