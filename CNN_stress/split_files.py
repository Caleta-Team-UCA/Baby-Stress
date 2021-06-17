from typing import Union
from pathlib import Path
import os
import pandas as pd
import typer
from shutil import copyfile, rmtree


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
        new_video_folder = f"{videos_folder_path}/{video_name}"

        # Delete folder and create it again
        if os.path.exists(new_video_folder):
            rmtree(f"{new_video_folder}")

        os.mkdir(f"{new_video_folder}")
        os.mkdir(f"{new_video_folder}/non-stressed")
        os.mkdir(f"{new_video_folder}/stressed")

        group.apply(
            lambda img_row: copyfile(
                f"{folder_path}/{img_row['stress']}/{img_row['file_name']}",
                f"{new_video_folder}/{img_row['stress']}/{img_row['file_name']}",
            ),
            axis=1,
        )

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


def main(folder_path: str = "/home/users/ucadatalab_group/javierj/SHARED/baby-stress"):
    split_files_by_video(folder_path)


if __name__ == "__main__":
    typer.run(main)
