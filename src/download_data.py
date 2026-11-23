"""Download the data subset. Run `copernicusmarine login` first."""
import copernicusmarine as cm
import config as C


def main():
    C.DATA_DIR.mkdir(parents=True, exist_ok=True)
    cm.subset(
        dataset_id=C.DATASET_ID,
        variables=C.VARIABLES,
        minimum_longitude=C.BBOX["lon"][0], maximum_longitude=C.BBOX["lon"][1],
        minimum_latitude=C.BBOX["lat"][0], maximum_latitude=C.BBOX["lat"][1],
        minimum_depth=C.DEPTH[0], maximum_depth=C.DEPTH[1],
        start_datetime=C.TIME[0], end_datetime=C.TIME[1],
        output_filename=C.RAW_FILE.name, output_directory=str(C.DATA_DIR),
    )
    print("saved", C.RAW_FILE)


if __name__ == "__main__":
    main()
