import json


def save_json_data_to_file(filename, json_data, message=None, log=None):
    if message is not None:
        log.info("Saving " + str(message) + " to " + filename + " ...")
        with open(filename, "w") as fh:
            json.dump(json_data, fh)
