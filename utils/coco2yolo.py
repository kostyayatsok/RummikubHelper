import json

json_path = ""
images_dir = ""

if __name__ == "__main__":
    with open(json_path) as f:
        data = json.load(f)
    image_id_to_file_name = {}
    for image in data["images"]:
        image_id_to_file_name[image["id"]] = image["file_name"]
    for annotation in data["annotations"]:
        filename = image_id_to_file_name[annotation["image_id"]]
        filename = annotation[:-3] + 'txt'
        with open(filename, "a") as f:
            f.write()
    