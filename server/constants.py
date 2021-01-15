import json

cfg_id_mapping = {0: "configs/vqa_scratch.yaml", 1: "configs/loc_scratch.yaml"}

checkpoint_file_mapping = {0: "", 1: ""}

image_file_mapping = [
    f"/static/sample_images/CLEVR_test_0000{k:02}.png" for k in range(100)
]

sample_questions = json.load(open("server/static/CLEVR_test_questions.json", "r"))
