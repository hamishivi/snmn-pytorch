import json

vqa_scratch_url = "https://drive.google.com/uc?id=1d81Jl_xwVqlbUhY8qaXtnmdMs0rwICVZ"
vqa_gt_layout_url = "https://drive.google.com/uc?id=1jTL27u0Ie0gBUc1Id3REP-DtJfMv0l-M"

cfg_id_mapping = {0: "configs/vqa_scratch.yaml", 1: "configs/loc_scratch.yaml"}

checkpoint_file_mapping = {
    1: "server/static/models/vqa_scratch.ckpt",
    0: "server/static/models/vqa_gt_layout.ckpt",
}

image_file_mapping = [
    f"/static/sample_images/CLEVR_test_0000{k:02}.png" for k in range(100)
]

sample_questions = json.load(open("server/static/CLEVR_test_questions.json", "r"))

# from original description
module_descriptions = {
    "_NoOp": "Doesn't do anything (i.e. nothing is updated in this timestep).",  # NoQA
    "_Find": "Looks at new image regions based on attended text.",  # NoQA
    "_Transform": "Shifts the image attention to somewhere new, conditioned on its previous glimpse.",  # NoQA
    "_Filter": "Tries to select out some image regions from where it looked before (based on attended text).",  # NoQA
    "_And": "Takes the intersection of the program's two previous glimpses as inputs, returning their intersection.",  # NoQA
    "_Or": "Takes the union of the program's two previous glimpses as inputs, returning their union.",  # NoQA
    "_Scene": "Tries to look at some objects in the image.",  # NoQA
    "_DescribeOne": "Takes the program's previous glimpse as input, and tries to infer the answer from it.",  # NoQA
    "_DescribeTwo": "Takes the program's two previous glimpses as inputs, and tries to infer the answer from them.",  # NoQA
}

readable_mapping = {
    "_NoOp": "No-op",
    "_Find": "Find",
    "_Transform": "Transform",
    "_Filter": "Filter",
    "_And": "And",
    "_Or": "Or",
    "_Scene": "Scene",
    "_DescribeOne": "Answer",
    "_DescribeTwo": "Compare",
}
