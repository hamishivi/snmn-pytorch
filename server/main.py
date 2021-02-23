import os
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
import numpy as np
import urllib.request

from server.predict import predict_sample
from server.constants import (
    sample_questions,
    image_file_mapping,
    checkpoint_file_mapping,
    cfg_id_mapping,
    module_descriptions,
    readable_mapping,
    vqa_scratch_url,
    vqa_gt_layout_url,
)
from nmn import MODULE_INPUT_NUM, MODULE_OUTPUT_NUM


app = FastAPI()

app.mount("/static", StaticFiles(directory="server/static"), name="static")
templates = Jinja2Templates(directory="server/templates")


@app.get("/", response_class=HTMLResponse)
async def get_home(request: Request):
    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "question_texts": sample_questions,
            "image_filenames": image_file_mapping,
        },
    )


# a little helper function to determine the i/o
# for each module, so we can visualise it later!
def module_inputs(module_names):
    values_used = []
    stacks = []
    stack = []
    output = []
    for i, module in enumerate(module_names):
        num_inputs = MODULE_INPUT_NUM[module]
        num_outputs = MODULE_OUTPUT_NUM[module]
        # module validity is not enforced by the default code,
        # so sometimes we get invalid stack use.
        output.append(num_outputs)
        values_used.append([])
        # stack doesnt change.
        if module == "_DescribeOne" or module == "_DescribeTwo":
            for j in range(1, num_inputs + 1):
                if len(stack) >= j:
                    values_used[-1].append(stack[-j])
        else:
            for j in range(num_inputs):
                if len(stack) > 0:
                    values_used[-1].append(stack.pop())
            for j in range(num_outputs):
                stack.append(i + 1)
            # save stack
            stacks.append([x for x in stack])
    return values_used, stacks, output


@app.get("/model")
def predict(image_id: int = 0, question_text: str = "blank", gt: int = 0):
    assert gt in [0, 1], "gt value must be 0 or 1"
    assert image_id in range(0, 100), "Invalid image id. Must be in range 0-99."
    from config import cfg

    # download models if havent
    if gt == 0 and not os.path.exists("server/static/models/vqa_gt_layout.ckpt"):
        urllib.request.urlretrieve(
            vqa_gt_layout_url, "server/static/models/vqa_gt_layout.ckpt"
        )
    if gt == 1 and not os.path.exists("server/static/models/vqa_scratch.ckpt"):
        urllib.request.urlretrieve(
            vqa_scratch_url, "server/static/models/vqa_scratch.ckpt"
        )

    cfg.merge_from_file(cfg_id_mapping[0])
    cfg.freeze()
    res = predict_sample(
        cfg,
        checkpoint_file_mapping[gt],
        question_text,
        "server" + image_file_mapping[image_id],
    )
    # need to post-process res to make output reasonable!
    # output: answer text,
    answer = res["answer_dict"].idx2word(np.argmax(res["answer_probs"]))
    module_list = [
        res["module_dict"].idx2word(x) for x in np.argmax(res["module_probs"], 1)
    ]
    module_desc = [module_descriptions[d] for d in module_list]
    inputs, stacks, output = module_inputs(module_list)
    # make module names readable
    module_list = [readable_mapping[n] for n in module_list]
    return {
        "answer": answer,
        "module_list": list(zip(module_list, module_desc)),
        "question_tokens": res["qtokens"],
        "question_attns": res["qattns"],
        "image_attns": res["iattns"],
        "module_inputs": inputs,
        "stacks": stacks,
        "outputs": output,
    }
