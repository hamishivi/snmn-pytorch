from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
import numpy as np
import skimage

from server.predict import predict_sample
from server.constants import (
    sample_questions,
    image_file_mapping,
    checkpoint_file_mapping,
    cfg_id_mapping,
)


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


@app.get("/model")
def predict(cfg_id: int = 0, image_id: int = 0, question_text: str = "blank"):
    assert cfg_id in cfg_id_mapping, "Invalid config id. Try 0 or 1."
    assert image_id in range(0, 100), "Invalid image id. Must be in range 0-99."
    from config import cfg

    cfg.merge_from_file(cfg_id_mapping[cfg_id])
    cfg.freeze()
    res = predict_sample(
        cfg,
        checkpoint_file_mapping[cfg_id],
        question_text,
        "server" + image_file_mapping[image_id],
    )
    # need to post-process res to make output reasonable!
    # output: answer text,
    answer = res["answer_dict"].idx2word(np.argmax(res["answer_probs"]))
    module_list = [
        res["module_dict"].idx2word(x) for x in np.argmax(res["module_probs"], 1)
    ]
    return {
        "answer": answer,
        "module_list": module_list,
        "question_tokens": res["qtokens"],
        "question_attns": res["qattns"],
        "image_attns": res["iattns"],
    }
