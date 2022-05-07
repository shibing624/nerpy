# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: pip install fastapi uvicorn
"""
import argparse
import uvicorn
import sys
import os
from fastapi import FastAPI, Query
from starlette.middleware.cors import CORSMiddleware
from loguru import logger

sys.path.append('..')
from nerpy import NERModel

pwd_path = os.path.abspath(os.path.dirname(__file__))
# Use fine-tuned model
parser = argparse.ArgumentParser()
parser.add_argument("--model_name_or_path", type=str, default="shibing624/bert4ner-base-chinese",
                    help="Model save dir or model name")
args = parser.parse_args()
s_model = NERModel('bert', args.model_name_or_path)

# define the app
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"])


@app.get('/')
async def index():
    return {"message": "index, docs url: /docs"}


@app.get('/entity')
async def entity(q: str = Query(..., min_length=1, max_length=128, title='query')):
    try:
        preds, outputs, entities = s_model.predict([q], split_on_space=False)
        result_dict = {'entity': entities}
        logger.debug(f"Successfully get sentence entity, q:{q}")
        return result_dict
    except Exception as e:
        logger.error(e)
        return {'status': False, 'msg': e}, 400


if __name__ == '__main__':
    uvicorn.run(app=app, host='0.0.0.0', port=8001)
