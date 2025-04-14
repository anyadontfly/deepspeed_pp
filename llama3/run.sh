#!/bin/bash

CUDA_VISIBLE_DEVICES=0,1 deepspeed train.py --deepspeed_config=ds_config.json -p 2 --steps=20