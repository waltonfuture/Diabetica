#!/bin/bash

python evaluation/dialog_eval.py --model qwen --path WaltonFuture/Diabetica-7B

python evaluation/dialog_score.py --model qwen 