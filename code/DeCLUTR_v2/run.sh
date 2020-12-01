#!/bin/sh
allennlp train "training_config/declutr_small.jsonnet"  --serialization-dir "output_test" --include-package "declutr" -f
