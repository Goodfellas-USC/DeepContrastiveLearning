#!/bin/sh
allennlp predict "output_epch_10" "/home/gnn_research/good_fellas/DeCLUTR/test_data.jsonl" --output-file "embeddings_10_epoch.jsonl"  --batch-size 32 --use-dataset-reader --overrides "{'dataset_reader.num_anchors': null}"  --include-package "declutr"
