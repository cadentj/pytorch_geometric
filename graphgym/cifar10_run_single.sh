#!/usr/bin/env bash

# Test for running a single experiment. --repeat means run how many different random seeds.
python main.py --cfg configs/pyg/cifar10_graph.yaml --repeat 1 # graph classification