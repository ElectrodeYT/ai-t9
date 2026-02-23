#!/bin/bash
ai-t9-train \
	--vocab data/vocab.json \
	--output data/model.npz \
	--embed-dim 32 \
	--device cuda \
	--debug \
	--load-pairs data/pairs.npz \
	$@
