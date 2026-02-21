#!/bin/bash
ai-t9-build-vocab \
	--output data \
	--corpus corpuses/

ai-t9-train \
	--vocab data/vocab.json \
	--corpus corpuses/ \
	--output data/model.npz \
	--save-ngram data/bigram.npz \
	--embed-dim 264 \
	--device cuda \
	--debug \
	--save-pairs data/pairs.npz \
	--pairs-only \
	$@
