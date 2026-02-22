#!/bin/bash
MODAL_DATA_PATH=modal_data
# Check if we have model.npz
if [ ! -f "$MODAL_DATA_PATH/model.npz" ]; then
    echo "Error: model.npz not found in $MODAL_DATA_PATH. Please run the training script first."
    exit 1
fi

# Same for vocab.json and dict.json
if [ ! -f "$MODAL_DATA_PATH/vocab.json" ]; then
    echo "Error: vocab.json not found in $MODAL_DATA_PATH. Please run the training script first."
    exit 1
fi

# Check if we have bigram.json, if we don't, we don't pass the argument for it
BIGRAM_ARG=""
if [ -f "$MODAL_DATA_PATH/bigram.json" ]; then
    BIGRAM_ARG="--bigram $MODAL_DATA_PATH/bigram.json"
fi

# Now run the GUI demo with the appropriate arguments
python examples/gui_demo.py \
    --model $MODAL_DATA_PATH/model.npz \
    --vocab $MODAL_DATA_PATH/vocab.json \
    --dict $MODAL_DATA_PATH/dict.json \
    $BIGRAM_ARG
