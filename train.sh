#!/bin/bash
ai-t9-train --vocab data/vocab.json --corpus corpuses/ --output data/model.npz --save-ngram data/bigram.json --device cuda --debug
