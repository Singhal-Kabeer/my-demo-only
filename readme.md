# CS781 Capstone: In-House Legal Intelligence System

This is my capstone project for CS781 at IIT Kanpur. The idea: law firms can't send their documents to hosted LLMs because of confidentiality, so they need something they can run on their own machines. This repo has everything I built for that, from scratch.

The project has seven phases. First we train a small language model on legal text (Phase 1). Then we take that model and teach it six different legal tasks (Phases 2 to 7).

| Phase | Task | What it does |
|-------|------|--------------|
| 1 | Legal language model | Learns legal language by predicting the next token |
| 2 | Named entity recognition | Finds things like dates, courts, judges and case numbers in a document |
| 3 | Rhetorical role prediction | Labels each sentence (fact, issue, ruling, etc.) |
| 4 | Court judgment prediction | Predicts the outcome of a case (0 or 1) |
| 5 | Statute identification | Finds which statutes are relevant to a case |
| 6 | Summarization | Writes a summary of a long legal document |
| 7 | Prior case retrieval | Finds earlier cases related to a given case |

## The rules we worked under

- Models between 50M and 300M parameters
- No pretrained models, only the corpus we were given
- Free Kaggle GPUs only, so everything is small and training time was tight

## The models

I trained four different models in Phase 1. All of them are around 118M parameters and share one tokenizer.

- **dec**: a plain decoder-only transformer (the baseline)
- **enc**: an encoder-decoder transformer
- **swa**: a decoder with sliding window attention (plus RoPE and RMSNorm)
- **mla**: a decoder with multi-head latent attention (plus ALiBi and RMSNorm)

The tokenizer is a byte-level BPE tokenizer written from scratch and trained on the Phase 1 corpus (vocab size about 50260).

For Phases 2 to 7, each model gets fine-tuned in a few different ways:

- **Full fine-tuning**: update every weight
- **LoRA** and **LoRMA**: only train small add-on adapter layers
- **Knowledge distillation (KD)**: train a smaller student (half the layers) to copy a bigger teacher

## Repo layout

```
phase1/
  tokenizer/   train the tokenizer and build the training data
  dec/ enc/ mla/ swa/    pretraining + inference for each model
phase2/ ... phase7/
  data prep notebook(s)
  dec/ enc/ mla/ swa/    trainers + inference notebooks
```

Inside each model folder you'll see files named like `full_trainer.ipynb`, `lora_trainer.ipynb`, `kd_trainer.ipynb` and so on. Each trainer has a matching `*_inference.ipynb` that makes the predictions for submission. `model_arch.py` has the model code, and `bpetokenizer.py` is the tokenizer.

## How to run things

I tested code on kaggle. So the code assumes availability of cuda at many places.

For every phase:

1. Run the data prep notebook for that phase.
2. It produces some pickle files. Copy those into the same folder as the trainer (or inference) notebook you want to run.
3. Open the trainer notebook and run it all the way through.
4. Do the same with the matching inference notebook to get your predictions.

Data provided by course is not included in the repository

### Phase 1: pretraining

1. Go to `phase1/tokenizer/`.
2. Run `bpe-trainer.ipynb` to train the tokenizer.
3. Run `dataset-maker.ipynb` to turn the corpus into training data.
4. Copy the outputs into whichever model folder you're training (`dec`, `enc`, `mla` or `swa`).
5. Run `trainer.ipynb` there. Pretraining took me about 11 hours on a Kaggle GPU for 400M tokens, so keep Kaggle's 12 hour limit in mind.
6. Run the inference notebook to generate continuations for the test prompts.


### Phase 2: named entity recognition

Run `dataset-prep.ipynb`, copy the pickles to the model folder, then run the trainer and inference notebooks. Here the model tags each token, and then the tags get converted back into character spans. Because the scoring is very strict about exact character positions, a good chunk of the code is cleaning up those spans.

### Phase 3: rhetorical roles

There isn't a separate prep notebook here. Data loading is handled by `dataloading.py` and `RRDataset.py`. Just go into a model folder and run the trainer and inference notebooks. Each sentence is classified on its own.

### Phase 4: court judgment prediction

Run `data_prep.ipynb`, copy the pickles over, then train and run inference. Only the last 1024 tokens of each document are used, because the outcome is usually near the end.

### Phase 5: statute identification

Run `data_prep.ipynb`, copy the pickles over, then train and run inference. Long documents are cut into overlapping chunks, each chunk is scored, and the best score per statute across chunks is used for the document. The cutoffs for turning scores into yes/no are tuned on a dev split. This phase only has full fine-tuning.

### Phase 6: summarization

Run `data_prep.ipynb`, copy the pickles over, then train and run inference. The decoder models read the article and then write the summary in one sequence. At inference time, long documents are split into pieces, each piece gets a mini summary, and they get joined together. The encoder-decoder model works a bit differently (it reads the head and tail of the document).

### Phase 7: prior case retrieval

Run `data_prep.ipynb`, copy the pickles over, then train and run inference. Every document is cut into chunks, turned into a single vector, and cases are matched by cosine similarity. The top 7 candidates for each query get submitted.

## Quick summary of what we found

Scores on the Codabench leaderboard for our best runs:

| Phase | Score |
|-------|-------|
| 1 | 0.8284 |
| 2 | 0.3552 |
| 3 | 0.3771 |
| 4 | 0.6605 |
| 5 | 0.3179 |
| 6 | 0.8374 |
| 7 | 0.2983 |

A few things that stood out:

- Full fine-tuning usually won, but LoRA and LoRMA got close once adapters were put on **all** linear layers. Putting them only on the query and value matrices left a big gap.
- LoRA and LoRMA were basically tied.
- Smaller distilled models mostly did worse than their teachers. The exception was Phase 3, where the distilled student did best.
- The encoder-decoder model did worst almost everywhere. Plausible explanation is that pretraining was not enough for this model.
- Fine-tuning really matters. In Phase 7 the pretrained model with no fine-tuning scored 0.104, compared to 0.298 after fine-tuning.

If I did this again I'd put more of the compute into Phase 1, since everything else depends on it.