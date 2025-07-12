#!/bin/bash

# Usage: ./prepare.sh [direction] [morfessor]
# Direction: hsb-de (default) or de-hsb
# Morfessor: Optional, apply Morfessor segmentation before BPE

# Parse arguments
DIRECTION=${1:-hsb-de}
USE_MORFESSOR_ARG=${2:-}

# Validate direction argument
if [[ "$DIRECTION" != "hsb-de" && "$DIRECTION" != "de-hsb" ]]; then
  echo "Error: Invalid direction '$DIRECTION'. Use 'hsb-de' or 'de-hsb'."
  exit 1
fi

# Set source and target languages based on direction
if [[ "$DIRECTION" == "hsb-de" ]]; then
  src=hsb
  tgt=de
else
  src=de
  tgt=hsb
fi

# Configuration
MOSES_SCRIPTS=./moses_scripts # Set this to your Moses installation
OUTPUT_DIR="./dataset"        # Output directory for all processed files
max_len=100 # Maximum sentence length (adjust as needed)

# Use Morfessor preprocessing (optional)
USE_MORFESSOR=${USE_MORFESSOR_ARG:-}
if [[ "$USE_MORFESSOR" == "morfessor" ]]; then
  USE_MORFESSOR=true
else
  USE_MORFESSOR=false
fi

# BPE configuration
BPE_OPERATIONS=16000 # Number of BPE merge operations

# Morfessor configuration
MORFESSOR_ALGORITHM="recursive" # Options: recursive, viterbi, baseline
MORFESSOR_DAMPING=0.01          # Damping parameter for Morfessor
MORFESSOR_ALPHA=1.0             # Alpha parameter for Morfessor

# Validate morfessor argument
if [[ -n "$USE_MORFESSOR_ARG" && "$USE_MORFESSOR_ARG" != "morfessor" ]]; then
  echo "Error: Invalid morfessor argument '$USE_MORFESSOR_ARG'. Use 'morfessor' or leave empty for BPE-only."
  exit 1
fi

echo "Processing direction: $DIRECTION ($src -> $tgt)"
if [[ "$USE_MORFESSOR" == "true" ]]; then
  echo "Using Morfessor + BPE segmentation pipeline"
else
  echo "Using BPE-only segmentation pipeline"
fi

# Directory structure - add direction to final binary dataset
ORIGINAL_DIR="$OUTPUT_DIR/original"
MOSES_DIR="$OUTPUT_DIR/output_moses"
if [[ "$USE_MORFESSOR" == "true" ]]; then
  MORFESSOR_DIR="$OUTPUT_DIR/output_morfessor"
  BPE_DIR="$OUTPUT_DIR/output_morfessor_bpe"
  DATA_BIN_DIR="$OUTPUT_DIR/fairseq_morfessor_bpe_${DIRECTION}"
else
  BPE_DIR="$OUTPUT_DIR/output_bpe"
  DATA_BIN_DIR="$OUTPUT_DIR/fairseq_bpe_${DIRECTION}"
fi

# Split files (in original directory)
train_src="$ORIGINAL_DIR/train.${src}"
train_tgt="$ORIGINAL_DIR/train.${tgt}"
dev_src="$ORIGINAL_DIR/dev.${src}"
dev_tgt="$ORIGINAL_DIR/dev.${tgt}"
test_src="$ORIGINAL_DIR/test.${src}"
test_tgt="$ORIGINAL_DIR/test.${tgt}"

# Create output directories if they don't exist
if [[ "$USE_MORFESSOR" == "true" ]]; then
  mkdir -p "$ORIGINAL_DIR" "$MOSES_DIR" "$MORFESSOR_DIR" "$BPE_DIR" "$DATA_BIN_DIR"
else
  mkdir -p "$ORIGINAL_DIR" "$MOSES_DIR" "$BPE_DIR" "$DATA_BIN_DIR"
fi

# Check if split files exist
for split in train dev test; do
  for lang in $src $tgt; do
    file="$ORIGINAL_DIR/${split}.${lang}"
    if [ ! -f "$file" ]; then
      echo "Error: Split file $file not found!"
      exit 1
    fi
  done
done

# Check dependencies
if ! command -v subword-nmt &>/dev/null; then
  echo "Error: subword-nmt not found! Install with: pip install subword-nmt"
  exit 1
fi

if [[ "$USE_MORFESSOR" == "true" ]] && ! python -c "import morfessor" &>/dev/null; then
  echo "Error: morfessor not found! Install with: pip install morfessor"
  exit 1
fi

# Check if fairseq is installed
if ! command -v fairseq-preprocess &>/dev/null; then
  echo "Error: fairseq not found! Install with: pip install fairseq"
  exit 1
fi

# Clean up any previous runs in output directories
rm -f "$MOSES_DIR"/*
rm -f "$BPE_DIR"/*
if [[ "$USE_MORFESSOR" == "true" ]]; then
  # For morfessor, preserve existing models but clean other files
  find "$MORFESSOR_DIR" -type f ! -name "morfessor_model.*.bin" -delete 2>/dev/null || true
fi

if [[ "$USE_MORFESSOR" == "true" ]]; then
  echo "Starting preprocessing for ${src}-${tgt} with Morfessor + BPE..."
else
  echo "Starting preprocessing for ${src}-${tgt} with BPE..."
fi

# Step 1: Process each split separately through Moses pipeline
echo "Step 1-5: Processing train/dev/test splits through Moses pipeline..."

for split in train dev test; do
  echo "Processing $split split..."

  # Step 1: Normalize punctuation
  for lang in $src $tgt; do
    input_file="$ORIGINAL_DIR/${split}.${lang}"
    $MOSES_SCRIPTS/tokenizer/normalize-punctuation.perl -l $lang <"$input_file" >"$MOSES_DIR/${split}.norm.$lang"
  done

  # Step 2: Tokenize
  for lang in $src $tgt; do
    $MOSES_SCRIPTS/tokenizer/tokenizer.perl -a -l $lang <"$MOSES_DIR/${split}.norm.$lang" >"$MOSES_DIR/${split}.tok.$lang"
  done

  # Step 3: Clean corpus (only for train split to avoid removing dev/test data)
  if [ "$split" == "train" ]; then
    $MOSES_SCRIPTS/training/clean-corpus-n.perl "$MOSES_DIR/${split}.tok" $src $tgt "$MOSES_DIR/${split}.clean" 1 $max_len
  else
    # For dev/test, just copy tok files to clean files (no filtering)
    cp "$MOSES_DIR/${split}.tok.$src" "$MOSES_DIR/${split}.clean.$src"
    cp "$MOSES_DIR/${split}.tok.$tgt" "$MOSES_DIR/${split}.clean.$tgt"
  fi
done

# STEP 1.b für Mnolingual Data
MONO_HSB_FILE="/Users/sebi/bpc_dev/witaj_monolingual.hsb"

MONO_HSB_TOKENIZED="$MOSES_DIR/mono.tok.hsb"

#-> Normalize punctuation 
$MOSES_SCRIPTS/tokenizer/normalize-punctuation.perl -l hsb < "$MONO_HSB_FILE" > "$MOSES_DIR/mono.norm.hsb"

#-> Tokenize
$MOSES_SCRIPTS/tokenizer/tokenizer.perl -a -l hsb < "$MOSES_DIR/mono.norm.hsb" > "$MONO_HSB_TOKENIZED"

# RNN Tagger
# Tagger for specially German as well 
# Check for Parallel data 
#Step 3.b:
echo "Step3b: Applying RNNTagger for POS tagging..."

OUTPUT_DIR="./output"
RNNTAGGER_OUTPUT_DIR="$OUTPUT_DIR/output_rnntagger"
RNN_GERMAN_SCRIPT="/Users/sebi/Desktop/RNNTagger/cmd/rnn-tagger-german.sh"
RNN_UPPER_SORBIAN_SCRIPT="/Users/sebi/Desktop/RNNTagger/cmd/rnn-tagger-upper-sorbian.sh"
mkdir -p "$RNNTAGGER_OUTPUT_DIR"

for split in train dev test; do
  echo "Processing POS tagging for $split split..."

  for lang in $src $tgt; do
    INPUT_FILE="$MOSES_DIR/${split}.tok.${lang}"
    OUTPUT_FILE="$RNNTAGGER_OUTPUT_DIR/${split}.tagged.${lang}"

    if [[ "$lang" == "de" ]]; then
      echo "Tagging $split.$lang with RNNTagger ..."
      "$RNN_GERMAN_SCRIPT" "$INPUT_FILE" > "$OUTPUT_FILE" \
        || { echo "RNNTagger shell script failed for $split.$lang"; exit 1; }
    
    else
      echo "Tagging $split.$lang with RNNTagger ..."
      "$RNN_UPPER_SORBIAN_SCRIPT" "$INPUT_FILE" > "$OUTPUT_FILE" \
        || { echo "RNNTagger failed for $split.$lang"; exit 1; }

    fi
  done

  src_file="$RNNTAGGER_OUTPUT_DIR/${split}.tagged.${src}"
  tgt_file="$RNNTAGGER_OUTPUT_DIR/${split}.tagged.${tgt}"

  src_lines=$(wc -l < "$src_file")
  tgt_lines=$(wc -l < "$tgt_file")

  if [ "$src_lines" -ne "$tgt_lines" ]; then
    echo "ERROR: Line mismatch in POS-tagged $split split! $src: $src_lines vs $tgt: $tgt_lines"
    exit 1
  else
    echo "Line count check passed for $split: $src_lines lines"
  fi
  

done

# Step 4: Train truecaser on training data only
echo "Step 4: Training truecaser on training data..."
for lang in $src $tgt; do
  $MOSES_SCRIPTS/recaser/train-truecaser.perl -model "$MOSES_DIR/truecase-model.$lang" -corpus "$RNNTAGGER_OUTPUT_DIR/train.tagged.$lang"
  #$MOSES_SCRIPTS/recaser/train-truecaser.perl -model "$MOSES_DIR/truecase-model.$lang" -corpus "$MOSES_DIR/train.tok.$lang"
done

# Step 5: Apply truecasing to all splits
echo "Step 5: Applying truecasing to all splits..."
for split in train dev test; do
  for lang in $src $tgt; do
    $MOSES_SCRIPTS/recaser/truecase.perl -model "$MOSES_DIR/truecase-model.$lang" <"$RNNTAGGER_OUTPUT_DIR/${split}.tagged.$lang" >"$MOSES_DIR/${split}.$lang"
    #$MOSES_SCRIPTS/recaser/truecase.perl -model "$MOSES_DIR/truecase-model.$lang" <"$MOSES_DIR/${split}.clean.$lang" >"$MOSES_DIR/${split}.$lang"
  done
done

# Step 6: Learn Morfessor segmentation (if enabled)
if [[ "$USE_MORFESSOR" == "true" ]]; then
  echo "Step 6a: Learning Morfessor segmentation on training data..."

  # Check if training files are not empty
  if [ ! -s "$MOSES_DIR/train.$src" ] || [ ! -s "$MOSES_DIR/train.$tgt" ]; then
    echo "Error: Training files are empty! Check the Moses preprocessing step."
    exit 1
  fi

  # Learn Morfessor models for each language separately
  for lang in $src $tgt; do
    if [ -f "$MORFESSOR_DIR/morfessor_model.$lang.bin" ]; then
      echo "Morfessor model for $lang already exists, skipping training..."
    else
      if [ "$lang" == "hsb" ]; then
        echo "Training Morfessor model for $lang (with monolingual data)..."
        cat "$MOSES_DIR/train.tok.$lang" "$MONO_HSB_TOKENIZED" > "$MOSES_DIR/train_tok_plus_mono_tok.$lang"
        echo "train_tok_plus_mono_tok.$lang created..."
        morfessor-train \
          -s "$MORFESSOR_DIR/morfessor_model.$lang.bin" \
          -d ones \
          "$MOSES_DIR/train_tok_plus_mono_tok.$lang"
      else
        echo "Training Morfessor model for $lang..."
        morfessor-train \
          -s "$MORFESSOR_DIR/morfessor_model.$lang.bin" \
          -d ones \
          "$MOSES_DIR/train.tok.$lang"
      fi
    fi
  done
#MAYBE CHANGE -d ones to -d full or -d poisson/flat

  # Check if Morfessor models were created successfully
  for lang in $src $tgt; do
    if [ ! -f "$MORFESSOR_DIR/morfessor_model.$lang.bin" ]; then
      echo "Error: Morfessor model for $lang was not created!"
      exit 1
    fi
  done

  echo "Morfessor models trained successfully for both languages"

  # Apply Morfessor segmentation to all splits
  echo "Step 6b: Applying Morfessor segmentation to all splits..."
  for split in train dev test; do
    for lang in $src $tgt; do
      echo "Applying Morfessor segmentation to $split.$lang..."
      morfessor-segment \
        -l "$MORFESSOR_DIR/morfessor_model.$lang.bin" \
        --output-format-separator "@@ " \
        --output-newlines \
        --output-format "{analysis} " \
        "$MOSES_DIR/$split.$lang" \
        >"$MORFESSOR_DIR/$split.morfessor.$lang"
    done
  done
  MORFESSOR_INPUT_DIR="$MORFESSOR_DIR"
  MORFESSOR_EXT="morfessor"
else
  # No Morfessor preprocessing, use Moses output directly
  MORFESSOR_INPUT_DIR="$MOSES_DIR"
  MORFESSOR_EXT=""
fi

# Step 7: Learn BPE on training data (always applied)
echo "Step 7: Learning BPE on training data..."

# Check if training files are not empty
if [[ "$USE_MORFESSOR" == "true" ]]; then
  TRAIN_SRC_FILE="$MORFESSOR_DIR/train.morfessor.$src"
  TRAIN_TGT_FILE="$MORFESSOR_DIR/train.morfessor.$tgt"
else
  TRAIN_SRC_FILE="$MOSES_DIR/train.$src"
  TRAIN_TGT_FILE="$MOSES_DIR/train.$tgt"
fi

if [ ! -s "$TRAIN_SRC_FILE" ] || [ ! -s "$TRAIN_TGT_FILE" ]; then
  echo "Error: Training files are empty! Check the preprocessing steps."
  exit 1
fi

# Combine training data and learn BPE
cat "$TRAIN_SRC_FILE" "$TRAIN_TGT_FILE" >"$BPE_DIR/train_combined.txt"
echo "Combined training data has $(wc -l <"$BPE_DIR/train_combined.txt") lines"

# Learn BPE with minimum frequency threshold
subword-nmt learn-bpe -s $BPE_OPERATIONS --min-frequency 2 <"$BPE_DIR/train_combined.txt" >"$BPE_DIR/bpe.codes"

# Check if BPE codes were created successfully
if [ ! -s "$BPE_DIR/bpe.codes" ]; then
  echo "Error: BPE codes file is empty or was not created!"
  echo "This might happen with very small datasets. Try reducing BPE_OPERATIONS."
  exit 1
fi

echo "BPE codes learned successfully ($(wc -l <"$BPE_DIR/bpe.codes") operations)"

# Step 8: Apply BPE to all splits
echo "Step 8: Applying BPE to all splits..."

for split in train dev test; do
  for lang in $src $tgt; do
    if [[ "$USE_MORFESSOR" == "true" ]]; then
      INPUT_FILE="$MORFESSOR_DIR/$split.morfessor.$lang"
      OUTPUT_FILE="$BPE_DIR/$split.morfessor.bpe.$lang"
    else
      INPUT_FILE="$MOSES_DIR/$split.$lang"
      OUTPUT_FILE="$BPE_DIR/$split.bpe.$lang"
    fi
    subword-nmt apply-bpe -c "$BPE_DIR/bpe.codes" <"$INPUT_FILE" >"$OUTPUT_FILE"
  done
done

# Set file extension for fairseq preprocessing
if [[ "$USE_MORFESSOR" == "true" ]]; then
  FINAL_EXT="morfessor.bpe"
else
  FINAL_EXT="bpe"
fi

# Step 9: Create fairseq binary dataset
echo "Step 9: Creating fairseq binary dataset..."
fairseq-preprocess \
  --source-lang $src --target-lang $tgt \
  --trainpref "$BPE_DIR/train.$FINAL_EXT" \
  --validpref "$BPE_DIR/dev.$FINAL_EXT" \
  --testpref "$BPE_DIR/test.$FINAL_EXT" \
  --destdir "$DATA_BIN_DIR" \
  --workers 4

echo "Fairseq binary dataset created successfully!"

# Show statistics
echo "=== Preprocessing Statistics ==="
echo "=== Original Split Statistics ==="
echo "Training sentences: $(wc -l <"$ORIGINAL_DIR/train.$src")"
echo "Development sentences: $(wc -l <"$ORIGINAL_DIR/dev.$src")"
echo "Test sentences: $(wc -l <"$ORIGINAL_DIR/test.$src")"
echo ""
echo "=== After Moses Processing ==="
echo "Training sentences: $(wc -l <"$MOSES_DIR/train.$src")"
echo "Development sentences: $(wc -l <"$MOSES_DIR/dev.$src")"
echo "Test sentences: $(wc -l <"$MOSES_DIR/test.$src")"

echo "BPE operations: $BPE_OPERATIONS"
if [[ "$USE_MORFESSOR" == "true" ]]; then
  echo "Morfessor algorithm: $MORFESSOR_ALGORITHM"
  echo "Morfessor damping: $MORFESSOR_DAMPING"
  echo "Morfessor alpha: $MORFESSOR_ALPHA"
fi

# Optional: Clean up intermediate files (comment out if you want to keep them)
# rm -f "$MOSES_DIR"/*.norm.* "$MOSES_DIR"/*.tok.* "$MOSES_DIR"/*.clean.* "$SEGMENTATION_DIR"/train_combined.txt

echo ""
echo "Preprocessing complete!"
echo ""
echo "=== Directory Structure ==="
echo "$ORIGINAL_DIR/ - Original train/dev/test split files"
echo "$MOSES_DIR/ - Moses preprocessing output, train/dev/test splits"
if [[ "$USE_MORFESSOR" == "true" ]]; then
  echo "$MORFESSOR_DIR/ - Morfessor models and segmented files"
fi
echo "$BPE_DIR/ - BPE codes and final segmented files"
echo "$DATA_BIN_DIR/ - Fairseq binary dataset (ready for training)"
echo ""
echo "=== Key Files for NMT Training ==="
echo "Fairseq binary data: $DATA_BIN_DIR/"
echo "BPE codes: $BPE_DIR/bpe.codes"
if [[ "$USE_MORFESSOR" == "true" ]]; then
  echo "Morfessor models: $MORFESSOR_DIR/morfessor_model.$src.bin, $MORFESSOR_DIR/morfessor_model.$tgt.bin"
fi
echo "Truecaser models: $MOSES_DIR/truecase-model.$src, $MOSES_DIR/truecase-model.$tgt"
echo ""
if [[ "$USE_MORFESSOR" == "true" ]]; then
  echo "Ready for NMT training with Morfessor + BPE segmentation!"
else
  echo "Ready for NMT training with BPE segmentation!"
fi
