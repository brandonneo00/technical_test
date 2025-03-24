# Assessment README

### For Task 2e:
- Ensure that you have Docker installed
- In your terminal, change directory to the asr folder (where the Dockerfile is located)
- Run `docker build -t asr-api .` in the terminal
- Run `docker run -d --name asr-api -p 8001:8001 asr-api` in the terminal
- To test the asr_api end point, you may use postman to test and attach a sample MP3 under `form-data` in Body, do not use `binary` to attach a sample MP3 in Body  

### For Task 3
- Apologies for not being able to complete Task 3 successfully due to time constraints, lack of GPU and some pytorch incompatibility issues with MacOS. I tried to explore Google Colab but was unable to successfully complete it within the stipulated time frame. 
- The subsequent tasks (task 4 and 5) were thus under the assumption of the 'base' model instead in order to attempt them 
1. Preprocessing was first done on the audio files by resampling them to 16kHz so as to fit the sampling rate of the audio data taken in by the model. Due to significant RAM constraints during training (limited by 12GB of RAM I had available), I filtered the training audio data to only include those that are 6s and below. This eased the RAM usage and made training possible. Finally, preprocessing was also done on the transcription by making them capitalized, to be compatible with the "Wav2Vec2CTCTokenizer.from_pretrained("facebook/wav2vec2-large-960h")" tokenizer used that contained only capital letters in its dictionary.
2. The unmodified Wav2Vec2CTCTokenizer.from_pretrained("facebook/wav2vec2-large-960h") tokenizer was used.
3. The Wav2Vec2FeatureExtractor feature extractor was used with the parameters: feature_size=1, sampling_rate=16000, padding_value=0.0, do_normalize=True. The feature size is set to 1 as the input is a single vector feature. The padding value of 0 indicates padding the audio data with silence. Normalization makes the statistics of all audio input data equal to help with traning. 
4. The general workflow is as follows:
- Resample audios into a new folder
- Generate a new cv-valid-train_filter.csv from cv-valid-train which only contains data where audio is <= 6s.
- A AudioTextDataset class was created which takes in the cv-valid-train_filtered.csv or cv-valid-test.csv filepath and returns either train_dataset, val_dataset or test_dataset (performing train-val split) that are each an array of dictionaries storing the feature extracted audio data labeled "input_values" and tokenized transcript labeled "labels"
- A DataCollatorCTCWithPadding is also created to feed the training data into the model in batches, padding the audio and transcript data to the same length beforehand.
- The traning arguments are set as follows:
```
training_args = TrainingArguments(
    output_dir="/content/drive/MyDrive/HTX/wav2vec2-finetuned-cv",
    evaluation_strategy="steps",
    save_steps=500,
    eval_steps=2000,
    learning_rate=3e-4,
    per_device_train_batch_size=8,
    num_train_epochs=1,
    warmup_steps=500,
    #gradient_accumulation_steps = 8,
    logging_dir="./logs",
    fp16=True,  # Mixed precision for faster training on GPUs
)
```
which runs the whole traning set over 1 epoch and evaluates against the validation set every 2000 steps. THe batch size is set to a relatively low value of 8 and gradient_accumulation used so as to ease the RAM usage. fp16 is set to true to reduce the computational resource while maintaining sufficient precision.
- Finally the trainer is initialized as such:
```
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    tokenizer=processor.feature_extractor,
    data_collator=data_collator,
    callbacks=[ClearCacheCallback()]
)
```
which contains a callback that empties the cache every 500 steps to reduce RAM usage. And the model is trained.

### For Task 3c
- The overall performance metric for ASR was chosen to be the Word Error Rate (WER) where we compare the model's predictions to the target text transcripts. Another secondary metric that can be considered is the Character Error Rate (CER) where the comparison between the model's transcription to the ground truth text happens on a character-by-character basic. 
- Both WER and CER are useful metrics as the main goal of ASR is to produce accurate text from speech and they provide a clear, quantitative measure of transcription quality. Furthermore, they are language-agnostic and task-adaptable as well. 

### For Task 4
1. An expected further improvement of model performance would come from utilizing the full traning dataset if we have sufficient RAM resources. Alternatively, a lower batch count during training and higher gradient_accumulation_steps can compenstate for the RAM usage with more traning time.
2. Data augmentation could be performed on the audio data, e.g. through low pass filtering, varying the tempo, or introducing artifects like echo to artificially increase the traning dataset.
3. The additional metadeta that came with the datasets: age, gender, accent could be incoporated into the model by adding additional layers on top of the base model for the extra inputs. These information can help the model make better predictions.

### For Task 5b
- It seems like there is some version issue with the InstructorEmbedding package required for this task. Also tried experimenting on Google Colab but also faced similar initialisation issues as seen in the notebook in this repo

#### Due to the size of the common_voice dataset being too large to push into Git, you may have to download the common_voice dataset on your own and the common_voice folder under a data folder in the main directory




