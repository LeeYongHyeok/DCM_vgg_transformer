# Audio-visual speech recognition based on DCM
this repo is implementing AVSR task in `Fairseq==0.8.0` toolkit.

1. The dependencies are noticed in `conda_env.yml` file.
2. Arguments about `train` or `inference` same with `speech_recognition` example in the original `Fairseq` toolkit.
3. The model is composed about three blocks. 1) `self-attention transformer based modality encoder`, 2) `dual-cross modality attention layer` and 3) `transformer based attention decoder`.
4. The mel-filterbank audio features and pre-trained CNN video features are fed in the model, then the model creates character-based sentence.
5. `WER` and `CER` calculated by `Sclite` package using prediction and ground-truth sentences.
