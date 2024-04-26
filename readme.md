# Whisper Hindi Evaluation Tool

This project evaluates the performance of a Hindi Automatic Speech Recognition (ASR) model using the Indic Whisper framework and JiWER library to calculate the Word Error Rate.


## Requirements

`Dataset:` Follow this link to download the [Katbath dataset](https://objectstore.e2enetworks.net/indic-asr-public/indicwhisper/vistaar/kathbath.zip). 
> Only Hindi audio data is used in this project.

`Indic Whisper:` The installation requires the latest version of the JAX package on your device. Follow the official [JAX Installation Guide](https://github.com/google/jax#installation).

Next, to Install Whisper JAX:
```
pip install git+https://github.com/sanchit-gandhi/whisper-jax.git
```

Now, to implement Indic Whisper:
```python
from whisper_jax import FlaxWhisperForConditionalGeneration, FlaxWhisperPipline
import jax.numpy as jnp

pipeline = FlaxWhisperPipline('parthiv11/indic_whisper_nodcil', dtype=jnp.bfloat16)
transcript = pipeline('sample.mp3')
```

`Reuqired Libraries:` JiWER, Pandas
```
pip install jiwer pandas
```


### Acknowledgemet

- [Ai4Bharat Vistaar](https://github.com/AI4Bharat/vistaar?tab=readme-ov-file)
- [Indic Whisper With JAX (more faster)](https://huggingface.co/parthiv11/indic_whisper_nodcil)
- [JAX Guide (Optimised OpenAI's Whisper Model)](https://github.com/sanchit-gandhi/whisper-jax?tab=readme-ov-file)
