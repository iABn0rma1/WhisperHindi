from whisper_jax import FlaxWhisperPipline
import jax.numpy as jnp
pipeline = FlaxWhisperPipline('parthiv11/indic_whisper_nodcil', dtype=jnp.bfloat16)

import jiwer
def evaluate(hyp, ref):
    transforms = jiwer.Compose(
        [
            jiwer.ExpandCommonEnglishContractions(),
            jiwer.RemoveEmptyStrings(),
            jiwer.ToLowerCase(),
            jiwer.RemoveMultipleSpaces(),
            jiwer.Strip(),
            jiwer.RemovePunctuation(),
            jiwer.ReduceToListOfListOfWords(),
        ]
    )
    wer = jiwer.wer(
                ref,
                hyp,
                truth_transform=transforms,
                hypothesis_transform=transforms,
    )
    print(f"Transcription: {hyp}")
    print(f"Reference: {ref}")
    print(f"WER: {wer}")
    return wer

import pandas as pd
df = pd.read_csv("kathbath/hindi/test/bucket.csv")
WERs = []
for index, row in df.iterrows():
    print(index)
    reference = row['transcript']
    result = pipeline(row['file_path'])
    WERs.append(evaluate(result["text"], reference))
    print("\n")
print(f"WER on Kathbath Dataset: {(sum(WERs) / len(WERs))*100}")