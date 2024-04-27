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
data = []
for index, row in df.iterrows():
    print(f"\n{index}")
    reference = row['transcript']
    result = pipeline(row['file_path'])
    res = evaluate(result["text"], reference)
    data.append({"file_path": row['file_path'], 'transcript': result['text'], "WER": res})

average = sum(float(record['WER']) for record in data) / len(data)
data.append({'file_path': 'Total', 'WER': f"{average:.2f}"})

df2 = pd.DataFrame(data)
df2.to_csv("output.csv", index=False)
