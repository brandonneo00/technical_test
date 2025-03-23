import pandas as pd
import requests, os


# df = pd.read_csv("../data/common_voice/cv-valid-dev/cv-valid-dev.csv")
df = pd.read_csv("../data/common_voice/cv-valid-dev.csv")

# /Users/brandyscrub/Documents/NUS/Y4S2/HTX/xData/new2/technical_test/asr/cv-decode.py
# /Users/brandyscrub/Documents/NUS/Y4S2/HTX/xData/new2/technical_test/data/common_voice/cv-valid-dev.csv

transcriptions = []

for _, row in df.iterrows():
    # file_path = os.path.join("cv-valid-dev", row['filename'])
    # file_path = os.path.join("../data/common_voice/cv-other-dev", df.loc[0]['filename'])
    file_path = os.path.join("../data/common_voice/cv-valid-dev", row['filename'])

    with open(file_path, 'rb') as f:
        response = requests.post("http://localhost:8001/asr", files={"file": f})
    if response.status_code == 200:
        data = response.json()
        transcriptions.append(data.get("transcription", ""))
    else:
        transcriptions.append("Error")

df['generated_text'] = transcriptions
# df.to_csv("cv-valid-dev/cv-valid-dev.csv", index=False)
df.to_csv("../data/common_voice/cv-valid-dev/cv-valid-dev.csv", index=False)

