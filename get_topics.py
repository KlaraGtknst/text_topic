import json
import pandas as pd

path = '/Users/klara/Developer/Uni/WiSe2425/text_topic/results/fca/topic_words_01_16_25.json'

topic_ids = [56, 36, 34, 118, 144, 94, 54, 58, 222, 223, 224, 225, 226]

terms2topic = {}
with open(path, 'r') as f:
    data = json.load(f)
    for topic_id in topic_ids:
        if topic_id in topic_ids and str(topic_id) in data:
            terms2topic[topic_id] = data[str(topic_id)]


# Convert to DataFrame and transpose
df = pd.DataFrame.from_dict(terms2topic).T

# Save as CSV
output_path = '/Users/klara/Developer/Uni/WiSe2425/text_topic/results/fca/topic_words_01_16_25.csv'
df.to_csv(output_path, index=True)