import spacy

# 1. Load the small English model
nlp = spacy.load("en_core_web_sm")

# 2. Define a sample text
text = "Barack Obama was born in Hawaii. He was elected President of the United States in 2008."

# 3. Process the text with the spaCy pipeline
doc = nlp(text)

# 4. Part-of-Speech Tagging
print("=== Part-of-Speech (POS) Tagging ===")
for token in doc:
    print(f"{token.text}\t{token.pos_}\t{token.dep_}")

print("\n=== Named Entity Recognition (NER) ===")
for ent in doc.ents:
    print(f"{ent.text}\t{ent.label_}")
