import spacy


class NamedEntityRecognition:
    def __init__(self):
        self.nlp = spacy.load("en_core_web_sm")

    def get_named_entities(self, text:str):
        doc = self.nlp(text)
        named_entities = []
        for ent in doc.ents:
            named_entities.append((ent.text, ent.label_))
        return named_entities

    def get_named_entities_from_subset(self, text:str, subset:list[str]):
        """
        Returns named entities from a subset of named entity types.
        :param text: Text to analyze as string
        :param subset: List of named entity categories to consider
        :return: List of entities and their categories
        """
        doc = self.nlp(text)
        named_entities = []
        for ent in doc.ents:
            if ent.label_ in subset:
                named_entities.append((ent.text, ent.label_))
        return named_entities

if __name__ == "__main__":
    ner = NamedEntityRecognition()
    print(ner.get_named_entities("Apple is a big company in the USA."))
    print(ner.get_named_entities_from_subset("Apple is a big company in the USA.", ['ORG']))