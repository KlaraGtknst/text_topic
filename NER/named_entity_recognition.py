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

    def get_named_entities_from_subset(self, text:str, subset_categories:list[str]):
        """
        Returns named entities from a subset of named entity types.
        :param text: Text to analyze as string
        :param subset_categories: List of named entity categories to consider
        :return: List of entities and their categories
        """
        doc = self.nlp(text)
        named_entities = []
        for ent in doc.ents:
            if ent.label_ in subset_categories:
                named_entities.append((ent.text, ent.label_))
        return named_entities

    def get_entities_from_named_entity_list(self, named_entities:list[tuple[str,str]], subset_categories:list[str]=[]):
        """
        Returns a list of entities from a list of named entities.
        :param named_entities: List of named entities
        :param subset_categories: List of named entity categories to consider; if empty, all categories are considered
        :return: List of entities
        """
        return [ent[0] for ent in named_entities] if subset_categories == [] else \
            [ent[0] for ent in named_entities if ent[1] in subset_categories]

    def get_categories_from_named_entity_list(self, named_entities:list[tuple[str,str]]):
        """
        Returns a list of categories from a list of named entities.
        :param named_entities: List of named entities
        :return: List of categories
        """
        return [ent[1] for ent in named_entities]

if __name__ == "__main__":
    ner = NamedEntityRecognition()
    sample_text = "Apple is a big company in the USA."
    print("sample text:", sample_text)
    print("all named entities:", ner.get_named_entities(sample_text))
    entity_categories_to_consider = ['PERSON', 'ORG']
    print(f"named entities for list of categories={entity_categories_to_consider}:",
          ner.get_named_entities_from_subset(text=sample_text, subset_categories=entity_categories_to_consider))

    named_entities = [('Apple', 'ORG'), ('USA', 'GPE'), ('John', 'PERSON')]
    print(f"entities for list of categories={entity_categories_to_consider}: ",
          ner.get_entities_from_named_entity_list(named_entities, subset_categories=entity_categories_to_consider))

    print(f"categories of named entities: ", ner.get_categories_from_named_entity_list(named_entities))