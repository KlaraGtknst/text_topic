import logging
import spacy
import constants
from utils.logging_utils import get_date, init_debug_config
from utils.os_manipulation import exists_or_create


class NamedEntityRecognition:
    def __init__(self, on_server: bool = True):
        init_debug_config(log_filename='named_entity_recognition_', on_server=on_server)
        self.nlp = spacy.load("en_core_web_sm")

    def get_named_entities(self, text: str):
        """
        Returns named entities from a text.
        :param text: Text to analyze as string
        :return: List of entities and their categories in the format (entity, category)
        """
        doc = self.nlp(text)
        named_entities = []
        for ent in doc.ents:
            named_entities.append((ent.text, ent.label_))
        logging.info(f"Obtained named entities")
        return named_entities

    def get_named_entities_from_subset(self, text: str, subset_categories: list[str]):
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

    def get_entities_from_named_entity_list(self, named_entities: list[tuple[str, str]],
                                            subset_categories: list[str] = []):
        """
        Returns a list of entities from a list of named entities.
        :param named_entities: List of named entities
        :param subset_categories: List of named entity categories to consider; if empty, all categories are considered
        :return: List of entities
        """
        return [ent[0] for ent in named_entities] if subset_categories == [] else \
            [ent[0] for ent in named_entities if ent[1] in subset_categories]

    def get_categories_from_named_entity_list(self, named_entities: list[tuple[str, str]]):
        """
        Returns a list of categories from a list of named entities.
        :param named_entities: List of named entities
        :return: List of categories
        """
        return [ent[1] for ent in named_entities]

    def get_named_entities_dictionary(self, text: str):
        """
        Returns a dictionary of named entities.
        :param text: Text to analyze as string
        :return: Dictionary of named entities
        """
        try:
            doc = self.nlp(text)
            named_entities = {}
            for ent in doc.ents:
                if ent.label_ not in named_entities:
                    named_entities[ent.label_] = []
                named_entities[ent.label_].append(ent.text)
            logging.info(f"Obtained named entities dictionary")
            return named_entities
        except Exception as e:  # eg. UnicodeEncodeError
            return str(e)

# if __name__ == "__main__":
#     ner = NamedEntityRecognition()
#     sample_text = "Apple is a big company in the USA, California, Silicon valley."
#     print("sample text:", sample_text)
#     named_entities = ner.get_named_entities(text=sample_text)
#     print("all named entities:", named_entities)
#     entity_categories_to_consider = ['PERSON', 'ORG']
#     print(f"named entities for list of categories={entity_categories_to_consider}:",
#           ner.get_named_entities_from_subset(text=sample_text, subset_categories=entity_categories_to_consider))
#
#     print(f"entities for list of categories={entity_categories_to_consider}: ",
#           ner.get_entities_from_named_entity_list(named_entities, subset_categories=entity_categories_to_consider))
#
#     print(f"categories of named entities: ", ner.get_categories_from_named_entity_list(named_entities))
#
#     print(f"named entities dictionary: ", ner.get_named_entities_dictionary(sample_text))
